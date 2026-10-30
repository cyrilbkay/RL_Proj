#!/usr/bin/env python3
"""
trick_rl.py - A 4-player, 52-card, 13-cards-each trick-taking game plus a PPO
self-play agent that learns to play it.

Rules (same as the web interface):
  * Each player gets 13 random cards. A random player leads the first trick.
  * You must follow the led suit if you can; otherwise play any card.
  * Highest card of the led suit wins the trick (no trump). Winner leads next.
  * Goal: win as many tricks as possible (13 tricks per hand, mean = 3.25).

Install:   pip install torch numpy
Notebook (Colab/Jupyter): save this as trick_rl.py, then in a cell run
           from trick_rl import main; main(['train', '--iters', '300'])
           (or just run the file's cell; it trains with defaults)
Train:     python trick_rl.py train --iters 300
Evaluate:  python trick_rl.py eval  --model model.pt
Play:      python trick_rl.py play  --model model.pt
Export:    python trick_rl.py export --model model.pt --out weights.json

Technique: PPO with action masking, GAE, a network shared by all four seats,
and an opponent pool of past checkpoints to keep self-play stable.
Seats are encoded relative to the acting player (0 = me, 1 = next to play...).
"""
import argparse
import copy
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

SUITS = "SHDC"  # spades, hearts, diamonds, clubs (ASCII-safe for terminals)
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
OBS_DIM = 337
SEAT_NAMES = ["You", "West", "North", "East"]


def card_str(c):
    return RANKS[c % 13] + SUITS[c // 13]


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
class TrickGame:
    """Card id = suit * 13 + rank (rank 0 = '2' ... 12 = 'A')."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        deck = self.rng.permutation(52)
        self.hand = np.zeros((4, 52), dtype=bool)
        for p in range(4):
            self.hand[p, deck[p * 13:(p + 1) * 13]] = True
        self.played = np.zeros(52, dtype=bool)        # cards seen so far
        self.played_by = np.zeros((4, 52), dtype=bool)  # who played what
        self.voids = np.zeros((4, 4), dtype=bool)     # voids[p, suit]: p showed out
        self.won = np.zeros(4, dtype=np.int64)
        self.trick = []                               # list of (seat, card)
        self.trick_no = 0
        self.turn = int(self.rng.integers(4))
        return self.turn

    @property
    def done(self):
        return self.trick_no == 13

    def legal_mask(self):
        h = self.hand[self.turn]
        if self.trick:
            led = self.trick[0][1] // 13
            suited = np.zeros(52, dtype=bool)
            suited[led * 13:(led + 1) * 13] = h[led * 13:(led + 1) * 13]
            if suited.any():
                return suited
        return h.copy()

    def observe(self):
        """Observation for the player whose turn it is (relative seats)."""
        p = self.turn
        rel = [(p + r) % 4 for r in range(4)]  # rel[0] = me
        played_by = np.concatenate([self.played_by[q] for q in rel[1:]])
        voids = np.concatenate([self.voids[q] for q in rel[1:]])
        trick_vec = np.zeros(52)
        for _, c in self.trick:
            trick_vec[c] = 1
        led = np.zeros(4)
        if self.trick:
            led[self.trick[0][1] // 13] = 1
        pos = np.zeros(4)
        pos[len(self.trick)] = 1
        won = np.array([self.won[q] for q in rel]) / 13.0
        return np.concatenate([
            self.hand[p], self.played, played_by, voids,
            trick_vec, led, pos, won, [self.trick_no / 13.0],
        ]).astype(np.float32)

    def step(self, card):
        """Play a card for the current player. Returns the trick winner if the
        trick just completed, else None."""
        p = self.turn
        if not self.hand[p, card]:
            raise ValueError("card not in hand")
        if self.trick:
            led = self.trick[0][1] // 13
            if card // 13 != led:
                self.voids[p, led] = True
        self.hand[p, card] = False
        self.played[card] = True
        self.played_by[p, card] = True
        self.trick.append((p, card))
        if len(self.trick) < 4:
            self.turn = (p + 1) % 4
            return None
        led = self.trick[0][1] // 13
        winner = max((t for t in self.trick if t[1] // 13 == led),
                     key=lambda t: t[1] % 13)[0]
        self.won[winner] += 1
        self.trick_no += 1
        self.trick = []
        self.turn = winner
        return winner


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, 52)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        h = self.body(obs)
        logits = self.pi(h).masked_fill(~mask, -1e9)  # action masking
        return logits, self.v(h).squeeze(-1)


@torch.no_grad()
def act_batch(net, obs, mask, greedy=False):
    logits, v = net(torch.as_tensor(obs), torch.as_tensor(mask))
    dist = Categorical(logits=logits)
    a = logits.argmax(-1) if greedy else dist.sample()
    return a.numpy(), dist.log_prob(a).numpy(), v.numpy()


# ----------------------------------------------------------------------------
# Agents (used for evaluation and play)
# ----------------------------------------------------------------------------
class RandomAgent:
    def act(self, env):
        return int(np.random.choice(np.flatnonzero(env.legal_mask())))


class HeuristicAgent:
    """Lead the highest card; follow by winning cheaply if possible,
    otherwise dump the lowest card."""

    def act(self, env):
        legal = np.flatnonzero(env.legal_mask())
        ranks = legal % 13
        if not env.trick:
            return int(legal[np.argmax(ranks)])
        led = env.trick[0][1] // 13
        best = max(c % 13 for _, c in env.trick if c // 13 == led)
        winners = [c for c in legal if c // 13 == led and c % 13 > best]
        if winners:
            return int(min(winners, key=lambda c: c % 13))
        return int(legal[np.argmin(ranks)])


class NetAgent:
    def __init__(self, net, greedy=True):
        self.net, self.greedy = net, greedy

    def act(self, env):
        a, _, _ = act_batch(self.net, env.observe()[None],
                            env.legal_mask()[None], self.greedy)
        return int(a[0])


def evaluate(agent, opponent, games=400, seed=123):
    """Average tricks won by `agent` against 3 copies of `opponent`.
    The agent rotates through all seats. Baseline (equal skill) = 3.25."""
    env = TrickGame(seed)
    total = 0
    for g in range(games):
        seat = g % 4
        env.reset()
        while not env.done:
            who = agent if env.turn == seat else opponent
            env.step(who.act(env))
        total += env.won[seat]
    return total / games


# ----------------------------------------------------------------------------
# PPO self-play
# ----------------------------------------------------------------------------
def rollout(envs, net, pool, pool_prob, rng, gamma=1.0, lam=0.95):
    """Play one hand in every env (in lockstep), return a training batch.
    Reward per trick: +1 for the winner, minus 0.25 for everyone (zero-sum)."""
    E, T = len(envs), 52
    obs_b = np.zeros((E, T, OBS_DIM), np.float32)
    mask_b = np.zeros((E, T, 52), bool)
    act_b = np.zeros((E, T), np.int64)
    logp_b = np.zeros((E, T), np.float32)
    val_b = np.zeros((E, T), np.float32)
    seat_b = np.zeros((E, T), np.int64)
    rew_b = np.zeros((E, T), np.float32)

    learner = np.ones((E, 4), bool)   # which seats generate training data
    opp_idx = np.full(E, -1)          # which pool checkpoint plays the others
    for i, env in enumerate(envs):
        env.reset()
        if pool and rng.random() < pool_prob:
            learner[i] = False
            learner[i, rng.integers(4)] = True
            opp_idx[i] = rng.integers(len(pool))

    ar = np.arange(E)
    for t in range(T):
        seats = np.array([e.turn for e in envs])
        obs = np.stack([e.observe() for e in envs])
        mask = np.stack([e.legal_mask() for e in envs])
        acts = np.zeros(E, np.int64)
        logp = np.zeros(E, np.float32)
        val = np.zeros(E, np.float32)

        is_l = learner[ar, seats]
        groups = {-1: np.flatnonzero(is_l)}
        for k in set(int(x) for x in opp_idx[~is_l]):
            groups[k] = np.flatnonzero((~is_l) & (opp_idx == k))
        for k, idx in groups.items():
            if len(idx) == 0:
                continue
            a, lp, v = act_batch(net if k == -1 else pool[k], obs[idx], mask[idx])
            acts[idx], logp[idx], val[idx] = a, lp, v

        obs_b[:, t], mask_b[:, t], act_b[:, t] = obs, mask, acts
        logp_b[:, t], val_b[:, t], seat_b[:, t] = logp, val, seats

        for i, env in enumerate(envs):
            w = env.step(int(acts[i]))
            if w is not None:  # trick finished: reward the 4 plays of this trick
                for j in range(t - 3, t + 1):
                    rew_b[i, j] = float(seat_b[i, j] == w) - 0.25

    # GAE on each learner seat's 13-step trajectory
    O, M, A, LP, ADV, RET = [], [], [], [], [], []
    for i in range(E):
        for p in range(4):
            if not learner[i, p]:
                continue
            idx = np.flatnonzero(seat_b[i] == p)
            r, v = rew_b[i, idx], val_b[i, idx]
            adv = np.zeros(len(idx), np.float32)
            last = 0.0
            for k in reversed(range(len(idx))):
                nv = v[k + 1] if k + 1 < len(idx) else 0.0
                delta = r[k] + gamma * nv - v[k]
                last = delta + gamma * lam * last
                adv[k] = last
            O.append(obs_b[i, idx]); M.append(mask_b[i, idx])
            A.append(act_b[i, idx]); LP.append(logp_b[i, idx])
            ADV.append(adv); RET.append(adv + v)
    cat = lambda xs: torch.as_tensor(np.concatenate(xs))
    return cat(O), cat(M), cat(A), cat(LP), cat(ADV), cat(RET)


def ppo_update(net, opt, batch, epochs=4, minibatch=1024, clip=0.2,
               vf_coef=0.5, ent_coef=0.01):
    obs, mask, act, old_logp, adv, ret = batch
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(obs)
    stats = {}
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, minibatch):
            ix = perm[s:s + minibatch]
            logits, v = net(obs[ix], mask[ix])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[ix])
            ratio = (logp - old_logp[ix]).exp()
            a = adv[ix]
            pol_loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            val_loss = ((v - ret[ix]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pol_loss + vf_coef * val_loss - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            stats = {"pol": pol_loss.item(), "val": val_loss.item(), "ent": ent.item()}
    return stats


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = ActorCritic(args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    envs = [TrickGame(args.seed * 100003 + i) for i in range(args.envs)]
    pool = []
    rand, heur = RandomAgent(), HeuristicAgent()

    for it in range(1, args.iters + 1):
        t0 = time.time()
        batch = rollout(envs, net, pool, args.pool_prob if pool else 0.0, rng)
        stats = ppo_update(net, opt, batch)

        if it % args.snapshot_every == 0:
            snap = copy.deepcopy(net).eval()
            pool = (pool + [snap])[-args.pool_size:]

        if it % args.eval_every == 0 or it == 1:
            agent = NetAgent(net)
            vr = evaluate(agent, rand, args.eval_games)
            vh = evaluate(agent, heur, args.eval_games)
            print(f"iter {it:4d} | vs random {vr:.2f} | vs heuristic {vh:.2f} "
                  f"(3.25 = even) | ent {stats['ent']:.2f} val {stats['val']:.3f} "
                  f"| {time.time() - t0:.1f}s/iter", flush=True)
            torch.save({"hidden": args.hidden, "state": net.state_dict()}, args.save)
    torch.save({"hidden": args.hidden, "state": net.state_dict()}, args.save)
    print("saved", args.save)


# ----------------------------------------------------------------------------
# Load / export / play
# ----------------------------------------------------------------------------
def load_net(path):
    ck = torch.load(path, map_location="cpu")
    net = ActorCritic(ck["hidden"])
    net.load_state_dict(ck["state"])
    return net.eval()


def export_json(net, out):
    """Dump MLP weights so you can re-implement the policy in JavaScript.
    Note: the JS side must build the same 337-dim observation as
    TrickGame.observe() (relative seats, played_by, voids...)."""
    lin = lambda m: {"W": m.weight.tolist(), "b": m.bias.tolist()}
    data = {"obs_dim": OBS_DIM,
            "body": [lin(net.body[0]), lin(net.body[2])],
            "pi": lin(net.pi), "v": lin(net.v)}
    with open(out, "w") as f:
        json.dump(data, f)
    print("wrote", out)



def play_cli(net):
    """Modern Pygame interface: human vs 3 trained bots."""
    try:
        import pygame
    except ImportError:
        print("pygame is required: pip install pygame")
        return

    pygame.init()
    pygame.display.set_caption("TRICK  |  Card Table")

    W, H = 1200, 800
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    # ---------------------------------------------------------
    # COLORS
    # ---------------------------------------------------------
    BG = (15, 20, 29)
    PANEL = (25, 33, 44)
    PANEL_LIGHT = (35, 45, 57)
    TABLE = (19, 91, 68)
    TABLE_LIGHT = (25, 111, 81)
    TABLE_DARK = (12, 65, 49)

    WHITE = (245, 247, 250)
    MUTED = (155, 169, 181)
    GOLD = (238, 190, 92)
    GREEN = (73, 196, 137)
    RED = (230, 91, 99)
    BLACK = (24, 29, 36)

    # ---------------------------------------------------------
    # FONTS
    # ---------------------------------------------------------
    def font(size, bold=False):
        return pygame.font.SysFont("arial", size, bold=bold)

    f_title = font(30, True)
    f_subtitle = font(14)
    f_player = font(17, True)
    f_score = font(23, True)
    f_card = font(24, True)
    f_card_suit = font(22, True)
    f_small = font(13)
    f_button = font(15, True)
    f_result = font(42, True)
    f_result_score = font(21, True)

    # ---------------------------------------------------------
    # GAME CONFIGURATION
    # ---------------------------------------------------------
    CARD_W, CARD_H = 76, 108
    TRICK_W, TRICK_H = 83, 116

    SEATS = {
        0: "YOU",
        1: "WEST",
        2: "NORTH",
        3: "EAST"
    }

    SUIT_SYMBOLS = ["♠", "♥", "♦", "♣"]

    SUIT_COLORS = [
        (32, 40, 51),
        (218, 65, 75),
        (218, 65, 75),
        (32, 40, 51)
    ]

    # ---------------------------------------------------------
    # DRAWING HELPERS
    # ---------------------------------------------------------
    def draw_text(text, fnt, color, x, y, center=False):
        surf = fnt.render(str(text), True, color)

        if center:
            screen.blit(
                surf,
                (int(x - surf.get_width() / 2),
                 int(y - surf.get_height() / 2))
            )
        else:
            screen.blit(surf, (int(x), int(y)))

        return surf.get_rect(
            center=(int(x), int(y))
        ) if center else surf.get_rect(topleft=(int(x), int(y)))

    def rounded_rect(surface, color, rect, radius=12, width=0):
        pygame.draw.rect(
            surface, color, rect,
            width=width,
            border_radius=radius
        )

    def draw_panel(rect, color=PANEL, radius=16):
        rounded_rect(screen, (8, 12, 18), rect.move(0, 4), radius)
        rounded_rect(screen, color, rect, radius)

    def draw_button(rect, text, hovered=False, primary=False):
        color = GREEN if primary else PANEL_LIGHT

        if hovered:
            color = (
                min(color[0] + 18, 255),
                min(color[1] + 18, 255),
                min(color[2] + 18, 255)
            )

        rounded_rect(screen, color, rect, 10)

        draw_text(
            text, f_button, WHITE,
            rect.centerx, rect.centery, True
        )

    # ---------------------------------------------------------
    # CARD RENDERING
    # ---------------------------------------------------------
    card_cache = {}

    def draw_card(card, x, y, w, h, hover=False,
                  dim=False, outline=None):

        key = (card, w, h)

        if key not in card_cache:
            surf = pygame.Surface((w + 8, h + 8), pygame.SRCALPHA)

            # Shadow
            rounded_rect(
                surf, (0, 0, 0, 75),
                pygame.Rect(4, 5, w, h), 10
            )

            # Card background
            rounded_rect(
                surf, (249, 248, 243),
                pygame.Rect(0, 0, w, h), 10
            )

            pygame.draw.rect(
                surf, (205, 207, 208),
                (0, 0, w, h), 1,
                border_radius=10
            )

            suit = card // 13
            rank = card % 13
            color = SUIT_COLORS[suit]

            # Top-left rank and suit
            draw_rank = font(max(16, int(w * 0.29)), True)
            draw_suit = font(max(15, int(w * 0.24)), True)

            rank_surf = draw_rank.render(RANKS[rank], True, color)
            suit_surf = draw_suit.render(SUIT_SYMBOLS[suit], True, color)

            surf.blit(rank_surf, (8, 5))
            surf.blit(suit_surf, (8, 29))

            # Center suit
            center_font = font(int(w * 0.48), True)
            center = center_font.render(
                SUIT_SYMBOLS[suit], True, color
            )

            surf.blit(
                center,
                (
                    (w - center.get_width()) // 2,
                    (h - center.get_height()) // 2
                )
            )

            # Bottom-right inverted corner
            bottom = pygame.transform.rotate(
                pygame.Surface((0, 0)), 0
            ) if False else None

            card_cache[key] = surf

        surf = card_cache[key].copy()

        if dim:
            overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            rounded_rect(
                overlay, (35, 39, 48, 155),
                pygame.Rect(0, 0, w, h), 10
            )
            surf.blit(overlay, (0, 0))

        screen.blit(surf, (x, y - 12 if hover else y))

        if hover:
            pygame.draw.rect(
                screen, GOLD,
                (x, y - 12, w, h), 3,
                border_radius=10
            )

        if outline:
            pygame.draw.rect(
                screen, outline,
                (x - 3, y - 3, w + 6, h + 6),
                3, border_radius=12
            )

    # ---------------------------------------------------------
    # CARD BACK
    # ---------------------------------------------------------
    back_cache = {}

    def draw_back(x, y, w=42, h=60):
        key = (w, h)

        if key not in back_cache:
            surf = pygame.Surface((w, h), pygame.SRCALPHA)

            rounded_rect(
                surf, (235, 237, 241),
                (0, 0, w, h), 7
            )

            rounded_rect(
                surf, (40, 69, 124),
                (3, 3, w - 6, h - 6), 5
            )

            pygame.draw.rect(
                surf, (105, 139, 194),
                (7, 7, w - 14, h - 14),
                1, border_radius=4
            )

            for i in range(-h, w, 8):
                pygame.draw.line(
                    surf, (80, 113, 169),
                    (i, 5), (i + h, h - 5), 1
                )

            back_cache[key] = surf

        screen.blit(back_cache[key], (x, y))

    # ---------------------------------------------------------
    # PLAYER PANEL
    # ---------------------------------------------------------
    def draw_player(seat, x, y, score, active, cards_left):
        rect = pygame.Rect(x - 75, y - 34, 150, 68)

        draw_panel(
            rect,
            PANEL_LIGHT if active else PANEL,
            13
        )

        if active:
            pygame.draw.rect(
                screen, GOLD, rect,
                2, border_radius=13
            )

        draw_text(
            SEATS[seat],
            f_player,
            GOLD if active else MUTED,
            x, y - 14, True
        )

        draw_text(
            str(score),
            f_score,
            WHITE,
            x - 20, y + 15, True
        )

        draw_text(
            f"{cards_left} cards",
            f_small,
            MUTED,
            x + 28, y + 15, True
        )

    # ---------------------------------------------------------
    # GAME INITIALIZATION
    # ---------------------------------------------------------
    env = TrickGame()
    bot = NetAgent(net)

    state = {}

    def new_hand():
        env.reset()

        state.update({
            "shown": [],
            "clear_at": None,
            "win_seat": None,
            "bot_at": pygame.time.get_ticks() + 700
        })

    def do_play(card, now):
        state["shown"].append((env.turn, int(card)))

        winner = env.step(int(card))

        if winner is not None:
            state["win_seat"] = winner
            state["clear_at"] = now + 1300
        else:
            state["bot_at"] = now + 550

    new_hand()

    # ---------------------------------------------------------
    # LAYOUT
    # ---------------------------------------------------------
    def layout():
        w, h = screen.get_size()

        human_turn = (
            not env.done
            and env.turn == 0
            and state["clear_at"] is None
        )

        legal = (
            set(np.flatnonzero(env.legal_mask()).tolist())
            if human_turn else set()
        )

        hand = sorted(
            np.flatnonzero(env.hand[0]).tolist()
        )

        n = len(hand)

        card_w = min(CARD_W, max(48, (w - 100) // max(n, 1)))
        gap = min(
            65,
            (w - 100 - card_w) / max(n - 1, 1)
        )

        total = gap * (n - 1) + card_w
        x0 = (w - total) / 2

        hand_y = h - CARD_H - 40

        rects = [
            pygame.Rect(
                round(x0 + i * gap),
                hand_y,
                card_w,
                CARD_H
            )
            for i in range(n)
        ]

        hover = None

        if human_turn:
            mx, my = pygame.mouse.get_pos()

            for i in reversed(range(n)):
                r = rects[i]

                if pygame.Rect(
                    r.x, r.y - 16,
                    r.w, r.h + 16
                ).collidepoint(mx, my):

                    if hand[i] in legal:
                        hover = i
                    break

        return human_turn, legal, hand, rects, hover

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------
    running = True

    while running:

        now = pygame.time.get_ticks()
        W, H = screen.get_size()

        human_turn, legal, hand, rects, hover = layout()

        over = env.done and state["clear_at"] is None

        # Buttons
        new_rect = pygame.Rect(W - 205, 22, 165, 42)
        quit_rect = pygame.Rect(35, 22, 95, 42)

        # -----------------------------------------------------
        # EVENTS
        # -----------------------------------------------------
        for ev in pygame.event.get():

            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(
                    (max(900, ev.w), max(650, ev.h)),
                    pygame.RESIZABLE
                )

            elif ev.type == pygame.KEYDOWN:

                if ev.key == pygame.K_ESCAPE:
                    running = False

                elif ev.key == pygame.K_n and over:
                    new_hand()

            elif ev.type == pygame.MOUSEBUTTONDOWN:

                if ev.button == 1:

                    if quit_rect.collidepoint(ev.pos):
                        running = False
                        continue

                    if new_rect.collidepoint(ev.pos) and over:
                        new_hand()
                        continue

                    if hover is not None and human_turn:
                        do_play(hand[hover], now)
                        break

        # -----------------------------------------------------
        # BOT MOVES AND TRICK ANIMATION
        # -----------------------------------------------------
        if (
            state["clear_at"] is not None
            and now >= state["clear_at"]
        ):
            state.update({
                "shown": [],
                "clear_at": None,
                "win_seat": None,
                "bot_at": now + 350
            })

        elif (
            not env.done
            and state["clear_at"] is None
            and env.turn != 0
            and now >= state["bot_at"]
        ):
            do_play(bot.act(env), now)

        human_turn, legal, hand, rects, hover = layout()

        over = env.done and state["clear_at"] is None

        # -----------------------------------------------------
        # BACKGROUND
        # -----------------------------------------------------
        screen.fill(BG)

        # Header
        draw_text(
            "TRICK",
            f_title,
            GOLD,
            W // 2, 35, True
        )

        draw_text(
            "CLASSIC CARD GAME  |  NO TRUMP",
            f_subtitle,
            MUTED,
            W // 2, 63, True
        )

        draw_button(
            quit_rect,
            "EXIT",
            quit_rect.collidepoint(pygame.mouse.get_pos())
        )

        draw_button(
            new_rect,
            "NEW HAND",
            new_rect.collidepoint(pygame.mouse.get_pos()),
            primary=over
        )

        # -----------------------------------------------------
        # SCOREBOARD
        # -----------------------------------------------------
        score_y = 105
        score_w = min(160, (W - 100) // 4)

        for seat in range(4):
            x = W // 2 + (seat - 1.5) * (score_w + 15)

            rect = pygame.Rect(
                int(x - score_w / 2),
                score_y,
                score_w,
                63
            )

            draw_panel(rect)

            active = (
                not over
                and state["clear_at"] is None
                and env.turn == seat
            )

            if active:
                pygame.draw.rect(
                    screen, GOLD, rect,
                    2, border_radius=12
                )

            draw_text(
                SEATS[seat],
                f_small,
                GOLD if active else MUTED,
                rect.centerx, rect.y + 15, True
            )

            draw_text(
                f"{env.won[seat]} tricks",
                f_player,
                WHITE,
                rect.centerx, rect.y + 40, True
            )

        # -----------------------------------------------------
        # PLAYING TABLE
        # -----------------------------------------------------
        table_rect = pygame.Rect(
            65, 195,
            W - 130,
            H - 385
        )

        draw_panel(table_rect, TABLE_DARK, 35)

        inner = table_rect.inflate(-12, -12)

        rounded_rect(
            screen,
            TABLE,
            inner,
            30
        )

        pygame.draw.rect(
            screen,
            TABLE_LIGHT,
            inner,
            2,
            border_radius=30
        )

        # Table decoration
        pygame.draw.ellipse(
            screen,
            TABLE_LIGHT,
            (
                W // 2 - 235,
                H // 2 - 130,
                470,
                260
            ),
            2
        )

        # Center label
        draw_text(
            "TRICK TABLE",
            font(12, True),
            (76, 149, 115),
            W // 2, H // 2 - 110, True
        )

        # -----------------------------------------------------
        # OPPONENTS' HIDDEN CARDS
        # -----------------------------------------------------
        # North
        n = int(env.hand[2].sum())

        if n:
            total = min(22, 470 / max(n, 1))

            for i in range(n):
                x = W // 2 - (n - 1) * total / 2 + i * total
                draw_back(x - 17, 211, 34, 48)

        # West
        n = int(env.hand[1].sum())

        if n:
            total = min(19, 200 / max(n, 1))

            for i in range(n):
                y = H // 2 - (n - 1) * total / 2 + i * total
                draw_back(88, y - 17, 48, 34)

        # East
        n = int(env.hand[3].sum())

        if n:
            total = min(19, 200 / max(n, 1))

            for i in range(n):
                y = H // 2 - (n - 1) * total / 2 + i * total
                draw_back(W - 136, y - 17, 48, 34)

        # -----------------------------------------------------
        # PLAYER LABELS
        # -----------------------------------------------------
        active_seat = (
            env.turn
            if not over and state["clear_at"] is None
            else state["win_seat"]
        )

        draw_player(
            2, W // 2, 295,
            env.won[2], active_seat == 2,
            int(env.hand[2].sum())
        )

        draw_player(
            1, 205, H // 2,
            env.won[1], active_seat == 1,
            int(env.hand[1].sum())
        )

        draw_player(
            3, W - 205, H // 2,
            env.won[3], active_seat == 3,
            int(env.hand[3].sum())
        )

        draw_player(
            0, W // 2, H - 174,
            env.won[0], active_seat == 0,
            int(env.hand[0].sum())
        )

        # -----------------------------------------------------
        # CURRENT TRICK
        # -----------------------------------------------------
        cx, cy = W // 2, H // 2

        offsets = {
            0: (0, 65),
            1: (-105, 0),
            2: (0, -65),
            3: (105, 0)
        }

        for seat, card in state["shown"]:

            ox, oy = offsets[seat]

            winner_outline = (
                GOLD
                if seat == state["win_seat"]
                else None
            )

            draw_card(
                card,
                cx + ox - TRICK_W // 2,
                cy + oy - TRICK_H // 2,
                TRICK_W,
                TRICK_H,
                outline=winner_outline
            )

        # -----------------------------------------------------
        # GAME STATUS
        # -----------------------------------------------------
        status_y = H - 300

        if over:
            status = "HAND COMPLETED"
            status_color = GOLD

        elif state["clear_at"] is not None:
            status = f"{SEATS[state['win_seat']]} WON THE TRICK"
            status_color = GOLD

        elif env.turn == 0:
            if env.trick:
                led = env.trick[0][1] // 13
                must_follow = any(
                    c // 13 == led for c in hand
                )

                status = (
                    "FOLLOW THE LED SUIT"
                    if must_follow
                    else "NO CARDS IN LED SUIT - PLAY ANY CARD"
                )
            else:
                status = "YOUR TURN - LEAD THE TRICK"

            status_color = GREEN

        else:
            status = f"{SEATS[env.turn]} IS THINKING..."
            status_color = MUTED

        draw_text(
            status,
            f_player,
            status_color,
            W // 2, status_y, True
        )

        # Trick counter
        current_trick = min(env.trick_no + 1, 13)

        draw_text(
            f"TRICK {current_trick} / 13",
            f_small,
            MUTED,
            W // 2, status_y + 27, True
        )

        # -----------------------------------------------------
        # HUMAN HAND
        # -----------------------------------------------------
        n = len(hand)

        for i, card in enumerate(hand):

            r = rects[i]

            is_legal = card in legal
            is_hover = i == hover

            draw_card(
                card,
                r.x,
                r.y,
                r.w,
                CARD_H,
                hover=is_hover,
                dim=human_turn and not is_legal
            )

        # -----------------------------------------------------
        # END GAME OVERLAY
        # -----------------------------------------------------
        if over:

            overlay = pygame.Surface(
                (W, H),
                pygame.SRCALPHA
            )

            overlay.fill((5, 9, 15, 210))
            screen.blit(overlay, (0, 0))

            best = int(env.won.max())

            winners = [
                SEATS[i]
                for i in range(4)
                if env.won[i] == best
            ]

            if env.won[0] == best:
                title = "YOU WIN!"
                subtitle = "Congratulations!"
            else:
                title = "HAND COMPLETED"
                subtitle = (
                    "Winner: " + " & ".join(winners)
                )

            # Result panel
            result_rect = pygame.Rect(
                W // 2 - 260,
                H // 2 - 230,
                520,
                460
            )

            draw_panel(result_rect, PANEL, 24)

            draw_text(
                title,
                f_result,
                GOLD,
                W // 2,
                result_rect.y + 65,
                True
            )

            draw_text(
                subtitle,
                f_subtitle,
                MUTED,
                W // 2,
                result_rect.y + 110,
                True
            )

            # Results
            for i in range(4):

                y = result_rect.y + 165 + i * 43

                row = pygame.Rect(
                    result_rect.x + 45,
                    y - 15,
                    result_rect.width - 90,
                    35
                )

                rounded_rect(
                    screen,
                    PANEL_LIGHT,
                    row,
                    8
                )

                draw_text(
                    SEATS[i],
                    f_player,
                    GOLD if env.won[i] == best else WHITE,
                    row.x + 18,
                    row.centery,
                    True
                )

                draw_text(
                    f"{env.won[i]} tricks",
                    f_result_score,
                    GOLD if env.won[i] == best else WHITE,
                    row.right - 65,
                    row.centery,
                    True
                )

            # New hand button
            result_button = pygame.Rect(
                W // 2 - 105,
                result_rect.bottom - 65,
                210,
                43
            )

            draw_button(
                result_button,
                "PLAY AGAIN",
                result_button.collidepoint(
                    pygame.mouse.get_pos()
                ),
                primary=True
            )

            # Handle result button
            for ev in pygame.event.get():

                if (
                    ev.type == pygame.MOUSEBUTTONDOWN
                    and ev.button == 1
                    and result_button.collidepoint(ev.pos)
                ):
                    new_hand()

        # -----------------------------------------------------
        # FOOTER
        # -----------------------------------------------------
        if not over:

            draw_text(
                "CLICK A CARD TO PLAY",
                f_small,
                MUTED,
                W // 2,
                H - 17,
                True
            )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# ----------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--iters", type=int, default=300)
    t.add_argument("--envs", type=int, default=128, help="hands per iteration")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--pool-prob", type=float, default=0.3)
    t.add_argument("--pool-size", type=int, default=8)
    t.add_argument("--snapshot-every", type=int, default=10)
    t.add_argument("--eval-every", type=int, default=10)
    t.add_argument("--eval-games", type=int, default=400)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--save", default="model.pt")

    e = sub.add_parser("eval")
    e.add_argument("--model", default="model.pt")
    e.add_argument("--games", type=int, default=2000)

    pl = sub.add_parser("play")
    pl.add_argument("--model", default="model.pt")

    x = sub.add_parser("export")
    x.add_argument("--model", default="model.pt")
    x.add_argument("--out", default="weights.json")
    return ap


def main(argv=None):
    """argv=None reads the real command line; pass a list in notebooks."""
    args = build_parser().parse_args(argv)
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        agent = NetAgent(load_net(args.model))
        print("vs random   :", evaluate(agent, RandomAgent(), args.games))
        print("vs heuristic:", evaluate(agent, HeuristicAgent(), args.games))
    elif args.cmd == "play":
        play_cli(load_net(args.model))
    elif args.cmd == "export":
        export_json(load_net(args.model), args.out)


def in_notebook():
    import sys  # local import so this works even if the header was cut off
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


if __name__ == "__main__":
    if in_notebook():
        # Notebook kernels pass their own args (-f kernel.json); ignore them.
        # Edit this line to run something else, e.g. ["eval"] or ["play"].
        main(["train", "--iters", "300"])
    else:
        main()