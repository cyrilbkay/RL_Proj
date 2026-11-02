#!/usr/bin/env python3
"""
trick_rl.py - A 4-player, 52-card, 13-cards-each bidding trick-taking game
plus a PPO self-play agent that learns to bid and play it.

Rules:
  * Each player gets 13 random cards. A random player bids first (1-13) and
    leads the first trick. Everyone sees the bids already made.
  * You must follow the led suit if you can; otherwise play any card.
  * HEARTS ARE TRUMP: any heart beats every other suit; otherwise the highest
    card of the led suit wins. Winner leads next.
  * Hand score: tricks won >= bid -> +bid, otherwise -bid.
  * Totals carry over hands; first to 41 ends the game.

Install:   pip install torch numpy pygame
Run:       python 400.py   (no args: trains if model.pt is missing, then plays)
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
HEARTS = 1
BID_BASE = 52          # action 52 = bid 1, ..., action 64 = bid 13
N_ACT = 65
OBS_DIM = 343
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
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
        self.bids = np.zeros(4, dtype=np.int64)   # 0 = hasn't bid yet
        self.bid_count = 0
        self.bidding = True
        self.start = int(self.rng.integers(4))
        self.turn = self.start
        return self.turn

    @property
    def done(self):
        return self.trick_no == 13

    def legal_mask(self):
        m = np.zeros(N_ACT, dtype=bool)
        if self.bidding:
            m[BID_BASE:] = True
            return m
        h = self.hand[self.turn]
        if self.trick:
            led = self.trick[0][1] // 13
            suited = h[led * 13:(led + 1) * 13]
            if suited.any():
                m[led * 13:(led + 1) * 13] = suited
                return m
        m[:52] = h
        return m

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
        bids = np.array([self.bids[q] for q in rel]) / 13.0
        need = (self.bids[p] - self.won[p]) / 13.0
        return np.concatenate([
            self.hand[p], self.played, played_by, voids,
            trick_vec, led, pos, won, [self.trick_no / 13.0],
            bids, [float(self.bidding)], [need],
        ]).astype(np.float32)

    def step(self, action):
        """Play a card for the current player. Returns the trick winner if the
        trick just completed, else None."""
        p = self.turn
        if self.bidding:
            self.bids[p] = action - BID_BASE + 1
            self.bid_count += 1
            if self.bid_count == 4:
                self.bidding = False
                self.turn = self.start
            else:
                self.turn = (p + 1) % 4
            return None
        card = action
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
        winner = max(self.trick, key=lambda t: strength(t[1], led))[0]
        self.won[winner] += 1
        self.trick_no += 1
        self.trick = []
        self.turn = winner
        return winner

    def scores(self):
        return np.where(self.won >= self.bids, self.bids, -self.bids)


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
        self.pi = nn.Linear(hidden, N_ACT)
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

def strength(card, led):
    """Hearts beat everything; otherwise only the led suit can win."""
    suit, rank = card // 13, card % 13
    if suit == HEARTS:
        return 100 + rank
    if suit == led:
        return rank
    return -1

class HeuristicAgent:
    """Lead the strongest card; follow by winning as cheaply as possible,
    otherwise dump the least valuable card (saving hearts)."""

    def bid(self, env):
        est = 0.0
        for c in np.flatnonzero(env.hand[env.turn]):
            suit, rank = c // 13, c % 13
            if suit == HEARTS:
                est += 0.25 + 0.6 * rank / 12
            elif rank == 12:
                est += 0.8
            elif rank == 11:
                est += 0.4
        return int(np.clip(round(est), 1, 13))

    def act(self, env):
        if env.bidding:
            return BID_BASE + self.bid(env) - 1
        legal = np.flatnonzero(env.legal_mask())

        # Leading: play the strongest card (hearts count as highest)
        if not env.trick:
            return int(max(legal, key=lambda c: strength(c, c // 13)))

        led = env.trick[0][1] // 13
        best = max(strength(c, led) for _, c in env.trick)

        # Cheapest card that beats the current winner
        winners = [c for c in legal if strength(c, led) > best]
        if winners:
            return int(min(winners, key=lambda c: strength(c, led)))

        # Can't win: dump the lowest card, preferring to keep hearts
        return int(min(legal, key=lambda c: (c // 13 == HEARTS, c % 13)))

class NetAgent:
    def __init__(self, net, greedy=True):
        self.net, self.greedy = net, greedy

    def act(self, env):
        a, _, _ = act_batch(self.net, env.observe()[None],
                            env.legal_mask()[None], self.greedy)
        return int(a[0])


def evaluate(agent, opponent, games=400, seed=123):
    """Average hand score of `agent` vs 3 copies of `opponent`."""
    env = TrickGame(seed)
    total = 0
    for g in range(games):
        seat = g % 4
        env.reset()
        while not env.done:
            who = agent if env.turn == seat else opponent
            env.step(who.act(env))
        total += env.scores()[seat]
    return total / games


def evaluate_wins(agent, opponent, games=400, seed=123):
    """Win rate of `agent` vs 3 copies of `opponent` (ties split). Even = 0.25."""
    env = TrickGame(seed)
    total = 0.0
    for g in range(games):
        seat = g % 4
        env.reset()
        while not env.done:
            who = agent if env.turn == seat else opponent
            env.step(who.act(env))
        sc = env.scores()
        top = sc == sc.max()
        total += top[seat] / top.sum()
    return total / games


def hand_reward(env, mode="mix"):
    """Per-seat reward at the end of a hand.
    score    : hand score / 13 (maximizes expected points)
    win      : +1 share for having the highest score (ties split), zero-sum
    mix      : win signal + a dense score-vs-table hint (recommended)
    relative : score minus the table average"""
    sc = env.scores().astype(np.float32)
    if mode == "score":
        return sc / 13.0
    top = sc == sc.max()
    win = top / top.sum()
    if mode == "win":
        return win - 0.25
    if mode == "mix":
        return (win - 0.25) + 0.5 * (sc - sc.mean()) / 13.0
    return (sc - sc.mean()) / 13.0


# ----------------------------------------------------------------------------
# PPO self-play
# ----------------------------------------------------------------------------
def rollout(envs, net, pool, pool_prob, rng, reward_mode="mix",
            gamma=1.0, lam=0.95):
    E, T = len(envs), 56
    obs_b = np.zeros((E, T, OBS_DIM), np.float32)
    mask_b = np.zeros((E, T, N_ACT), bool)
    act_b = np.zeros((E, T), np.int64)
    logp_b = np.zeros((E, T), np.float32)
    val_b = np.zeros((E, T), np.float32)
    seat_b = np.zeros((E, T), np.int64)

    learner = np.ones((E, 4), bool)
    opp_idx = np.full(E, -1)
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
            env.step(int(acts[i]))

    O, M, A, LP, ADV, RET = [], [], [], [], [], []
    for i in range(E):
        final = hand_reward(envs[i], reward_mode)
        for p in range(4):
            if not learner[i, p]:
                continue
            idx = np.flatnonzero(seat_b[i] == p)      # 1 bid + 13 plays
            v = val_b[i, idx]
            r = np.zeros(len(idx), np.float32)
            r[-1] = final[p]
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
        batch = rollout(envs, net, pool, args.pool_prob if pool else 0.0, rng,
                        args.reward)
        stats = ppo_update(net, opt, batch)

        if it % args.snapshot_every == 0:
            snap = copy.deepcopy(net).eval()
            pool = (pool + [snap])[-args.pool_size:]

        if it % args.eval_every == 0 or it == 1:
            agent = NetAgent(net)
            vr = evaluate(agent, rand, args.eval_games)
            vh = evaluate(agent, heur, args.eval_games)
            wr = evaluate_wins(agent, rand, args.eval_games)
            wh = evaluate_wins(agent, heur, args.eval_games)
            print(f"iter {it:4d} | score: rand {vr:+.2f} heur {vh:+.2f} "
                  f"| win rate (0.25 = even): rand {wr:.2f} heur {wh:.2f} | ent {stats['ent']:.2f} val {stats['val']:.3f} "
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
    Note: the JS side must build the same 343-dim observation as
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

        b = int(env.bids[seat])
        draw_text(
            f"{score} / {b if b else '-'}",
            f_score,
            WHITE,
            x - 28, y + 15, True
        )

        draw_text(
            f"{cards_left} cards",
            f_small,
            MUTED,
            x + 42, y + 15, True
        )

    # ---------------------------------------------------------
    # GAME INITIALIZATION
    # ---------------------------------------------------------
    env = TrickGame()
    bot = NetAgent(net)

    state = {}

    TARGET = 41
    totals = np.zeros(4, dtype=np.int64)
    game = {"over": False, "hands": 0}

    def new_hand():
        env.reset()

        state.update({
            "shown": [],
            "clear_at": None,
            "win_seat": None,
            "scored": False,
            "bot_at": pygame.time.get_ticks() + 700
        })

    def new_game():
        totals[:] = 0
        game["over"] = False
        game["hands"] = 0
        new_hand()

    def do_play(action, now):
        if env.bidding:
            env.step(int(action))
            state["bot_at"] = now + 600
            return
        state["shown"].append((env.turn, int(action)))
        winner = env.step(int(action))
        if winner is not None:
            state["win_seat"] = winner
            state["clear_at"] = now + 1300
        else:
            state["bot_at"] = now + 550

    new_hand()

    # ---------------------------------------------------------
    # LAYOUT
    # ---------------------------------------------------------
    def bid_rects():
        w, h = screen.get_size()
        bw, bh, gap, cols = 58, 48, 10, 7
        x0 = w // 2 - (cols * bw + (cols - 1) * gap) // 2
        y0 = h // 2 - 70
        return [pygame.Rect(x0 + (i % cols) * (bw + gap),
                            y0 + (i // cols) * (bh + gap), bw, bh)
                for i in range(13)]

    def draw_bid_strip():
        w, h = screen.get_size()
        cw, gap = 120, 10
        x0 = w // 2 - (4 * cw + 3 * gap) // 2
        order = [(env.start + k) % 4 for k in range(4)]   # bidding order
        for k, s_ in enumerate(order):
            r = pygame.Rect(x0 + k * (cw + gap), h // 2 + 150, cw, 54)
            draw_panel(r, PANEL_LIGHT if env.turn == s_ else PANEL, 12)
            draw_text(SEATS[s_], f_small, MUTED, r.centerx, r.y + 14, True)
            b = int(env.bids[s_])
            draw_text(b if b else "...", f_score,
                      GOLD if b else MUTED, r.centerx, r.y + 36, True)

    def layout():
        w, h = screen.get_size()

        human_turn = (not env.done and not env.bidding
              and env.turn == 0 and state["clear_at"] is None)

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
                    new_game() if game["over"] else new_hand()

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:

                if quit_rect.collidepoint(ev.pos):
                    running = False
                    continue

                if over:
                    result_button = pygame.Rect(W // 2 - 105, H // 2 + 230 - 65, 210, 43)
                    if new_rect.collidepoint(ev.pos) or result_button.collidepoint(ev.pos):
                        new_game() if game["over"] else new_hand()
                        break
                    continue

                if env.bidding:
                    if env.turn == 0 and state["clear_at"] is None:
                        for i, r in enumerate(bid_rects()):
                            if r.collidepoint(ev.pos):
                                do_play(BID_BASE + i, now)
                                break
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

        if over and not state["scored"]:
            totals += env.scores()
            state["scored"] = True
            game["hands"] += 1
            if totals.max() >= TARGET:
                game["over"] = True

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
            "BIDDING  |  HEARTS TRUMP",
            f_subtitle,
            MUTED,
            W // 2, 63, True
        )

        draw_text(
            f"FIRST TO {TARGET}  |  HAND {game['hands'] + (0 if over else 1)}",
            f_small, MUTED, W // 2, 85, True
        )

        draw_button(
            quit_rect,
            "EXIT",
            quit_rect.collidepoint(pygame.mouse.get_pos())
        )

        draw_button(
            new_rect,
            "NEW GAME" if game["over"] else "NEXT HAND",
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
                f"{SEATS[seat]}  |  {totals[seat]} pts",
                f_small,
                GOLD if active else MUTED,
                rect.centerx, rect.y + 15, True
            )

            b = int(env.bids[seat])
            draw_text(
                f"{env.won[seat]} / {b if b else '-'}",
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
        if env.bidding:
            status = ("CHOOSE YOUR BID (1-13)" if env.turn == 0
                else f"{SEATS[env.turn]} IS BIDDING...")
            status_color = GREEN if env.turn == 0 else MUTED
        elif over:
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
        if env.bidding:
            draw_bid_strip()
        if env.bidding and env.turn == 0 and state["clear_at"] is None:
            panel = pygame.Rect(W // 2 - 280, H // 2 - 130, 560, 240)
            draw_panel(panel, PANEL, 20)
            draw_text("HOW MANY TRICKS WILL YOU WIN?", f_player, GOLD,
                    W // 2, panel.y + 30, True)
            mp = pygame.mouse.get_pos()
            for i, r in enumerate(bid_rects()):
                draw_button(r, str(i + 1), r.collidepoint(mp))
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

            sc = env.scores()
            if game["over"]:
                top = int(totals.max())
                champs = [SEATS[i] for i in range(4) if totals[i] == top]
                if totals[0] == top:
                    title, subtitle = "YOU WIN THE GAME!", f"Final total: {top}"
                else:
                    title, subtitle = "GAME OVER", "Winner: " + " & ".join(champs)
                best = top
            else:
                best = int(sc.max())
                title = "YOU WON THE HAND" if sc[0] == best else "HAND COMPLETED"
                subtitle = f"First to {TARGET} wins the game"

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

                hl = (totals[i] if game["over"] else sc[i]) == best
                col = GOLD if hl else WHITE

                draw_text(
                    SEATS[i], f_player, col,
                    row.x + 14, row.centery - 9
                )

                t = f_player.render(
                    f"{env.won[i]}/{env.bids[i]}   {sc[i]:+d}   =   {totals[i]}",
                    True, col
                )
                screen.blit(t, (row.right - 14 - t.get_width(),
                                row.centery - t.get_height() // 2))

            # New hand button
            result_button = pygame.Rect(
                W // 2 - 105,
                result_rect.bottom - 65,
                210,
                43
            )

            draw_button(
                result_button,
                "NEW GAME" if game["over"] else "NEXT HAND",
                result_button.collidepoint(
                    pygame.mouse.get_pos()
                ),
                primary=True
            )


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
    t.add_argument("--iters", type=int, default=1000)
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
    t.add_argument("--reward", default="mix",
                   choices=["score", "win", "mix", "relative"],
                   help="what the bot maximizes (default: mix = win rate + score hint)")

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
    """argv=None reads the real command line."""
    args = build_parser().parse_args(argv)
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        agent = NetAgent(load_net(args.model))
        print("vs random   :", evaluate(agent, RandomAgent(), args.games))
        print("vs heuristic:", evaluate(agent, HeuristicAgent(), args.games))
        print("win rate vs random   :", evaluate_wins(agent, RandomAgent(), args.games))
        print("win rate vs heuristic:", evaluate_wins(agent, HeuristicAgent(), args.games))
    elif args.cmd == "play":
        play_cli(load_net(args.model))
    elif args.cmd == "export":
        export_json(load_net(args.model), args.out)


if __name__ == "__main__":
    import os
    if len(sys.argv) > 1:
        main()                                   # python 400.py train/play/...
    else:                                        # plain "Run" with no arguments
        if not os.path.exists("model.pt"):
            main(["train", "--iters", "1000"])
        main(["play", "--model", "model.pt"])