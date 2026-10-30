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

## Running the file

Train:     python trick_rl.py train --iters 300
Evaluate:  python trick_rl.py eval  --model model.pt
Play:      python trick_rl.py play  --model model.pt
Export:    python trick_rl.py export --model model.pt --out weights.json
 

 
Technique: PPO with action masking, GAE, a network shared by all four seats,
and an opponent pool of past checkpoints to keep self-play stable.
Seats are encoded relative to the acting player (0 = me, 1 = next to play...).