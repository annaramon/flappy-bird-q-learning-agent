# Flappy Bird Q-Learning Agent

An autonomous agent that learns to play Flappy Bird using **tabular Q-Learning**, trained through exploration-exploitation balancing and custom reward shaping.

## Demo

The agent starts by crashing immediately, but after training progressively learns to navigate through pipes by optimizing its Q-table over thousands of episodes.

## How It Works

### State Space

The agent observes 5 discrete variables via `get_state()`, producing **159,936 possible states**. At each step, it chooses one of two actions: **jump** or **stay**.

### Q-Table

```python
q_table = {
    (state_1, state_2, state_3, state_4, state_5): [q_value_stay, q_value_jump],
    ...  # 159,936 entries
}
```

Each state maps to two Q-values — one per action. The agent picks the action with the highest Q-value (exploitation) or a random action (exploration), controlled by an epsilon-greedy strategy.

### Exploration vs. Exploitation

The agent uses a decaying exploration rate:

- Starts at **1.0** (fully random) to discover which actions lead to rewards
- Decays over episodes, shifting toward **exploiting** learned Q-values
- Floors at a minimum rate to prevent fully deterministic behavior

**Key finding:** If the decay is too aggressive, the agent gets trapped — e.g., always dying at the same pipe because it never explored alternatives. Too slow, and it behaves randomly for too long.

## Hyperparameter Tuning

Systematically tested the impact of each parameter:

| Parameter | Problem | Solution |
|---|---|---|
| `DISCOUNT_RATE = 0` | Agent only considers immediate rewards — crashes into ceiling every game | Increased to **0.9** — agent learned to anticipate upcoming pipes |
| `LR = 0.3` | Q-values update too aggressively, unstable learning | Reduced to **0.2** — smoother convergence |
| `LR = 0.001` | Q-values barely differentiate between actions, needs millions of episodes | Too conservative for this state space |

### Final Configuration

```python
LR = 0.2                      # Stable updates without losing past knowledge
DISCOUNT_RATE = 0.9            # Heavily weighs future rewards
MAX_EXPLORATION_RATE = 1.0     # Start fully exploratory
MIN_EXPLORATION_RATE = 0.0002  # Never go fully greedy
EXPLORATION_DECAY_RATE = 0.00015
```

**Why this works:** A low learning rate provides stable convergence — the agent is cautious about overwriting past experience. A high discount rate forces the agent to consider long-term consequences (upcoming pipes), not just the immediate reward of surviving one more frame.

## Reward Engineering

Extended the base reward function to incentivize precise flying:

| Event | Reward | Rationale |
|---|:---:|---|
| Pass through a pipe | +50 | Primary objective |
| Move toward pipe gap | +1 | Encourage correct positioning |
| Move away from pipe gap | -1 | Discourage wrong direction |
| Collide with pipe or boundary | -1 | Penalize failure |
| **Fly through center of gap** | **+10** | **Reward precision, not just survival** |
| **Unnecessary altitude change** | **-1** | **Penalize erratic movement** |

**Design principle:** Penalties stay low to avoid suppressing exploration. Precision rewards are higher to guide the agent toward optimal trajectories, not just "good enough" ones.

## Results

- With `DISCOUNT_RATE = 0`: agent crashes every single game (ceiling collision)
- With `DISCOUNT_RATE = 0.9` + `LR = 0.2`: agent quickly learns to navigate pipes and sustain long runs
- Custom reward shaping further improved consistency by encouraging centered, smooth flight paths

## Tech Stack

- Python
- Pygame
- Q-Learning (tabular)
- Epsilon-greedy exploration strategy

## Project Structure

```
├── agent.py            # Q-Learning agent + training loop
├── flappy_bird.py      # Game engine with reward system
├── resources/           # Sprites, fonts, and background assets
├── requirements.txt
└── README.md
```

## Getting Started

```bash
pip install -r requirements.txt
python agent.py
```

Set `VISUALIZATION = False` in `agent.py` to train without rendering (much faster).

