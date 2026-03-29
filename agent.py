import random
import numpy as np
from flappy_bird import FlappyBird

# Hyperparameters
LR = 0.2
NUM_EPISODES = 1000
DISCOUNT_RATE = 0.9
MAX_EXPLORATION_RATE = 1.0
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY_RATE = 0.001

VISUALIZATION = True


class Agent:
    """Q-Learning agent that learns to play Flappy Bird through tabular Q-Learning."""

    def __init__(self):
        self.q_table = {}
        x_bins, y_bins, sense_bins, y2_bins, sense2_bins = 51, 28, 2, 28, 2
        for x in range(x_bins):
            for y in range(y_bins):
                for s in range(sense_bins):
                    for y2 in range(y2_bins):
                        for s2 in range(sense2_bins):
                            self.q_table[(x, y, s, y2, s2)] = [0, 0]

        self.n_games = 0
        self.exploration_rate = MAX_EXPLORATION_RATE

    def get_state(self, game):
        """Extract a discretized 5-dimensional state from the game environment.

        Returns a tuple of:
            - Horizontal distance to current pipe (0-50, binned by 22px)
            - Vertical distance to current pipe gap (0-27, binned by 15px)
            - Direction to current pipe gap (0=above, 1=below)
            - Vertical distance to next pipe gap (0-27, binned by 15px)
            - Direction to next pipe gap (0=above, 1=below)
        """
        state = []

        # Horizontal distance to nearest pipe
        delta_x = game.current_wall.x - game.character.x
        delta_x = delta_x // 22
        delta_x = max(0, min(delta_x, 50))
        state.append(delta_x)

        # Vertical distance and direction to current pipe gap
        sense = 0
        delta_y = game.character.y - game.current_wall.hole
        if delta_y < 0:
            sense = 1
        delta_y = min(abs(delta_y // 15), 27)
        state.append(int(delta_y))
        state.append(sense)

        # Vertical distance and direction to next pipe gap
        sense_next = 0
        next_wall = game.walls[1] if game.current_wall == game.walls[0] else game.walls[0]
        delta_y_next = game.character.y - next_wall.hole
        if delta_y_next < 0:
            sense_next = 1
        delta_y_next = min(abs(delta_y_next // 15), 27)
        state.append(int(delta_y_next))
        state.append(sense_next)

        return tuple(state)

    def get_action(self, state):
        """Select an action using epsilon-greedy strategy.

        Returns 0 (stay) or 1 (jump).
        """
        if random.uniform(0, 1) > self.exploration_rate:
            return np.argmax(self.q_table[state])
        else:
            return random.choice([0, 1])


def train():
    """Main training loop. Runs NUM_EPISODES games and logs progress every 100."""
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    period_steps = 0
    period_score = 0

    agent = Agent()
    game = FlappyBird(vis=VISUALIZATION)
    steps = 0

    while True:
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)

        # Q-Learning update: Q(s,a) <- (1-α)Q(s,a) + α(r + γ·max Q(s',a'))
        new_state = agent.get_state(game)
        agent.q_table[state][action] = (
            (1 - LR) * agent.q_table[state][action]
            + LR * (reward + DISCOUNT_RATE * np.max(agent.q_table[new_state]))
        )
        state = new_state

        if done:
            # Decay exploration rate
            agent.exploration_rate = (
                MIN_EXPLORATION_RATE
                + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE)
                * np.exp(-EXPLORATION_DECAY_RATE * period_steps)
            )
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score

            period_steps += steps
            period_score += score
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            steps = 0

            # Log progress every 100 games
            if agent.n_games % 100 == 0:
                np.save("q_table.npy", agent.q_table)
                print(
                    f"Game {agent.n_games} | "
                    f"Mean Score: {period_score / 100:.1f} | "
                    f"Record: {record} | "
                    f"Exploration Rate: {agent.exploration_rate:.4f} | "
                    f"Avg Steps: {period_steps / 100:.0f}"
                )
                record = 0
                period_score = 0
                period_steps = 0

            if agent.n_games == NUM_EPISODES:
                np.save("q_table.npy", agent.q_table)
                print(f"Training complete. Q-table saved to q_table.npy")
                break
        else:
            steps += 1


if __name__ == "__main__":
    train()
