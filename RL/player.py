import random
import operator
import numpy as np


class Player:
    def __init__(self, tag):
        self.tag = tag

    def get_action(self, state):
        while True:
            try:
                move = int(input(f"Player {self.tag}, enter your move (0–6): "))
                if 0 <= move < 7 and state[0][move] == " ":
                    return move
            except:
                pass
            print("Invalid move. Try again.")

class Agent(Player):
    def __init__(self, tag, exploration_factor=1.0, alpha=0.3, gamma=0.9):
        super().__init__(tag)
        self.exploration_factor = exploration_factor
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = 0.999
        self.exp_min = 0.01
        self.values = {}
        self.reward = 0

    def update_exp_factor(self, episode):
        decay_rate = 0.01
        self.exploration_factor = max(self.exp_min, np.exp(-decay_rate * episode))

    def update_learning_rate(self, episode):
        decay_rate = 0.01
        min_alpha = 0.01
        self.alpha = max(min_alpha, np.exp(-decay_rate * episode))
            

    def get_action(self, state):
        hash_state = tuple(tuple(row) for row in state)
        if hash_state not in self.values:
            self.values[hash_state] = {a: 0 for a in range(7)}
        if random.random() < self.exploration_factor:
            valid_actions = [i for i in range(7) if state[0][i] == " "]
            return random.choice(valid_actions)
        return max(self.values[hash_state].items(), key=operator.itemgetter(1))[0]

    def learn(self, state, action, reward, new_state, done, episode):
        if state not in self.values:
            self.values[state] = {a: 0 for a in range(7)}
        if new_state not in self.values:
            self.values[new_state] = {a: 0 for a in range(7)}

        current_q = self.values[state][action]
        max_future_q = max(self.values[new_state].values()) if not done else 0
        self.reward += reward
        target = self.reward + self.gamma * max_future_q
        new_q = current_q + self.alpha * (target - current_q)
        self.values[state][action] = new_q

        self.update_exp_factor(episode)
        self.update_learning_rate(episode)
