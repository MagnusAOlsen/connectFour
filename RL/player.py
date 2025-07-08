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
    def __init__(self, tag, exploration_factor=1.0, alpha=0.3, gamma=0.95):
        super().__init__(tag)
        self.initial_exploration_factor = exploration_factor    
        self.exploration_factor = exploration_factor
        self.initial_alpha = alpha
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = 0.999
        self.exp_min = 0.01
        self.values = {}
        self.reward = 0

    def update_exp_factor(self, episode, decay_until=250000):
        if episode >= decay_until:
            self.exploration_factor = self.exp_min
        else:
            ratio = episode / decay_until  # Normalize within decay range
            cosine = 0.5 * (1 + np.cos(np.pi * ratio))  # Cosine from 1 to 0
            self.exploration_factor = self.exp_min + (self.initial_exploration_factor - self.exp_min) * cosine




    
            
            

    def get_action(self, state):
        hash_state = tuple(tuple(row) for row in state)
        if hash_state not in self.values:
            print("State not found in values, initializing...")
            self.values[hash_state] = {a: np.random.rand() for a in range(7)}
        if random.random() < self.exploration_factor:
            valid_actions = [i for i in range(7) if state[0][i] == " "]
            return random.choice(valid_actions)
        return max(self.values[hash_state].items(), key=operator.itemgetter(1))[0]

    def learn(self, state, action, reward, new_state, done, episode):
        np.random.seed(episode)
        random.seed(episode)
        hash_state = tuple(tuple(row) for row in state)
        hash_new_state = tuple(tuple(row) for row in new_state)
        if hash_state not in self.values:
            self.values[state] = {a: np.random.rand() for a in range(7)}
        if hash_new_state not in self.values:
            self.values[new_state] = {a: np.random.rand() for a in range(7)}

        current_q = self.values[state][action]
        max_future_q = max(self.values[new_state].values()) if not done else 0
        #self.reward += reward
        target = reward + self.gamma * max_future_q
        new_q = current_q + self.alpha * (target - current_q)
        self.values[state][action] = new_q
        

        self.update_exp_factor(episode)
        #self.update_learning_rate(episode)
