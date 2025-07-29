import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim=6*7, output_dim=7):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
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

class Agent:
    def __init__(self, tag, lr=0.001, gamma=0.95, exploration_factor=1.0):
        self.tag = tag
        self.gamma = gamma
        self.epsilon = exploration_factor
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def encode_state(self, state):
        # Flatten board and encode as tensor: " " → 0, self → 1, opponent → -1
        flat = []
        for row in state:
            for cell in row:
                if cell == " ":
                    flat.append(0)
                elif cell == self.tag:
                    flat.append(1)
                else:
                    flat.append(-1)
        return torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(self.device)

    def get_action(self, state):
        if random.random() < self.epsilon:
            valid_actions = [i for i in range(7) if state[0][i] == " "]
            return random.choice(valid_actions)

        with torch.no_grad():
            state_tensor = self.encode_state(state)
            q_values = self.model(state_tensor)
            q_values = q_values.cpu().numpy().flatten()

        valid_actions = [i for i in range(7) if state[0][i] == " "]
        q_values = [(i, q_values[i]) for i in valid_actions]
        return max(q_values, key=lambda x: x[1])[0]

    def learn(self, state, action, reward, next_state, done, episode):
        state_tensor = self.encode_state(state)
        next_tensor = self.encode_state(next_state)

        self.model.train()
        q_vals = self.model(state_tensor)
        q_val = q_vals[0][action]

        with torch.no_grad():
            q_next = self.model(next_tensor)
            max_q_next = torch.max(q_next) if not done else torch.tensor(0.0).to(self.device)

        target = reward + self.gamma * max_q_next
        loss = self.loss_fn(q_val, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
