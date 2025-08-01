import random
import operator
from player import Agent, Player
import matplotlib.pyplot as plt


class GameBoard:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = [[" " for _ in range(7)] for _ in range(6)]
        self.player = "X"
        self.winner = None
        return self.state, {}

    def two_in_row(self):
        board = self.state

        # Horizontal
        for r in range(6):
            for c in range(6):  # Only need to go up to column 5 for 2-in-a-row
                if board[r][c] == self.player and board[r][c+1] == self.player:
                    return True

        # Vertical
        for r in range(5):  # Only need to go up to row 4 for 2-in-a-row
            for c in range(7):
                if board[r][c] == self.player and board[r+1][c] == self.player:
                    return True

        # Diagonal (\ direction)
        for r in range(5):
            for c in range(6):
                if board[r][c] == self.player and board[r+1][c+1] == self.player:
                    return True

        # Diagonal (/ direction)
        for r in range(1, 6):
            for c in range(6):
                if board[r][c] == self.player and board[r-1][c+1] == self.player:
                    return True

        return False
    def three_in_row(self):
        board = self.state

        # Horizontal
        for r in range(6):
            for c in range(5):  # Only need to go to column 4
                if (board[r][c] == self.player and
                    board[r][c+1] == self.player and
                    board[r][c+2] == self.player):
                    return True

        # Vertical
        for r in range(4):  # Only need to go to row 3
            for c in range(7):
                if (board[r][c] == self.player and
                    board[r+1][c] == self.player and
                    board[r+2][c] == self.player):
                    return True

        # Diagonal \ (bottom-left to top-right)
        for r in range(4):
            for c in range(5):
                if (board[r][c] == self.player and
                    board[r+1][c+1] == self.player and
                    board[r+2][c+2] == self.player):
                    return True

        # Diagonal / (top-left to bottom-right)
        for r in range(2, 6):
            for c in range(5):
                if (board[r][c] == self.player and
                    board[r-1][c+1] == self.player and
                    board[r-2][c+2] == self.player):
                    return True

        return False
    
    def opposition_three_in_row(self):
        opponent = "O" if self.player == "X" else "X"
        board = self.state

        # Horizontal
        for r in range(6):
            for c in range(5):  # Only need to go to column 4
                if (board[r][c] == opponent and
                    board[r][c+1] == opponent and
                    board[r][c+2] == opponent):
                    return True

        # Vertical
        for r in range(4):  # Only need to go to row 3
            for c in range(7):
                if (board[r][c] == opponent and
                    board[r+1][c] == opponent and
                    board[r+2][c] == opponent):
                    return True

        # Diagonal \ (bottom-left to top-right)
        for r in range(4):
            for c in range(5):
                if (board[r][c] == opponent and
                    board[r+1][c+1] == opponent and
                    board[r+2][c+2] == opponent):
                    return True

        # Diagonal / (top-left to bottom-right)
        for r in range(2, 6):
            for c in range(5):
                if (board[r][c] == opponent and
                    board[r-1][c+1] == opponent and
                    board[r-2][c+2] == opponent):
                    return True

        return False
    
    def move(self, action):
        reward = 0
        
        if self.state[0][action] != " ":
            return self.state, -10, False, {}  # Invalid move (column full)

        for i in reversed(range(6)):
            if self.state[i][action] == " ":
                self.state[i][action] = self.player
                break
        
        if self.opposition_three_in_row():
            reward -= 100

        elif self.three_in_row():
            reward += 25

        elif self.two_in_row():
            reward +=  15

        if self.check_winner():
            if self.winner == self.player:
                return self.state, 50, True, {}
            else:
                return self.state, -50, True, {}
        elif not any(" " in row for row in self.state):
            self.winner = "T"
            return self.state, 1, True, {}
            

        self.player = "O" if self.player == "X" else "X"
        return self.state, reward, False, {}

    def check_winner(self):
        board = self.state
        # Horizontal & Vertical
        for r in range(6):
            for c in range(4):
                if board[r][c] != " " and all(board[r][c+i] == board[r][c] for i in range(4)):
                    self.winner = str(board[r][c])
                    return True
        for r in range(3):
            for c in range(7):
                if board[r][c] != " " and all(board[r+i][c] == board[r][c] for i in range(4)):
                    self.winner = str(board[r][c])
                    return True
        # Diagonals
        for r in range(3):
            for c in range(4):
                if board[r][c] != " " and all(board[r+i][c+i] == board[r][c] for i in range(4)):
                    self.winner = board[r][c]
                    return True
        for r in range(3, 6):
            for c in range(4):
                if board[r][c] != " " and all(board[r-i][c+i] == board[r][c] for i in range(4)):
                    self.winner = board[r][c]
                    return True
        return False

    def render(self):

        for row in self.state:
            print(" | ".join(row))
        print("-" * 29)
        print(" 0   1   2   3   4   5   6 ")





def train_agent(episodes, agent1, agent2):
    env = GameBoard()

    agent1_learning_rate = []
    agent2_learning_rate = []
    agent1_exploration_factor = []
    agent2_exploration_factor = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        current_player = agent1 if episode % 2 == 0 else agent2
        other_player = agent2 if current_player == agent1 else agent1
        print(f"\r{episode + 1}/{episodes} training rounds done", end="", flush=True)

        while not done:
            action = current_player.get_action(state)
            next_state, reward, done, _ = env.move(action)

            state_hash = tuple(tuple(row) for row in state) # Convert state to a hashable type
            next_hash = tuple(tuple(row) for row in next_state)

            current_player.learn(state_hash, action, reward, next_hash, done, episode)

            state = next_state
            current_player, other_player = other_player, current_player
        
        agent1_learning_rate.append(agent1.alpha)
        agent2_learning_rate.append(agent2.alpha)
        agent1_exploration_factor.append(agent1.exploration_factor)
        agent2_exploration_factor.append(agent2.exploration_factor)

    return agent1_learning_rate, agent2_learning_rate, agent1_exploration_factor, agent2_exploration_factor


def play_human_vs_agent(agent, player):
    env = GameBoard()
    #player = Player("O")
    state, _ = env.reset()
    env.render()
    done = False

    while not done:
        if env.player == agent.tag:
            hash_state = tuple(tuple(row) for row in state)
            action = agent.get_action(state)
            print(action)
            print(agent.values[hash_state])
        else:
            action = player.get_action(state)

        state, reward, done, _ = env.move(action)
        env.render()

    if env.winner == "T":
        print("It's a tie!")
    else:
        print(f"{env.winner} wins!")



if __name__ == "__main__":
    agent1 = Agent(tag="X")
    agent2 = Agent(tag="O")
    print("Playing a game with the agent...")
    #play_human_vs_agent(agent1, Player(tag="O"))
    print("Training agents... please wait.")
    agent1_learning_rate, agent2_learning_rate, agent1_exploration_rate, agent2_exploration_rate = train_agent(10000, agent1, agent2)
    print("Training complete!")
    print("Agent 1 Values:")
    

    plt.plot(agent1_learning_rate, label='Agent 1 Learning Rate')
    plt.plot(agent2_learning_rate, label='Agent 2 Learning Rate')
    plt.xlabel('Episodes')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()
    plt.plot(agent1_exploration_rate, label='Agent 1 Exploration Factor')
    plt.plot(agent2_exploration_rate, label='Agent 2 Exploration Factor')
    plt.xlabel('Episodes')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()

    play_human_vs_agent(agent1, Player(tag="O"))


