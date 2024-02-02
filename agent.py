import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
# from model import QNetwork, DQNAgent
from helper import plot
import sys

class Agent:
    def __init__(self, model, max_memory, batch_size, lr=0.001, gamma=1):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = gamma # discount rate
        self.memory = deque(maxlen=max_memory) # popleft()
        # self.model = Linear_QNet(11, 256, 3)
        self.model = model
        # self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.trainer = QTrainer(self.model, lr=lr, gamma=gamma) 
        self.batch_size = batch_size


    def get_state(self, game):
        head = game.snake[0]
        tail = game.snake[-1]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Tail Location 
            tail.x < head.x,  # tail left
            tail.x > head.x,  # tail right
            tail.y < head.y,  # tail up
            tail.y > head.y,  # tail down

            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached


    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(0.01, 1.0 - 0.01 * self.n_games)
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


    def train(self, epochs=5):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        game = SnakeGameAI()
        while True:
            # else, 
            oldState = self.get_state(game)
            lastMove = self.get_action(oldState)

            # perform move and get new state
            reward, done, score = game.play_step(lastMove)
            newState = self.get_state(game)

            # train short memory
            self.train_short_memory(oldState, lastMove, reward, newState, done)

            # remember
            self.remember(oldState, lastMove, reward, newState, done)

            if done:
                # train long memory, plot result
                game.reset()
                self.n_games += 1
                self.train_long_memory()

                if score > record:
                    record = score
                    self.model.save()

                if self.n_games > epochs:
                    modelName = type(self.model).__name__
                    self.model.save(file_name=f'{modelName}-E:{epochs}|S{record}.pth')
                    sys.exit()

                print('Game', self.n_games, 'Score', score, 'Record:', record)

                # plot_scores.append(score)
                # total_score += score
                # mean_score = total_score / self.n_games
                # plot_mean_scores.append(mean_score)
                # plot(plot_scores, plot_mean_scores)
