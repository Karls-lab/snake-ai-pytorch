import multiprocessing
from agent import Agent 
from model import Linear_QNet, DuelingQNetwork

"""
File can run multiple instances of the game
"""

if __name__ == '__main__':
    # train()
    simpleModel = Linear_QNet(15, 256, 3)
    agent1 = Agent(model=simpleModel, max_memory=100_000, batch_size=100, gamma=.1)
    agent1.train(100)

    # train()
    # num_features, hidden layers, output
    # model = DuelingQNetwork(15, 256, 3)
    # agent1 = Agent(model=model, max_memory=1_000, batch_size=1000, gamma=.85)
    # agent1.train(200)



