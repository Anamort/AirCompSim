import random

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers.legacy import SGD, Adam
from keras.losses import mse
from DRL import MemoryItem, State
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
This module provides an implementation of Double Deep Q Learning (DDQN) algorithm for DRL-based experiments.
'''

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.0001
MIN_EPSILON = 0.01
EPSILON_FACTOR = 0.99
TAU = 0.01
BATCH_SIZE = 64

class DDQNAgent(nn.Module):
    id: int = 0

    def __init__(self, state_size, action_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = DDQNAgent.id
        DDQNAgent.id += 1
        #self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu' if torch.backends.mps.is_available() else 'cpu')
        self.memoryItems = []
        self.reward = 0
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        ).to(torch.float32).to(self.device)
        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=LEARNING_RATE)
        self.actionSpace = action_size
        self.epsilon = 1

        self.targetNetwork = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        ).to(torch.float32).to(self.device)

        for target_param, param in zip(self.targetNetwork.parameters(), self.network.parameters()):
            target_param.data.copy_(param)



    def forward(self, state):
        return self.network(state)


    def getSample(self):
        count = min(BATCH_SIZE, len(self.memoryItems))
        batch = random.sample(self.memoryItems, count)

        states = torch.from_numpy(np.array([arr[0] for arr in batch])).to(torch.float32).to(self.device)
        actions = (torch.asarray([arr[1] for arr in batch])).to(torch.float32).to(self.device)
        rewards = torch.from_numpy(np.array([arr[2] for arr in batch])).to(torch.float32).to(self.device)
        nextStates = torch.from_numpy(np.array([arr[3] for arr in batch])).to(torch.float32).to(self.device)

        return states, actions, rewards, nextStates

    def learn(self):
        count = min(BATCH_SIZE, len(self.memoryItems))
        states, actions, rewards, nextStates = self.getSample()
        self.network.train()
        forwardPredictions = self.network.forward(states)
        predicted = torch.gather(forwardPredictions, dim=1, index=actions.type(torch.int64).unsqueeze(1))
        targetForward = self.targetNetwork.forward(nextStates)
        qNetworkArgMax = torch.argmax(self.network.forward(nextStates), dim=1).to(torch.float32).to(self.device)
        targetRes = torch.gather(targetForward, dim=1, index=qNetworkArgMax.type(torch.int64).unsqueeze(1))

        targetRes = targetRes.squeeze(1)
        target = rewards + DISCOUNT_FACTOR * targetRes
        target = torch.reshape(target, (count, 1))
        #print("Target: ", target)
        lossValue = self.loss(predicted, target)
        self.optimizer.zero_grad()

        lossValue.backward()
        self.optimizer.step()

        # target network update
        for target_param, param in zip(self.targetNetwork.parameters(), self.network.parameters()):
            target_param.data.copy_(TAU * param + (1 - TAU) * target_param)



    def action(self, state):
        state = torch.from_numpy(state).to(torch.float32).to(self.device)

        if np.random.random() <= self.epsilon:
            self.epsilon *= EPSILON_FACTOR
            self.epsilon = max(MIN_EPSILON, self.epsilon)
            return random.randint(0, self.actionSpace-1)  # make a random action
        else:
            return torch.argmax(self.network.forward(state))

    def predictAction(self, state):
        state = torch.from_numpy(state).to(torch.float32).to(self.device)
        return torch.argmax(self.network.forward(state))

    def saveModel(self):
        torch.save(self.network.state_dict(), "DDQN-Model-" + str(self.id) + ".pt")
        print("Model saved successfully")








