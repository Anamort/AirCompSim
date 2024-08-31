import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Location import Location
from Server import Server, EdgeServer, UAV, CloudServer
from User import User

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

import logging

from torch.autograd import Variable
import torch

import shutil

torch.set_default_dtype(torch.float64)

device = torch.device('mps')
class State(object):
    id: int = 0

    def __init__(self, simTime):
        self.id = State.id
        State.id += 1
        # self.episode = episode
        self.simTime = simTime
        self.state = []  # np.zeros([])
        logging.info("SimTime: %s ---> The state %s has been created.", str(self.simTime), str(self.id))


    def getState(self):
        SIM_BOUNDRY = 100


        # uav * 4
        for uav in UAV.uavs:
            self.state.append(uav.location.x / SIM_BOUNDRY)
            self.state.append(uav.location.y / SIM_BOUNDRY)

        val = np.array(self.state)
        return val


class MemoryItem(object):
    id: int = 0
    memoryItems = []

    def __init__(self, state: State, nextState: State, reward, action, isDone: bool):
        MemoryItem.id += 1
        self.state = state
        self.nextState = nextState
        self.reward = reward
        self.action = action  # [0 - 5] 0: noMove, 1: left, 2: right, 3: up, 4: down
        self.isDone = isDone
        MemoryItem.memoryItems.append((self.state, self.action, self.reward, self.nextState))
        if len(MemoryItem.memoryItems) > 1000000:

            item = MemoryItem.memoryItems[0]
            index = 0
            #while item[2] > 0:
            #    index += 1
            #    item = MemoryItem.memoryItems[index]

            MemoryItem.memoryItems.pop(0)





    @classmethod
    def getSample(cls, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        count = min(count, len(MemoryItem.memoryItems))
        batch = random.sample(MemoryItem.memoryItems, count)

        s_arr = np.array([arr[0] for arr in batch])
        a_arr = np.array([arr[1] for arr in batch])
        r_arr = np.array([arr[2] for arr in batch])
        s1_arr = np.array([arr[3] for arr in batch])

        return s_arr, a_arr, r_arr, s1_arr




# Source code: https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/blob/master/ActorCritic/actor_critic_torch.py
class ActorCriticNetwork(nn.Module):
    # n_actions in AirSim case is : Case-1 = the number UAVs
    #                               Case-2 = the number UAVs + speed of UAVs ...
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # pi = self.pi(x)
        pi = F.sigmoid(self.pi(x)) * 5
        v = self.v(x)

        return (pi, v)


class ActorCriticAgent(object):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions,
                                               fc1_dims, fc2_dims)
        self.log_prob = None

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        # probabilities = F.softmax(probabilities, dim=1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob

        # return action.item()
        return probabilities

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        delta = reward + self.gamma * critic_value_ * (1 - int(done)) - critic_value

        actor_loss = -self.log_prob * delta
        critic_loss = delta ** 2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()


####################################  DDPG ####################################

# Source Code: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/train.py

EPS = 0.003 # 1 #0.003
BATCH_SIZE = 64
#LEARNING_RATE = 0.000000000001 # 0.000000001
LEARNING_RATE = 0.0001
GAMMA = 0.99
TAU = 0.001 #0.001

NODE_SIZE = 256

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
	    :param state_dim: Dimension of input state (int)
	    :param action_dim: Dimension of input action (int)
	    :return:
	    """
        super(Critic, self).__init__()
        self.device = T.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(self.device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim, 512)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())

        self.fcs2 = nn.Linear(512, 256)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())
        self.fca1 = nn.Linear(action_dim, 256)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc2 = nn.Linear(512, 256)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(256, 128)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        # self.fcb1 = nn.Linear(NODE_SIZE, NODE_SIZE)
        # self.fcb1.weight.data = fanin_init(self.fcb1.weight.data.size())
        #
        # self.fcb2 = nn.Linear(NODE_SIZE, NODE_SIZE)
        # self.fcb2.weight.data = fanin_init(self.fcb2.weight.data.size())
        #
        # self.fcb3 = nn.Linear(NODE_SIZE, NODE_SIZE)
        # self.fcb3.weight.data = fanin_init(self.fcb1.weight.data.size())

        #self.fcb4 = nn.Linear(NODE_SIZE, NODE_SIZE)
        #self.fcb4.weight.data = fanin_init(self.fcb1.weight.data.size())

        # self.fcb5 = nn.Linear(NODE_SIZE, NODE_SIZE)
        # self.fcb5.weight.data = fanin_init(self.fcb1.weight.data.size())
        #
        #
        self.fc4 = nn.Linear(128, 1)
        self.fc4.weight.data.uniform_(-EPS, EPS)

        self.fullyConnectedLayers = nn.Sequential(
            nn.Linear(state_dim, 512),  # [B, 8192]
            nn.ReLU(),  # [B, 8192]
            nn.Linear(512, 128),  # [B, 8192]
            nn.ReLU(),  # [B, 8192]

            nn.Linear(128, 1),  # [B, numberOfClasses]
        ) #.to(self.device)

    def forward(self, state, action):
        """
	    returns Value function Q(s,a) obtained from critic network
	    :param state: Input state (Torch Variable : [n,state_dim] )
	    :param action: Input Action (Torch Variable : [n,action_dim] )
	    :return: Value function : Q(S,a) (Torch Variable : [n,1] )
	    """
        s1 = self.fcs1(state)
        s2 = self.fcs2(s1)
        a1 = self.fca1(action)
        x = torch.cat((s2, a1), dim=1)

        x = self.fc2(x)
        x = self.fc3(x)


        x = self.fc4(x)

        return x


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_lim):
        """
	    :param state_dim: Dimension of input state (int)
	    :param action_dim: Dimension of output action (int)
	    :param action_lim: Used to limit action in [-action_lim,action_lim]
	    :return:
	    """
        super(Actor, self).__init__()
        self.device = T.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(self.device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_dim, 512)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(512, 256)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(256, 128)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc5 = nn.Linear(128, action_dim)
        self.fc5.weight.data.uniform_(-EPS, EPS)

        self.fullyConnectedLayers = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        """
	    returns policy function Pi(s) obtained from actor network
	    this function is a gaussian prob distribution for all actions
	    with mean lying in (-1,1) and sigma lying in (0,1)
	    The sampled action can , then later be rescaled
	    :param state: Input state (Torch Variable : [n,state_dim] )
	    :return: Output action (Torch Variable: [n,action_dim] )
	    """
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fcb1(x))
        # x = F.relu(self.fcb2(x))
        # x = F.relu(self.fcb3(x))
        #x = F.relu(self.fcb4(x))
        action = F.tanh(self.fc5(x))


        return action


def soft_update(target, source, tau):
    """
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    """
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
    """
	Saves the models, with all training parameters intact
	:param state:
	:param is_best:
	:param filename:
	:return:
	"""
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2): #mu=0, theta=0.15, sigma=0.2  mu=1.2, theta=1, sigma=0.3
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


# use this to plot Ornstein Uhlenbeck random motion
# if __name__ == '__main__':
#     ou = OrnsteinUhlenbeckActionNoise(1)
#     states = []
#     for i in range(1000):
#         states.append(ou.sample())
#     import matplotlib.pyplot as plt
#
#     plt.plot(states)
#     plt.show()



class Trainer(nn.Module):

    def __init__(self, state_dim, action_dim, action_lim, ram):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param ram: replay memory buffer object
        :return:
        """

        # if torch.backends.mps.is_available():
        #     mps_device = torch.device("mps")
        #     x = torch.ones(1, device=mps_device)
        #     print(x)
        # else:
        #     print("MPS device not found.")

        super(Trainer, self).__init__()
        self.device = T.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(self.device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.ram = ram
        self.iter = 0
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state))
        action = self.target_actor.forward(state).detach()
        return action.data.numpy()

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state))
        action = self.actor.forward(state).detach()
        new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
        return new_action

    def optimize(self):
        """
        :param s: current state
	    :param a: action taken
	    :param r: reward received
	    :param s1: next state
        Samples a random batch from replay memory and performs optimization
        :return:
        """

        s1, a1, r1, s2 = self.ram.getSample(BATCH_SIZE)
        s1 = Variable(torch.from_numpy(s1))
        a1 = Variable(torch.from_numpy(a1))
        r1 = Variable(torch.from_numpy(r1))
        s2 = Variable(torch.from_numpy(s2))

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2).detach()
        next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        # y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = r1 + GAMMA * next_val
        # y_pred = Q( s1, a1)
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        # compute critic loss, and update the critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1)
        loss_actor = -1 * torch.sum(self.critic.forward(s1, pred_a1))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)

        # if self.iter % 100 == 0:
        # 	print 'Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.numpy(),\
        # 		' Loss_critic :- ', loss_critic.data.numpy()
        # self.iter += 1


    def save_models(self, episode_count, score):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), 'Models/' + str(episode_count) + '_actor.pt' + str(score))
        torch.save(self.target_critic.state_dict(), 'Models/' + str(episode_count) + '_critic.pt' + str(score))
        print("Models saved successfully")

    def load_models(self, episode, score):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load('SavedMM/ForRandomized/' + str(episode) + '_actor.pt' + str(score)))
        self.critic.load_state_dict(torch.load('SavedMM/ForRandomized/' + str(episode) + '_critic.pt' + str(score) ))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print("Models loaded succesfully")
