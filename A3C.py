import torch
from torch import nn
import torch.nn.functional as F
import flappy_bird_gym
#from flappy_bird_env_simple import FlappyBirdEnvSimple
import gym
import torch.multiprocessing as mp
import os
import pickle


def t(m_array):
    """Converts a multidimensional array to a torch tensor and returns it."""
    return torch.Tensor(m_array)


class SharedAdam(torch.optim.Adam):
    """An Adam optimizer shared by all workers."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Worker(mp.Process):
    """A single reinforcement agent worker."""

    def __init__(self, parameters, i):
        super(Worker, self).__init__()

        # Environment, network, optimizer
        self.env = gym.make(parameters['environment name'])
        self.local_actor_critic = ActorCritic([self.env.observation_space.shape[0]], self.env.action_space.n)
        self.global_actor_critic = parameters['global actor critic']
        self.optimizer = parameters['optimizer']

        # Constants
        self.max_episodes = parameters['max episodes']
        self.t_max = parameters['t_max']
        self.entropy_reg_factor = parameters['entropy regularization factor']
        self.gamma = parameters['gamma']

        # Miscellaneous
        self.memory = Memory()
        self.episode_idx = parameters['global episode index']
        self.name = 'w%02i' % i

        # Initialize the local network
        self.local_actor_critic.load_state_dict(
            self.global_actor_critic.state_dict())

        # Initialize Data Storage
        self.data = Data_storage(parameters['rel_path'], parameters['environment name'])

    def run(self):
        for i in range(self.max_episodes):
            done = False
            total_reward = 0
            observation = self.env.reset()
            steps = 1
            render_ep = 10
            #if i % render_ep == 0:
            #    render = True
            #else:
            #    render = False
            while not done: # Until episode is done
                # Sample an action based on state
                state = torch.tensor([observation], dtype=torch.float)
                probs, value = self.local_actor_critic.forward(state)
                probs = torch.softmax(probs, dim=1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().numpy()[0]
                # Simulate environment
                observation_, reward, done, info = self.env.step(action)
                accumulated_reward = reward
                for i in range(3):
                    if done:
                        break
                    observation_, reward, done, info = self.env.step(0)
                    accumulated_reward += reward

                total_reward += accumulated_reward
                # Add data to memory
                self.memory.add(observation, action, accumulated_reward, value)
                self.data.add_prob(probs)
                # Call training
                if steps % self.t_max == 0 or done:
                    # Train locally
                    self.train(done)

                    # Update global networks and copy to local
                    self.update_global()

                    # Reset memory
                    self.memory.clear()

                # Iterate to next state
                steps += 1
                observation = observation_
                #if render:
                #    self.env.render()
                #    time.sleep(1/30)
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode', self.episode_idx.value, total_reward)
            self.data.add_score(total_reward)
            self.data.add_conv(self.local_actor_critic)

            if len(self.data.score_plot) > 100000:
                self.data.save_score(self.name)
                self.data.save_conv(self.name)
                self.data.save_prob(self.name)
                self.data.clear_plot()
        self.data.save_score(self.name)
        self.data.save_conv(self.name)
        self.data.save_prob(self.name)
        self.data.clear_plot()
        if self.name == 'w00':
            self.data.save_net(self.global_actor_critic)

    def train(self, done):
        """Trains the neural networks (actor and critic) based on the
            memory of the previous episode."""

        states = torch.tensor(self.memory.states, dtype=torch.float)
        actions = torch.tensor(self.memory.actions, dtype=torch.float)

        probs, values = self.local_actor_critic.forward(states)

        R = values[-1]*(1-int(done))

        batch_return = []
        for reward in self.memory.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float)

        returns = batch_return

        values = values.squeeze()
        critic_loss = (returns-values)**2

        probs = torch.softmax(probs, dim=1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)
        entropy = -torch.mul(probs, torch.log(probs)).mean()
        total_loss = (critic_loss + actor_loss).mean() - entropy * self.entropy_reg_factor

        #for param in self.local_actor_critic.parameters():
        #    square = param.data.pow(2)
        #    total_loss = total_loss + torch.mean(square) * 20

        self.optimizer.zero_grad()
        total_loss.backward()

    def update_global(self):
        """Updates the global networks."""
        # Transfer gradient
        for local_param, global_param in zip(self.local_actor_critic.parameters(),
                                             self.global_actor_critic.parameters()):
            global_param._grad = local_param.grad
        # Updates global network based on gradients
        self.optimizer.step()
        # Copies global network to local one
        self.local_actor_critic.load_state_dict(
            self.global_actor_critic.state_dict())


class Memory():
    """The memory of a worker."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []

    def add(self, state, action, reward, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []


class Data_storage():
    """Stores and loads data not directly required for the algorithm to run."""

    def __init__(self, path, env_name):

        # path for data storage
        self.path = path

        # Generates test states for the convergence plot dependent on the environment
        self.test_states = self._teststates(env_name)

        # Initialize lists for plot data
        self.score_plot = []
        self.conv_plot = []
        self.prob_plot = []

    def clear_plot(self):
        self.score_plot = []
        self.conv_plot = []
        self.prob_plot = []

    def add_score(self, total_reward):
        self.score_plot.append(total_reward)

    def add_conv(self, local_actor_critic):
        temp_list = []
        for state in self.test_states:
            _, value = local_actor_critic(t(state))
            temp_list.append(value)
        self.conv_plot.append(torch.mean(t(temp_list)))

    def add_prob(self, probs):
        self.prob_plot.append(probs.squeeze()[0])

    def save_score(self, name):
        folderpath = os.path.join(self.path, 'score_plot')
        filepath = os.path.join(folderpath, name + '.txt')
        f = open(filepath, 'a+')
        for element in self.score_plot:
            f.write(str(element) + '\n')
        f.close()

    def save_conv(self, name):
        folderpath = os.path.join(self.path, 'conv_plot')
        filepath = os.path.join(folderpath, name + '.txt')
        f = open(filepath, 'a+')
        for element in self.conv_plot:
            f.write(str(element.item()) + '\n')
        f.close()

    def save_prob(self, name):
        folderpath = os.path.join(self.path, 'prob_plot')
        filepath = os.path.join(folderpath, name + '.txt')
        f = open(filepath, 'a+')
        for element in self.prob_plot:
            f.write(str(element.item()) + '\n')
        f.close()

    def save_net(self, network):
        filepath = os.path.join(self.path, 'network.txt')
        f = open(filepath, 'wb')
        pickle.dump(network, f)
        f.close()

    def _teststates(self, env_name):
        """Generates test states for the convergence plot dependent on the environment."""

        teststates = []
        if env_name == 'FlappyBird-v0':
            for i in range(0, 280, 28):
                for j in range(0, 510, 51):
                    teststates.append(t([i, j]))
        else:
            for i in range(-4, 4, 1):
                for j in range(-10, 10, 5):
                    for k in range(-418, 418, 200):
                        for l in range(-10, 10, 5):
                            teststates.append(t([i, j, k / float(1000), l]))
        return teststates


class ActorCritic(nn.Module):
    """A neural network which works as a function approximator for the policy."""

    def __init__(self, state_dim, n_actions):
        """Initiates the neural network."""
        super().__init__()
        #self.pi1 = nn.Linear(*state_dim, 128)
        #self.v1 = nn.Linear(*state_dim, 128)
        #self.pi = nn.Linear(64, n_actions)
        #self.v = nn.Linear(64, 1)
        self.first = nn.Linear(*state_dim, 128)
        self.critic_mid = nn.Linear(128, 64)
        self.actor_mid = nn.Linear(128, 64)
        self.critic = nn.Linear(64, 1)
        self.actor = nn.Linear(64, n_actions)
    def forward(self, state):
        """Samples an action from the policy based on the current state."""
        #pi1 = F.relu(self.pi1(state))
        #v1 = F.relu(self.v1(state))

        #pi = self.pi(pi2)
        #v = self.v(pi2)

        first = F.relu(self.first(state))
        actor_mid = torch.tanh(self.actor_mid(first))
        critic_mid = torch.tanh(self.critic_mid(first))
        actor = self.actor(actor_mid)
        critic = self.critic(critic_mid)

        #return pi, v
        return actor, critic


def main(parameters, network=None):
    """Initializes relevant parameters and starts the workers."""

    # Calculates state- and action space dimensions
    temp_env = gym.make(parameters['environment name'])

    # Creates a global network or loads a new one
    if network:
        parameters['global actor critic'] = network
    else:
        parameters['global actor critic'] = ActorCritic([temp_env.observation_space.shape[0]], temp_env.action_space.n)

    # Shares the memory of the global network
    parameters['global actor critic'].share_memory()

    # Creates a shared Adam
    parameters['optimizer'] = SharedAdam(parameters['global actor critic'].parameters(), lr=parameters['learning rate'])

    # Global episode count
    parameters['global episode index'] = global_ep = mp.Value('i', 0)

    workers = [Worker(parameters, i)
               for i in range(parameters['number of threads'])]
    [w.start() for w in workers]
    [w.join() for w in workers]
