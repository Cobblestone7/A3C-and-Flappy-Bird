import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import flappy_bird_gym
#from flappy_bird_env_simple import FlappyBirdEnvSimple
import gym
import torch.multiprocessing as mp
import load
import pickle
#import random

def t(m_array):
    """Helper function. Returns a multidimensional array to a torch tensor."""
    #return torch.from_numpy(m_array).float()
    return torch.Tensor(m_array)


class SharedAdam(torch.optim.Adam):
    """Adam optimizer shared by all workers."""
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

    def __init__(self, global_actor_critic, optimizer, env_name, gamma, name, global_ep_idx, max_episodes, t_max, plot_fold, conv_plot_fold, store_session, path, prob_plot_fold, entropy_reg_factor):
        super(Worker, self).__init__()
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        #self.env = FlappyBirdEnvSimple()
        self.state_dim = [self.env.observation_space.shape[0]]
        self.n_actions = self.env.action_space.n
        self.max_episodes = max_episodes
        self.t_max = t_max
        self.local_actor_critic = ActorCritic(self.state_dim, self.n_actions)
        self.global_actor_critic = global_actor_critic
        self.optimizer = optimizer
        self.gamma = gamma
        self.memory = Memory()
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.plot = []
        self.conv_plot = []
        self.prob_plot = []
        self.states = []
        self.plot_fold = plot_fold
        self.conv_plot_fold = conv_plot_fold
        self.prob_plot_fold = prob_plot_fold
        self.store_session = store_session
        self.path = path
        self.entropy_reg_factor = entropy_reg_factor
        self.teststates()
        self.local_actor_critic.load_state_dict(
            self.global_actor_critic.state_dict())

    def teststates(self):
        if self.env_name == 'FlappyBird-v0':
            for i in range(0, 280, 28):
                for j in range(0, 510, 51):
                    self.states.append(t([i, j]))
        else:
            for i in range(-4, 4, 1):
              for j in range(-10, 10, 5):
                  for k in range(-418, 418, 200):
                      for l in range(-10, 10, 5):
                        self.states.append(t([i, j, k / float(1000), l]))

    def add_plot(self, total_reward):
        self.plot.append(total_reward)
        temp_list = []
        for state in self.states:
            _, value = self.local_actor_critic(t(state))
            temp_list.append(value)
        self.conv_plot.append(torch.mean(t(temp_list)))

    def add_prob_plot(self, probs):
        self.prob_plot.append(probs.squeeze()[0])

    def clear_plot(self):
        self.plot = []
        self.conv_plot = []
        self.prob_plot = []

    def save_plot(self):
        filepath = self.path + '/' + self.plot_fold + '/' + self.name + '.txt'
        f = open(filepath, 'a+')
        for element in self.plot:
            f.write(str(element) + '\n')
        f.close()

    def save_conv(self):
        filepath = self.path + '/' + self.conv_plot_fold + '/' + self.name + '.txt'
        f = open(filepath, 'a+')
        for element in self.conv_plot:
            f.write(str(element.item()) + '\n')
        f.close()

    def save_prob(self):
        filepath = self.path + '/' + self.prob_plot_fold + '/' + self.name + '.txt'
        f = open(filepath, 'a+')
        for element in self.prob_plot:
            f.write(str(element.item()) + '\n')
        f.close()

    def save_net(self):
        f = open(self.path + '/' + self.store_session, 'w+b')
        pickle.dump(self.global_actor_critic, f)
        f.close()


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
                self.add_prob_plot(probs)
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
            self.add_plot(total_reward)

            if len(self.plot) > 100000:
                self.save_plot()
                self.save_conv()
                self.save_prob()
                self.clear_plot()
        self.save_plot()
        self.save_conv()
        self.save_prob()
        self.save_net()
        self.clear_plot()

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

def main():
    learning_rate, gamma, t_max, max_episodes, load_session, state_dim, n_actions, environment_name, plot_fold, conv_plot_fold, store_session, path, prob_plot_fold, entropy_reg_factor, n_workers = load.read_config('config.txt')
    state_dim = [state_dim]
    global_actor_critic = load.load_session(load_session)
    if global_actor_critic is False:
        global_actor_critic = ActorCritic(state_dim, n_actions)
    global_actor_critic.share_memory()
    lr = learning_rate
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr)
    global_ep = mp.Value('i', 0)

    workers = [Worker(global_actor_critic,
                     optim,
                     environment_name,
                     gamma,
                     i,
                     global_ep,
                     max_episodes,
                     t_max,
                     plot_fold,
                     conv_plot_fold,
                     store_session,
                     path,
                     prob_plot_fold,
                     entropy_reg_factor)
               for i in range(n_workers)]
    [w.start() for w in workers]
    [w.join() for w in workers]


if __name__ == '__main__':
    main()
