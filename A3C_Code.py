import torch
from torch import nn
import flappy_bird_gym
import gym
import torch.multiprocessing as mp


class SharedAdam(torch.optim.Adam):
    """An Adam optimizer shared by all workers."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0):
        """Creates an Adam optimizer."""
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
    def __init__(self,
                    environment_name,
                    learning_rate,
                    gamma,
                    t_max,
                    max_episodes,
                    entropy_regularization_factor,
                    number_of_workers,
                    lamda,
                    global_actor_critic,
                    optimizer,
                    global_episode_index,
                    i
                    ):
        """Creates a worker."""
        super(Worker, self).__init__()
        # Environment, networks, optimizer
        self.env = gym.make(environment_name)
        self.local_actor_critic = ActorCritic([self.env.observation_space.shape[0]],
                                                self.env.action_space.n)
        self.global_actor_critic = global_actor_critic
        self.optimizer = optimizer

        # Constants
        self.max_episodes = max_episodes
        self.t_max = t_max
        self.entropy_reg_factor = entropy_regularization_factor
        self.gamma = gamma
        self.lamda = lamda

        # Memory
        self.memory = Memory()

        # Worker id and episode index
        self.episode_idx = global_episode_index
        self.name = 'w%02i' % i

        # Initialize the local network
        self.local_actor_critic.load_state_dict(
            self.global_actor_critic.state_dict())


    def run(self):
        """The tasks of a single worker.
        The workers interact with the environment here."""
        for i in range(1, self.max_episodes + 1): # For every episode
            # Reset the parameters and the environment
            done = False
            total_reward = 0
            observation = self.env.reset()
            steps = 1

            while not done: # Until episode is done
                # Sample an action based on state
                state = torch.tensor([observation], dtype=torch.float)
                probs, value = self.local_actor_critic.forward(state)
                probs = torch.softmax(probs, dim=1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().numpy()[0]

                # Simulate environment
                observation_, reward, done, info = self.env.step(action)

                # Increment total reward
                total_reward += reward

                # Add data to memory
                self.memory.add(observation, action, reward, value)

                if steps % self.t_max == 0 or done: # If episode is done or
                                                    # tmax is reached
                    # Add next state to state list
                    self.memory.states.append(observation_)
                    # Train locally
                    self.train(done)

                    # Update the global network and copy it to local network
                    self.update_global()

                    # Reset the memory
                    self.memory.clear()

                # Iterate to the next state
                steps += 1
                observation = observation_

            # Increment the episode index
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1

            # Print the result
            print(self.name, 'episode', self.episode_idx.value, total_reward)

    def train(self, done):
        """Train the neural networks (actor and critic) based on the
            recent memory."""
        # Convert states and actions to tensors
        states = torch.tensor(self.memory.states, dtype=torch.float)
        actions = torch.tensor(self.memory.actions, dtype=torch.float)

        # Calculate the action distributions and values
        probs, values = self.local_actor_critic.forward(states)

        # Approximate the value function of the last state and initial values
        # for the return and lambda return
        R = values[-1] * (1 - int(done))
        Rlamda = R

        # If terminal state, the value function is zero
        if done:
            values[-1] = 0

        # Construct return lists
        batch_return = []
        batch_return_lamda = []


        for i in range(len(self.memory.rewards) - 1, -1, -1): # For t-1 to 0
            # Calculate returns
            R = self.memory.rewards[i] + self.gamma * R
            Rlamda = self.memory.rewards[i] + self.gamma * (self.lamda * Rlamda
                    + (1 - self.lamda) * values[i+1])

            # Append to return lists
            batch_return.append(R)
            batch_return_lamda.append(Rlamda)

        # Reverse the list of returns
        batch_return.reverse()
        batch_return_lamda.reverse()

        # Convert the lists of returns to a tensor
        batch_return = torch.tensor(batch_return, dtype=torch.float)
        returns = batch_return
        batch_return_lamda = torch.tensor(batch_return_lamda, dtype=torch.float)
        returns_lamda = batch_return_lamda

        # Calculate the critic loss with lambda returns
        values = values.squeeze()
        critic_loss = (returns_lamda-values[:-1])**2

        # Calculate the log probabilites
        probs = torch.softmax(probs, dim=1)
        dist = torch.distributions.Categorical(probs[:-1])
        log_probs = dist.log_prob(actions)

        # Calculate the actor loss
        actor_loss = -log_probs*(returns-values[:-1])

        # Calculate the entropy
        entropy = -torch.mul(probs, torch.log(probs)).mean()

        # Calculate total loss
        total_loss = ((critic_loss + actor_loss).mean()
                    - entropy * self.entropy_reg_factor)

        # Reset the gradient and propagate the loss
        self.optimizer.zero_grad()
        total_loss.backward()

    def update_global(self):
        """Update the global network based on the gradient of the worker."""
        # Transfer gradient
        for local_param, global_param in zip(self.local_actor_critic.parameters(),
                                             self.global_actor_critic.parameters()):
            global_param._grad = local_param.grad

        # Update the global network based on gradient
        self.optimizer.step()

        # Copy the global network to the local network
        self.local_actor_critic.load_state_dict(
            self.global_actor_critic.state_dict())


class Memory():
    """The memory of a single worker."""
    def __init__(self):
        """Creates attributes."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []

    def add(self, state, action, reward, value):
        """Add experience to the memory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)

    def clear(self):
        """Clear the memory."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []


class ActorCritic(nn.Module):
    """A neural network which works as a function approximator
    for the policy and value function."""
    def __init__(self, state_dim, n_actions):
        """Create the neural network."""
        super().__init__()
        self.first = nn.Linear(*state_dim, 128)
        self.common_mid = nn.Linear(128, 128)
        self.critic_mid = nn.Linear(128, 64)
        self.actor_mid = nn.Linear(128, 64)
        self.critic = nn.Linear(64, 1)
        self.actor = nn.Linear(64, n_actions)

    def forward(self, state):
        """Sample an action distribution from the policy and the value
        function approximation based on the current state."""
        first = torch.tanh(self.first(state))
        common_mid = torch.tanh(self.common_mid(first))
        actor_mid = torch.tanh(self.actor_mid(common_mid))
        critic_mid = torch.tanh(self.critic_mid(common_mid))
        actor = self.actor(actor_mid)
        critic = self.critic(critic_mid)
        return actor, critic


def main():
    """Initialize relevant parameters and start the workers."""
    # Constants
    environment_name = 'CartPole-v0'
    learning_rate = 1e-4
    gamma = 0.99
    t_max = 10
    max_episodes = 1000
    entropy_regularization_factor = 0 # 0 yields normal A3C
    number_of_workers = 8
    lamda = 1 # 1 yields normal A3C

    # Create a temporary environment
    temp_env = gym.make(environment_name)

    # Create a global network or loads a new one
    global_actor_critic = ActorCritic(
    [temp_env.observation_space.shape[0]], # Dimension of observation space
    temp_env.action_space.n # Dimension of action space
    )

    # Share the memory of the global network
    global_actor_critic.share_memory()

    # Create a shared Adam optimizer
    optimizer = SharedAdam(
    global_actor_critic.parameters(),
    lr=learning_rate
    )

    # Create global episode count
    global_episode_index = mp.Value('i', 0)

    # Create workers
    workers = [Worker(environment_name,
                        learning_rate,
                        gamma,
                        t_max,
                        max_episodes,
                        entropy_regularization_factor,
                        number_of_workers,
                        lamda,
                        global_actor_critic,
                        optimizer,
                        global_episode_index,
                        i
                        )
               for i in range(number_of_workers)]

    # Start the workers
    [w.start() for w in workers]

    # Wait for workers to finish
    [w.join() for w in workers]


if __name__ == '__main__':
    main()
