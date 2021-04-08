import pickle
import os.path
import sys
import gym
#from flappy_bird_env_simple import FlappyBirdEnvSimple

def read_config(filename):
    """Reads parameters from the config file."""
    # Read the parameters
    f = open(filename, 'r')
    parameters = {}
    while True:
        line = f.readline()
        if line == "":
            break
        variable_list = line.split(':')
        parameters[variable_list[0].strip()] = variable_list[1].strip()
    f.close()

    # Store the parameters
    learning_rate = float(parameters['learning rate'])
    gamma = float(parameters['gamma'])
    t_max = int(parameters['t_max'])
    max_episodes = int(float(parameters['max episodes']))
    store_session = parameters['store session']
    load_session = parameters['load session']
    environment_name = parameters['environment name']
    env = gym.make(environment_name)
    #env = FlappyBirdEnvSimple()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    plot_fold = parameters['directory for plots']
    conv_plot_fold = parameters['directory for conv_plots']
    prob_plot_fold = parameters['directory for prob_plots']
    entropy_reg_factor = float(parameters['entropy regularization factor'])
    n_workers = int(parameters['number of threads'])
    path = os.getcwd()
    # Check folders
    if not os.path.exists(path + '/' + plot_fold):
        os.mkdir(path + '/' + plot_fold)
    if not os.path.exists(path + '/' + conv_plot_fold):
        os.mkdir(path + '/' + conv_plot_fold)
    if not os.path.exists(path + '/' + prob_plot_fold):
        os.mkdir(path + '/' + prob_plot_fold)

    # Warns if storage files are missing
    if store_session == "" or plot_fold == "" or conv_plot_fold == "" or prob_plot_fold == "":
        print("One or more storage filenames are missing. Data might not be saved properly.")
        iterate = [store_session, plot_fold, conv_plot_fold, prob_plot_fold]
        for i in range(4):
            if iterate[i] == "":
                iterate[i] = False
    return learning_rate, gamma, t_max, max_episodes, load_session, state_dim, n_actions, environment_name, plot_fold, conv_plot_fold, store_session, path, prob_plot_fold, entropy_reg_factor, n_workers



def load_session(filename):
    """Loads an old session or creates a new one if none is available."""
    if os.path.isfile(filename):
        f_global_actor_critic = open(filename, 'rb')
        global_actor_critic = pickle.load(f_global_actor_critic)
        f_global_actor_critic.close()
    else:
        print("Couldn't find old session. Creating new session instead.")
        return False
    return global_actor_critic
