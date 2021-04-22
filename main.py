import A3C_lamda
import os
import pickle


def read_config(filename):
    """Read parameters from the config file and convert them
    to the proper data type. Returns the parameters in a dictionary."""
    f = open(filename, 'r')
    parameters = {}
    while True:
        line = f.readline()
        if line == "":
            break
        elif line == '\n' or line[0] == '#':
            continue
        variable_list = line.split(':')
        name = variable_list[0].strip()
        value = variable_list[1].strip()
        if name in ['learning rate', 'gamma', 'entropy regularization factor']:
            parameters[name] = float(value)
        elif name in ['t_max', 'max episodes', 'number of threads',
                     'number of realizations', 'max load episodes', 'probplot frequency']:
            parameters[name] = int(float(value))
        else:
            parameters[name] = value
    f.close()
    return parameters


def load_session(path, max_episodes):
    """Import parameters and networks from an old session and continue
    where it ended. Train every realization an additional
    max_episodes times."""
    # Import parameters and set max episodes
    parameters = read_config(os.path.join(path, 'config.txt'))
    parameters['max episodes'] = max_episodes

    # Continue the algorithm for every realization in the loaded session
    for realization in os.listdir(path):
        if os.path.isfile(realization):
            continue
        rel_path = os.path.join(path, realization)
        f = open(os.path.join(rel_path, 'network.txt'), 'rb')
        network = pickle.load(f)
        f.close()
        parameters['rel_path'] = rel_path
        A3C.main(parameters, network=network)


def new_session(parameters, path):
    """Create a new session and run the algorithm once per realization."""
    # Create root folder
    os.mkdir(path)

    # Create the config
    f = open(os.path.join(path, 'config.txt'), 'w+')
    for key, value in parameters.items():
        if key == 'load session':
            continue
        line = key + ': ' + str(value) + '\n'
        f.write(line)
    f.close()

    # Create realization folders and run A3C for every realization
    for nr in range(1, parameters['number of realizations'] + 1):
        rel_path = os.path.join(path, 'realization_' + str(nr))

        # Create folders
        os.mkdir(rel_path)
        for folder in ['score_plot', 'conv_plot', 'prob_plot']:
            os.mkdir(os.path.join(rel_path, folder))

        # Create worker folders in prob_plot
        for i in range(parameters['number of threads']):
            os.mkdir(os.path.join(rel_path, 'prob_plot', 'w%02i' % i))

        # Run A3C for every realization
        parameters['rel_path'] = rel_path
        A3C.main(parameters)


def main():
    """Run a new session or continue an old one depending on the
    specifications in the config."""
    # Read parameters from config file
    parameters = read_config('config.txt')
    main_path = parameters['session name']
    load_path = parameters['load session']

    # Create new session or load old one
    if load_path == '':
        new_session(parameters, main_path)
    else:
        load_session(load_path, parameters['max load episodes'])


if __name__ == '__main__':
    main()
