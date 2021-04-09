import A3C
import os
import pickle


def read_config(filename):
    """Reads parameters from the config file."""

    # Read the parameters
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
        elif name in ['t_max', 'max episodes', 'number of threads', 'number of realizations', 'max load episodes']:
            parameters[name] = int(float(value))
        else:
            parameters[name] = value
    f.close()
    return parameters


def load_session(path, max_episodes):
    parameters = read_config(os.path.join(path, 'config.txt'))
    parameters['max episodes'] = max_episodes

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
    # Create root folder
    os.mkdir(path)

    # Add the config
    f = open(os.path.join(path, 'config.txt'), 'w+')
    for key, value in parameters.items():
        if key == 'load session':
            continue
        line = key + ': ' + str(value) + '\n'
        f.write(line)
    f.close()

    # Create realization folder and run A3C for every realization
    for nr in range(1, parameters['number of realizations'] + 1):
        rel_path = os.path.join(path, 'realization_' + str(nr))
        # Create folders
        os.mkdir(rel_path)
        for folder in ['score_plot', 'conv_plot', 'prob_plot']:
            os.mkdir(os.path.join(rel_path, folder))
        # Run A3C for every realization
        parameters['rel_path'] = rel_path
        A3C.main(parameters)


def main():
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
