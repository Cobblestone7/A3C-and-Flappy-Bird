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
        elif name in ['t_max', 'max episodes', 'number of threads', 'number of realizations']:
            parameters[name] = int(float(value))
        else:
            parameters[name] = value
    f.close()
    return parameters


def load_session(path):
    parameters = load.read_config(os.oath.join(path, 'config.txt'))

    for realization in os.listdir(path):
        if os.path.isfile(realization):
            continue
        f = open(os.path.join(realization, 'networks.txt'), 'rb')
        network = pickle.load(f)
        f.close()
        parameters['rel_path'] = realization
        A3C.main(parameters, network=network)


def new_session(parameters, path):
    os.mkdir(path)
    f = open(os.path.join(path, 'config.txt'), 'w+')
    for key, value in parameters.items():
        if key == 'load session':
            continue
        line = key + ' : ' + str(value)
        f.write(line)
    f.close()

    for nr in range(1, parameters['number of realizations'] + 1):
        parameters['rel_path'] = os.path.join(main_path, 'Realization_' + str(nr))
        A3C.main(parameters)


def main():
    # Read parameters from config file
    parameters = load.read_config('config.txt')
    main_path = parameters['main session']
    load_path = parameters['load session']

    # Create new session or load old one
    if load_path == '':
        new_session(parameters, main_path)
    else:
        load_session(load_path)


if __name__ == '__main__':
    main()