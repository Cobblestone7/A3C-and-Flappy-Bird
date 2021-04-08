import A3C
import load
import os
import pickle


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


def main():
    # Read parameters from config file
    parameters = load.read_config('config.txt')
    main_path = parameters['main session']
    load_path = parameters['load session']

    # Create new session or load old one
    if load_path == '':
        os.mkdir(main_path)
        for nr in range(1, parameters['number of realizations'] + 1):
            parameters['rel_path'] = os.path.join(main_path, 'Realization_' + str(nr))
            A3C.main(parameters)
    else:
        load_session(load_path)


if __name__ == '__main__':
    main()