import matplotlib.pyplot as plt
import numpy as np
import os


def _import(session, i, plot_type):
    worker_name = ('w%02i' % i) + '.txt'
    data = []
    for realization in os.listdir(session):
        if os.path.isfile(realization):
            continue
        rel_path = os.path.join(session, realization)
        data_path = os.path.join(rel_path, plot_type, worker_name)
        data_list = []
        f = open(data_path, 'r')
        while True:
            element = f.readline().strip()
            if element == '':
                break
            data_list.append(float(element))
        f.close()
        data.append(data_list)
    return data


def import_prob(session, realization_nr, worker_nr, episode_nr):
    i = worker_nr
    path = os.path.join(session, 'realization_' + str(realization_nr), 'prob_plot', 'w%02i' % i)
    filename = os.path.join(path, 'episode ' + str(episode_nr) + '.txt')
    f = open(filename, 'r')
    prob_data = []
    while True:
        entry = f.readline().strip()
        if entry == '':
            break
        prob_data.append(float(entry))
    f.close()
    return prob_data

def average(session, plot_type):
    f = open(os.path.join(session, 'config.txt'), 'r')
    while True:
        line = f.readline().split(':')
        if line[0].strip() == 'number of threads':
            n = int(float(line[1].strip()))
        if line[0].strip() == 'number of realizations':
            m = int(float(line[1].strip()))
        if line[0].strip() == '':
            break
    f.close()
    average = []
    for i in range(n):
        worker = _import(session, i, plot_type)
        min = len(worker[0])
        for j in range(m):
            if len(worker[j]) < min:
                min = len(worker[j])
        for k in range(m):
            worker[k] = worker[k][0:min - 1]
        mean = np.mean(worker, axis=0)
        average.append(mean)
    min = len(average[0])
    for i in range(n):
        if len(average[i]) < min:
            min = len(average[i])
    for j in range(n):
        average[j] = average[j][0:min - 1]
    mean = np.mean(average, axis=0)
    std = np.std(average, axis=0)
    return mean, std


def plot_mean(mean, std, type, show=True, filename=False):
    title_font = {'fontname': 'Arial', 'size': '20',
                      'color': 'black', 'weight': 'normal'}
    axis_font = {'fontname': 'Arial', 'size': '18'}
    sigma_upper = mean + std
    sigma_lower = mean - std
    plt.figure()
    if type == 'probability':
        plt.title('mean ' + 'probability' + ' vs ' + 'steps', **title_font)
        plt.xlabel('step', **axis_font)
    else:
        plt.title('mean ' + type + ' vs ' + 'episodes', **title_font)
        plt.xlabel('episodes', **axis_font)
    plt.ylabel('mean' + type, **axis_font)
    x_data = [j for j in range(1, len(mean) + 1)]
    plt.plot(x_data, mean, label='mean', zorder=5, color='#242424')
    plt.fill_between(x_data, sigma_lower, sigma_upper, label='plus-minus sigma', color='#c4c7cc')
    plt.legend()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()


def plot_prob(data, labels=False, ylabel='probability to jump', show=True, filename=False):
    title_font = {'fontname': 'Arial', 'size': '20',
                  'color': 'black', 'weight': 'normal'}
    axis_font = {'fontname': 'Arial', 'size': '18'}
    plt.figure()
    plt.xlabel('steps', **axis_font)
    plt.ylabel(ylabel, **axis_font)

    for i in range(len(data)):
        y_data = data[i]
        x_data = [j for j in range(1, len(y_data) + 1)]
        if labels:
            plt.plot(x_data, y_data, label=labels[i])
        else:
            plt.plot(x_data, y_data)
    if labels:
        plt.legend()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()


mean, std = average('test', 'score_plot')
plot_mean(mean, std, 'score', show=False)

mean, std = average('test', 'conv_plot')
plot_mean(mean, std, 'convergence', show=False)

data = []
labels = []
for i in range(0, 100, 10):
    data.append(import_prob('test', 1, 0, i))
    labels.append('episode ' + str(i))
plot_prob(data, labels=labels, ylabel='probability to go right', show=True, filename=False)
