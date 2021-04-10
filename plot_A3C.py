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


def plot(mean, std, type, show=True, filename=False):
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
    labels = ['plus sigma', 'minus sigma', 'mean ' + type]
    y_data = [sigma_upper, sigma_lower, mean]
    for i in range(3):
        plt.plot([j for j in range(1, len(y_data[i]) + 1)], y_data[i], label=labels[i])
    plt.legend()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

mean, std = average('CartPole lr=1e-5', 'score_plot')
plot(mean, std, 'score', show=False)

mean, std = average('CartPole lr=1e-5', 'prob_plot')
plot(mean, std, 'probability', show=False)

mean, std = average('CartPole lr=1e-5', 'conv_plot')
plot(mean, std, 'convergence', show=True)
