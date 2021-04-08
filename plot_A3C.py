import matplotlib.pyplot as plt
import numpy as np
import os

def import_data(directory):
    """Imports data from directory. The format is an array with x_data as first element
    and an instance of y_data for the subsequent elements. The labels are episodes, for
    the x_data, and the filenames for the y_data."""
    path = os.getcwd()
    data_path = os.path.join(path, directory)
    data_labels = ['episodes']
    files_data = []     # entry 0 is episodes, the rest are score for every worker
    for filename in os.listdir(data_path):
        score = []
        f = open(os.path.join(data_path, filename), 'r')
        while True:
            curscore = f.readline().strip()
            if curscore == '':
                break
            score.append(float(curscore))
        f.close()
        if not files_data:
            files_data.append(list(range(1, len(score) + 1)))
        assert len(score) == len(files_data[0])
        files_data.append(score)
        data_labels.append(filename.split('.')[0])
    return files_data, data_labels

def import_data1(directory):
    """Imports data from directory. The format is an array with x_data as first element
    and an instance of y_data for the subsequent elements. The labels are episodes, for
    the x_data, and the filenames for the y_data."""
    path = os.getcwd()
    data_path = os.path.join(path, directory)
    data_labels = []
    files_data = []     # entry 0 is episodes, the rest are score for every worker
    x_data = []
    for filename in os.listdir(data_path):
        score = []
        f = open(os.path.join(data_path, filename), 'r')
        while True:
            curscore = f.readline().strip()
            if curscore == '':
                break
            score.append(float(curscore))
        f.close()
        x_data.append(list(range(1, len(score) + 1)))
        files_data.append(score)
        data_labels.append(filename.split('.')[0])
    return x_data, files_data, data_labels

def plot(data, title='', xlabel='', ylabel='', labels=None, show=True, filename=False):
    """Plots the data. Data and labels have to be in the same form as output from import_data()."""

    # Font sizes in plots
    title_font = {'fontname': 'Arial', 'size': '20',
                      'color': 'black', 'weight': 'normal'}
    axis_font = {'fontname': 'Arial', 'size': '18'}

    # Plot score as a function of episodes.
    # Study variance and progress here.
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)
    plt.legend()
    if labels:
        for i in range(1, len(data)):
            plt.plot(data[0], data[i], label=labels[i])
        plt.legend()
    else:
        for i in range(1, len(data)):
            plt.plot(data[0], data[i])
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

def plot1(x_data, y_data, title='', xlabel='', ylabel='', labels=None, show=True, filename=False):
    """Plots the data. Data and labels have to be in the same form as output from import_data()."""

    # Font sizes in plots
    title_font = {'fontname': 'Arial', 'size': '20',
                      'color': 'black', 'weight': 'normal'}
    axis_font = {'fontname': 'Arial', 'size': '18'}

    # Plot score as a function of episodes.
    # Study variance and progress here.
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)
    plt.legend()
    if labels:
        for i in range(1, len(x_data)):
            plt.plot(x_data[i], y_data[i], label=labels[i])
        plt.legend()
    else:
        for i in range(1, len(x_data)):
            plt.plot(x_data[i], y_data[i])
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

def meanscore(data, k):
    """Creates meanscore data. Data has to be in the same form as output from import_data().
    The data output has the same format as input."""
    meandata = [data[0][k:len(data[0]) - k]]
    for i in range(1, len(data)):
        curr_data = []
        for j in range(k, len(data[i]) - k):
            curr_data.append(np.mean(data[i][j - k:k + j]))
        meandata.append(curr_data)
    return meandata


data, labels = import_data('plot_fold')
mean_data = meanscore(data, 10)
#plot(data, title='Reward plot', xlabel='x', ylabel='y', labels=labels, show=False)
plot(mean_data, title='Mean reward plot', xlabel='x', ylabel='y', labels=labels, show=False)

data1, labels1 = import_data('conv_plot_fold')
plot(data1, title='Convergence plot', xlabel='x', ylabel='y', labels=labels1, show=False)

x_data, y_data, labels2 = import_data1('prob_plot_fold')
plot1(x_data, y_data, title='Probability plot', xlabel='x', ylabel='y', labels=labels2, show=True)
