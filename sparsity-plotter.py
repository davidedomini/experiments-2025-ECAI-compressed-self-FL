import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import glob
import re


def extractVariableNames(filename):
    """
    Gets the variable names from the Alchemist data files header.

    Parameters
    ----------
    filename : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    with open(filename, 'r') as file:
        dataBegin = re.compile(r'\d')
        lastHeaderLine = ''
        for line in file:
            if dataBegin.match(line[0]):
                break
            else:
                lastHeaderLine = line
        if lastHeaderLine:
            regex = re.compile(r' (?P<varName>\S+)')
            return regex.findall(lastHeaderLine)
        return []


def openCsv(path):
    """
    Converts an Alchemist export file into a list of lists representing the matrix of values.

    Parameters
    ----------
    path : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    regex = re.compile(r'\d')
    with open(path, 'r') as file:
        lines = filter(lambda x: regex.match(x[0]), file.readlines())
        return [[float(x) for x in line.split()] for line in lines]


def load_data_from_csv(path, threshold, sparsity):  
    files = glob.glob(f'{path}experiment*sparsity-{sparsity}_*lossThreshold-{threshold}*.csv')
    dataframes = []
    print(f'For thershold {threshold} and sparsity {sparsity} loaded {len(files)} files')
    for file in files:
        areas_value = int([part for part in file.split('_') if part.startswith("areas-")][0].split('-')[1])
        columns = extractVariableNames(file)
        data = openCsv(file)
        df = pd.DataFrame(data, columns=columns)
        df['Areas'] = areas_value
        dataframes.append(df)
    return dataframes


def add_seed(data):
    for seed, df in enumerate(data):
        df['seed'] = seed
    return data

def metric_to_symbol(metric):
    symbol = ''
    if 'TrainLoss' in metric:
        symbol = '$NLL - Train$'
    elif 'ValidationLoss' in metric:
        symbol = '$NLL - Validation$'
    elif 'ValidationAccuracy' in metric:
        symbol = '$Accuracy - Validation$'
    elif 'AreaCount' in metric:
        symbol = '$|F|$'
    elif 'AreaCorrectness' in metric:
        symbol = r'$\Diamond$'
    else:
        symbol = metric
    return symbol

def plot(data, sparsity, threshold, metric, out_dir):
    areas_values =[3, 5, 9]
    viridis = plt.colormaps['viridis']
    indexes = np.linspace(0.1, 0.9, len(areas_values))

    for i, areas in enumerate(areas_values):
        data_filtered = data[data['Areas'] == areas]
        df_grouped = data_filtered.groupby('time')[metric].agg(['mean', 'std'])
        time = df_grouped.index
        mean = df_grouped['mean']
        std = df_grouped['std']
        plt.plot(time, mean, color=viridis(indexes[i]), label=f'{areas}', linewidth=2)
        plt.fill_between(time, mean - std, mean + std, color=viridis(indexes[i]), alpha=0.2)

    plt.title(f'$\psi$ = {sparsity}')
    plt.xlabel("Time")
    plt.ylabel(metric_to_symbol(metric))
    plt.legend(title="Areas")
    plt.ylim(0, 1)
    plt.tight_layout()
    # plt.grid(True)
    plt.savefig(f'{out_dir}/{metric}-sparsity-{sparsity}-threshold {threshold}.pdf')
    plt.close()


matplotlib.rcParams.update({'axes.titlesize': 30})
matplotlib.rcParams.update({'axes.labelsize': 28})
matplotlib.rcParams.update({'xtick.labelsize': 25})
matplotlib.rcParams.update({'ytick.labelsize': 25})
matplotlib.rcParams.update({'legend.fontsize': 20})
matplotlib.rcParams.update({'legend.title_fontsize': 22})
plt.rcParams.update({'text.usetex': True})
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')


charts_dir = 'charts/sparsity-gfl/'
Path(charts_dir).mkdir(parents=True, exist_ok=True)
path = 'data/'
threshold = [20.0, 40.0, 80.0]
sparsity = [0.5, 0.9, 0.95, 0.99]
# metrics = ['AreaCount', 'AreaCorrectness', 'TrainLoss[mean]', 'ValidationLoss[mean]', 'ValidationAccuracy[mean]']
metrics = ['AreaCount', 'ValidationAccuracy[mean]']
for t in threshold:
    for s in sparsity:
        data = load_data_from_csv(path, t, s)
        data = add_seed(data)
        data = pd.concat(data, ignore_index=True)
        data = data.dropna()
        for metric in metrics:
            plot(data, s, t, metric, charts_dir)
    


    