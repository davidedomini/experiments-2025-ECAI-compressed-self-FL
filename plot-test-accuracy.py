import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd
import matplotlib
import glob


def parse_areas(name):
    for elem in name.split('_'):
        if 'areas' in elem:
            n_areas = int(elem[-1])
            return n_areas

def parse_sparsity(name):
    for elem in name.split('_'):
        if 'sparsity' in elem:
            sparsity = float(elem.split('-')[-1])
            return sparsity

def get_data(directory, threshold):
    files = glob.glob(f'{directory}/*_lossThreshold-{threshold}*.csv')
    df = pd.DataFrame(columns=['Test accuracy', 'Areas', 'Algorithm'])
    print(f'For algorithm {algorithm} found {len(files)} files')
    for f in files:
        areas = parse_areas(f)
        sparsity = parse_sparsity(f)
        acc = pd.read_csv(f).iloc[0].mean()
        df = df._append({'Test accuracy': acc, 'Areas': areas, 'Sparsity': sparsity}, ignore_index=True)
    return df

if __name__ == '__main__':

    output_directory = 'charts/test'
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams.update({'axes.titlesize': 52})
    matplotlib.rcParams.update({'axes.labelsize': 52})
    matplotlib.rcParams.update({'xtick.labelsize': 46})
    matplotlib.rcParams.update({'ytick.labelsize': 46})
    matplotlib.rcParams.update({'legend.fontsize': 34})
    matplotlib.rcParams.update({'legend.title_fontsize': 34})
    plt.rcParams.update({'text.usetex': True})
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')

    data_baseline = get_data('data-test-baseline/*.csv', 'Baseline')

    data_self_fl = {}

    for sparsity in [0.5, 0.9, 0.95,0.99]:
        for th in [20, 40, 80]:
            d = get_data(f'data-test/*sparsity-{sparsity}_lossThreshold-{th}.0.csv', 'Self-FL')
            data_self_fl[th] = d

        for th in data_self_fl.keys():
            plt.figure(figsize=(12, 8))
            data_comparison = pd.concat([data_baseline, data_self_fl[th]])
            # sns.color_palette('viridis', as_cmap=True)
            # sns.set_palette('viridis')
            colors = sns.color_palette("viridis", as_cmap=True)
            palette = [colors(0.1), colors(0.9)]
            # viridis = plt.colormaps['viridis']
            ax = sns.boxplot(data=data_comparison, x='Areas', y='Test accuracy', hue='Algorithm', palette=palette, linewidth=2, fill=False)
            sns.move_legend(ax, 'lower left')
            plt.title(f'$ \psi = {sparsity}$')
            plt.ylabel('$Accuracy - Test$')
            plt.ylim(0, 1)
            ax.yaxis.grid(True)
            ax.xaxis.grid(True)
            plt.tight_layout()
            plt.savefig(f'{output_directory}/test-accuracy-comparison-threshold-{th}_sparsity-{sparsity}.pdf', dpi=500)
            plt.close()
