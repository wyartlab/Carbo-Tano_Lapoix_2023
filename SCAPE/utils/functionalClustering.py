from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def build_neuron_traces(Exp, Cells):

    for i, Cell in enumerate(Cells):
        spks_trace = Cell.spks.copy()
        dff_trace = Cell.dff.copy()
        if i == 0:
            array_all_spks = spks_trace
            array_all_dff = dff_trace
        else:
            array_all_spks = np.vstack((array_all_spks, spks_trace))
            array_all_dff = np.vstack((array_all_dff, dff_trace))

    setattr(Exp, 'spks_traces', array_all_spks)
    setattr(Exp, 'dff_traces', array_all_dff)


def find_max_cross_corr(i, j, df, n_lags):
    all_lags = list(range(-n_lags, n_lags))
    corr_values = [df.iloc[i].corr(df.iloc[j].shift(n)) for n in all_lags]
    max_index, max_corr = np.argmax(corr_values), np.max(corr_values)
    return all_lags[max_index], max_corr


def compute_cross_corr_matrix(neuron_traces, lags_matrix, corr_matrix, lags):
    for i in range(neuron_traces.shape[0]):
        for j in range(neuron_traces.shape[0]):
            if np.isnan(lags_matrix[j,i]):
                lags_matrix[i, j], corr_matrix[i, j] = find_max_cross_corr(i, j, pd.DataFrame(neuron_traces), lags)
    return corr_matrix, lags_matrix


def get_corr_matrix(neuron_traces: np.array, cross_corr=False, lags=2):
    if not cross_corr:
        corr_matrix = pd.DataFrame(neuron_traces).T.corr()
        lags_matrix = np.nan
    else:
        corr_matrix = np.zeros((neuron_traces.shape[0], neuron_traces.shape[0]))
        corr_matrix[:] = np.nan
        lags_matrix = corr_matrix.copy()
        corr_matrix, lags_matrix = compute_cross_corr_matrix(neuron_traces, lags_matrix, corr_matrix, lags)
        ## need to complete matrix
        corr_matrix = np.nan_to_num(corr_matrix) + np.nan_to_num(corr_matrix).T - np.diag(np.diag(np.nan_to_num(corr_matrix)))

    return corr_matrix, lags_matrix


def plot_raster(neuron_traces, vmax, ylabel):
    n_neurons = neuron_traces.shape[0]
    plt.figure(figsize=(15, int(n_neurons / 10) + 1))
    plt.imshow(neuron_traces, aspect="auto", vmin=0, vmax=vmax)
    plt.ylabel(ylabel)
    # ax[0].set_yticks(np.arange(neuron_names.shape[0]))
    # ax[0].set_yticklabels(neuron_names)
    plt.colorbar()
    plt.show()


def plot_raster_concat(neuron_traces, labels, vmax, figPath=None):
    nClusters = len(set(labels))

    fig, ax = plt.subplots(figsize=(15, int(neuron_traces.shape[0] / 10) + 1))
    im = ax.imshow(np.concatenate([neuron_traces[labels == cluster] for cluster in range(nClusters)]),
                   aspect="auto",
                   vmin=0, vmax=vmax, cmap='Greys')

    previous_length = 0
    for cluster in range(nClusters):
        ax.plot([0, neuron_traces.shape[1]],
                [len(neuron_traces[labels == cluster]) + previous_length] * 2,
                '--', color='grey')
        previous_length += len(neuron_traces[labels == cluster])
    ax.set_ylabel(str(nClusters) + ' clusters')
    plt.tight_layout()
    if figPath is not None:
        fig.savefig(figPath + '/concat_rasterplot.svg')
    # fig.savefig(fig_path + '/rasterplot_clusters.svg')
    fig, ax = plt.subplots(figsize=(2, 6))
    plt.colorbar(im, ax=ax)
    ax.remove()

    # plt.savefig(fig_path + '/colorbar_rasterplot_clusters.svg')
    # matplotlib.image.imsave(fig_path + '/rasterplot.png',
    #                         np.concatenate([neuron_traces[labels == cluster] for cluster in range(nClusters)]))


def plot_traces_cluster(neuron_traces, labels, cmap, time_indices_bh, tail_angle, time_indices_ci,
                        plotSingleTrace=False, figPath=None):

    nClusters = len(set(labels))
    cells = np.arange(neuron_traces.shape[0])
    colors = [cmap(i / 10) for i in range(nClusters)]

    for label in set(labels):
        plt.figure(figsize=(15, 10))
        plt.plot(time_indices_bh, tail_angle, 'k')
        plt.title('In cluster ' + str(label), y=1.05)
        plt.xlabel('Time [s]')
        plt.ylabel('DFF')

        if plotSingleTrace:
            for i, cell in enumerate(np.array(cells)[labels == label]):
                plt.plot(time_indices_ci, neuron_traces[cell, :] - i * 30, label='cell ' + str(cell),
                         color=colors[label])
        else:
            i = 0

        plt.grid(b=None)

        sum_trace = np.sum(neuron_traces[np.array(cells)[labels == label]], axis=0) / len(np.array(cells)[labels == label])
        plt.plot(time_indices_ci, sum_trace - (i + 1) * 30, label='summed trace',
                 color='silver')

        plt.tight_layout()
        if figPath is not None:
            plt.savefig(figPath + '/traces_cluster_{}.svg'.format(label))


def plot_stats_cluster(neuron_traces, labels, time_indices_ci, time_indices_bh, tail_angle, cmap, figPath=None):
    cells = np.arange(neuron_traces.shape[0])

    df_clusters = pd.DataFrame({'cell': np.repeat(cells, neuron_traces.shape[1]),
                                'label': np.repeat(labels, neuron_traces.shape[1]),
                                'time_point': np.tile(time_indices_ci, len(cells)),
                                'dff': neuron_traces.flatten(),
                                })
    fig, ax = plt.subplots(figsize=(15,10))
    ax.plot(time_indices_bh, tail_angle/30+5, color='silver', label='tail angle / 30 [°]')
    for i in range(max(labels)+1):
        temp_df = df_clusters[df_clusters.label == i].copy()
        temp_df['dff'] = temp_df['dff']-i*5
        sns.lineplot(data=temp_df,
                     x='time_point', y='dff', ax=ax, ci='sd')

    plt.tight_layout()
    if figPath is not None:
        fig.savefig(figPath + '/stats_clusters.svg')


def plot_pos_clusters(labels, Cells, Exp, figPath=None):

    cells = np.arange(len(labels))

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olivedrab', 'cyan']
    all_planes = list(Exp.suite2p_outputs.keys())
    all_planes.sort()
    fig, ax = plt.subplots(6,5, figsize=(20, 15))
    for i, key in enumerate(all_planes):
        ax.flatten()[i].imshow(Exp.suite2p_outputs[key]['ops']['meanImg'], cmap='Greys')
    for i, cell in enumerate(cells):
        plane = Cells[i].plane
        axis = np.where(all_planes == plane)[0][0]
        ax.flatten()[axis].plot(Cells[i].y_pos, Cells[i].x_pos, 'o', color=colors[labels[i]])
    fig.suptitle('Automatic clustering')
    plt.tight_layout()
    if figPath is not None:
        fig.savefig(figPath + '/position_clusters.svg')


def runAggloClustering_vizu(nClusters, cmap, corr_dff, vmax, Exp, Cells, root_figPath=None):

    neuron_traces =  Exp.dff_traces

    if root_figPath is not None:
        figPath = os.path.join(root_figPath, '{}_clusters'.format(nClusters))
        if not os.path.isdir(figPath):
            os.mkdir(figPath)
    else:
        figPath = None

    clustering = AgglomerativeClustering(n_clusters=nClusters, ).fit(corr_dff)

    # Plot dendogram

    dend = dendrogram(linkage(corr_dff, method='ward'))

    # Sorting the labels assigned to each cell

    labels = clustering.labels_

    for cluster in range(nClusters):
        print('Number of cells in cluster {}: {}'.format(cluster, list(labels).count(cluster)))
        plot_raster(neuron_traces[labels == cluster], vmax, 'cluster ' + str(cluster))

    # Visualisation of cell activity in each cluster

    plot_raster_concat(neuron_traces, labels, vmax, figPath=figPath)

    # Plot traces in each cluster

    # plot_traces_cluster(neuron_traces, labels, cmap, Exp.time_indices_bh, Exp.tail_angle, Exp.time_indices_SCAPE,
    #                     plotSingleTrace=False)

    # Plot mean and std in each cluster
    cells = np.arange(neuron_traces.shape[0])
    df_clusters = pd.DataFrame({'cell': np.repeat(cells, neuron_traces.shape[1]),
                                'label': np.repeat(labels, neuron_traces.shape[1]),
                                'time_point': np.tile(Exp.time_indices_SCAPE.copy(), len(cells)),
                                'dff': neuron_traces.flatten(),
                                })
    plot_stats_cluster(neuron_traces, labels, Exp.time_indices_SCAPE, Exp.time_indices_bh, Exp.tail_angle.copy(), cmap, figPath=figPath)

    plot_pos_clusters(labels, Cells, Exp, figPath=figPath)

    return labels