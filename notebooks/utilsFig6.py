from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np


def plot_raster(neuron_traces, vmax, ylabel):
    n_neurons = neuron_traces.shape[0]
    plt.figure(figsize=(15, int(n_neurons / 10) + 1))
    plt.imshow(neuron_traces, aspect="auto", vmin=0, vmax=vmax)
    plt.ylabel(ylabel)
    # ax[0].set_yticks(np.arange(neuron_names.shape[0]))
    # ax[0].set_yticklabels(neuron_names)
    plt.colorbar()
    plt.show()



def runAggloClustering_vizu(nClusters, cmap, cmap_name, corr_dff, vmax, data):

    neuron_traces, dff, time_indices, stim_trace, time_indices_stim, stat, ops, cells, fig_path = data

    clustering = AgglomerativeClustering(n_clusters=nClusters, ).fit(corr_dff)
    
    # Plot dendogram
    
    dend = dendrogram(linkage(corr_dff, method='ward'))

    # Sorting the labels assigned to each cell

    labels = clustering.labels_

    for cluster in range(nClusters):
        print('Number of cells in cluster {}: {}'.format(cluster, list(labels).count(cluster)))
        plot_raster(neuron_traces[labels == cluster], vmax, 'cluster ' + str(cluster))

    # Visualisation of cell activity in each cluster

    fig, ax = plt.subplots(figsize=(15, int(neuron_traces.shape[0] / 10) + 1))
    im = ax.imshow(np.concatenate([neuron_traces[labels == cluster] for cluster in range(nClusters)]), 
                   aspect="auto", 
                   vmin=0, vmax=vmax, cmap='Greys')
    previous_length = 0
    for cluster in range(nClusters):
        ax.plot([0, neuron_traces.shape[1]], 
                [len(neuron_traces[labels == cluster])+previous_length]*2,
                '--', color='grey')
        previous_length += len(neuron_traces[labels == cluster])
    ax.set_ylabel(str(nClusters) + ' clusters')
    fig.savefig(fig_path + '/rasterplot_clusters.svg')
    fig,ax = plt.subplots(figsize=(2,6))
    plt.colorbar(im,ax=ax)
    ax.remove()
    plt.savefig(fig_path + '/colorbar_rasterplot_clusters.svg')
    import matplotlib
    matplotlib.image.imsave(fig_path + '/rasterplot.png', 
                            np.concatenate([neuron_traces[labels == cluster] for cluster in range(nClusters)]))
    print('\n traces in each cluster:')

    colors = [cmap(i / 10) for i in range(nClusters)]
    for label in set(labels):
        plt.figure(figsize=(15, 10))
        plt.plot(time_indices_stim, stim_trace + 200, 'k')
        plt.title('In cluster ' + str(label), y=1.05)
        plt.xlabel('Time [s]')
        plt.ylabel('DFF')
        for i, cell in enumerate(np.array(cells)[labels == label]):
            plt.plot(time_indices,dff[cell,:] - i * 30, label='cell ' + str(cell),
                     color=colors[label])
        plt.grid(b=None)
        
        sum_trace = np.sum(dff[np.array(cells)[labels == label]], axis=0)/len(np.array(cells)[labels == label])
        plt.plot(time_indices, sum_trace - (i+1)*30, label='summed trace',
                 color='silver')
        plt.savefig(fig_path + '/calcium_trace_cluster' + str(label) + '.svg')
    
    

    auto_array = np.zeros((ops['Ly'], ops['Lx']))
    auto_array[:] = np.nan
    for i, cell in enumerate(cells):
        ypix = stat[cell]['ypix']
        xpix = stat[cell]['xpix']
        try:
            auto_array[ypix, xpix] = labels[i]
            # auto_array[ops['Ly']-xpix, ops['Lx']-ypix] = labels[i]
                
        except IndexError:
            pass
        
    plt.figure(figsize=(15, 15))
    plt.imshow(ops['meanImg'], cmap='Greys')
    plt.imshow(auto_array, alpha=0.7, vmin=0, vmax=10, cmap=cmap_name)
    # plt.imshow(np.swapaxes(auto_array, 0,1), alpha=0.7, vmin=0, vmax=10, cmap=cmap_name)
    plt.title('Automatic clustering')
    plt.grid(b=None)
    plt.colorbar()
    plt.savefig(fig_path + '/clusters_pos.svg')

    return labels