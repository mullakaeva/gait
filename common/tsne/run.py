from sklearn import manifold
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from matplotlib.patches import Ellipse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from .wrapper import Wrapper
from .tsne import TSNE

def calculate_gaussian_similarity(X, perplexity=30, metric='euclidean'):
    """ Compute pairiwse probabilities for MNIST pixels.
    """
    squared_eu_dist = pairwise_distances(X, metric=metric, squared=True)
    pij = manifold.t_sne._joint_probabilities(squared_eu_dist, perplexity, False)
    pij = squareform(pij)
    return pij

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def run_tsne(X, labels, vis_dir, save_prefix="" , target_dim = 2, perplexity = 30, batch_size = 4096, epochs = 1, iteration = 500):
    """
    Args:
        X: Numpy array with shape (num_samples, num_features). High-dimensional features that you want to cluster and visualise.
        labels: Numpy array with shape (num_samples,). Labels of classes.
    """
    
    num_classes = np.unique(labels).shape[0]
    colors_list = list("bgrcmyk") + ['0.75']
    

    pij_2d = calculate_gaussian_similarity(X, perplexity= perplexity)
    num_samples = X.shape[0]
    i, j = np.indices(pij_2d.shape)
    i = i.ravel()
    j = j.ravel()
    pij = pij_2d.ravel().astype('float32')
    # Remove self-indices
    idx = i != j
    i, j, pij = i[idx], j[idx], pij[idx]

    n_topics = target_dim
    n_dim = target_dim

    model = TSNE(num_samples, n_topics, n_dim)
    wrap = Wrapper(model, batchsize=batch_size, epochs=epochs)
    for itr in range(iteration):
        print("Now at iteration %d"%itr)
        wrap.fit(pij, i, j)
        embed = model.logits.weight.cpu().data.numpy()

        if itr % 1 == 0:
            fig, ax = plt.subplots(2,4, figsize=(14,7))
            ax = ax.ravel()
            for class_idx in range(num_classes):
                embed_this_class = embed[labels.astype(int) == class_idx,:]
                embed_other_classes = embed[labels.astype(int) != class_idx,:]
                ax[class_idx].scatter(embed_other_classes[:, 0], embed_other_classes[:, 1], c = "0.1", alpha=0.25)
                ax[class_idx].scatter(embed_this_class[:, 0], embed_this_class[:, 1], c = "r", alpha=0.75)
                ax[class_idx].set_title("Class {}".format(class_idx))
                ax[class_idx].axis("off")
            fig.suptitle("epoch: {}".format(itr))
            save_name = save_prefix + "_iter%05d.png"%(itr)
            fig_save_path = os.path.join(vis_dir, save_name)
            plt.savefig(fig_save_path, bbox_inches='tight', dpi=300)
            plt.close()
