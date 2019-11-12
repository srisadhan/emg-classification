from sklearn.manifold import TSNE
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D
import sys

def project_data_on_manifold(data, algorithm):
    """Project the data on a manifold using .

    Parameters
    ----------
    data : nd-array
        an array of features stacked across columns.
    algorithm : String
        either t-SNE or UMAP
    Returns
    -------
    data_embedded : nd-array
        embedding of the data in a lower dimension

    """


    if algorithm.upper() == 'TSNE':
        fit = TSNE(n_components=2, perplexity=100, learning_rate=50.0)
    elif algorithm.upper() == 'UMAP':
        # fit = UMAP(n_neighbors=neighbor, min_dist=0.0, n_components=3,metric='chebyshev')
        # X_embedded = fit.fit_transform(X)
        X_embedded = UMAP(n_neighbors=neighbor, min_dist=0.0, n_components=3,metric='chebyshev')
    else:
        print('No method provided to reduce the dimensionality')
        sys.exit()

    X_embedded = fit.fit_tranform(X)

    return X_embedded
