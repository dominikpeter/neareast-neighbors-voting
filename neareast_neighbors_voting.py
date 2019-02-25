
import numpy as np

from sklearn.neighbors import NearestNeighbors, VALID_METRICS

from scipy.sparse import csr_matrix, lil_matrix
from sklearn.utils.validation import check_is_fitted


def softmax(x, axis=None):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis)


def normalize(x, axis=None):
    return x / np.sum(x, axis=axis)


class NearestNeighborsVoting(NearestNeighbors):
    """Encode categorical integer features as a one-hot numeric array.
    The input to this transformer should be an array-like of integers or

    Parameters
    ----------
    categories : 'auto' or a list of lists/arrays of values, default='auto'.
    sparse : boolean, default=True
        Will return sparse matrix if set True else will return an array.
    dtype : number type, default=np.float
        Desired dtype of output.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``).

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.

    """
    def __init__(self,
                 n_neighbors=5,
                 metrics_used_for_voting=["minkowski", "cosine", "euclidean"],
                 radius=1.0,
                 algorithm="auto",
                 **kwargs):
        self.metrics_used_for_voting = metrics_used_for_voting
        super(NearestNeighbors, self).__init__(n_neighbors=n_neighbors,
                                               radius=radius,
                                               algorithm=algorithm,
                                               leaf_size=30,
                                               **kwargs)

    def _get_kneighbors(self, X, n_neighbors, metric):
        try:
            check_is_fitted(self, ["_fit_method", "_fit_X"], all_or_any=any)
            self._fit(self._fit_X)
            dist, ind = self.kneighbors(X, return_distance=True)
            if dist.shape == (X.shape[0], n_neighbors):
                return dist, ind, metric
            else:
                raise ValueError
        except ValueError:
            print(f"Failed on {metric} metric")
            pass

    def _stack_kneighbors(self, X, n_neighbors, metrics):
        dist, ind, met = [], [], []
        for i in metrics:
            d, ix, m = self._get_kneighbors(X, n_neighbors, i)
            dist.append(d)
            ind.append(ix)
            met.append(m)
        indexes = np.stack(ind)
        distances = np.stack(dist)
        return indexes, distances, met

    def _get_voting(self, ind, dist, metrics=None, weights=None):
        ix = np.argmax(np.bincount(ind, weights))
        dist = np.nanmean(dist)
        if metrics:
            metrics = np.array(metrics)
            metric = metrics[ind == ix]
            return dist, ix, "|".join(metric)
        return dist, ix

    def _voting(self, indexes, distances, metrics):
        ds, rs, ks = indexes.shape[0], indexes.shape[1], indexes.shape[2]
        ind = np.zeros(shape=(rs, ks), dtype=np.int64)
        dist = np.zeros(shape=(rs, ks), dtype=np.float64)
        met = np.zeros(shape=(rs, ks), dtype=np.object)
        for k in range(ks):
            for r in range(rs):
                dist[r, k], ind[r, k], met[r, k] = self._get_voting(
                    indexes[:, r, k], distances[:, r, k], metrics)
        return dist, ind, met

    def kneighbors_voting(self, X,
                          n_neighbors=None, metrics_used_for_voting=None,
                          return_distance=True,
                          return_metrics=False):
        self._X_voting = X.copy()
        self.n_neighbors = n_neighbors if n_neighbors else self.n_neighbors
        if metrics_used_for_voting:
            self.metrics_used_for_voting = metrics_used_for_voting
        self.distances_, self.indexes_, self.metrics_ = self._voting(
            *self._stack_kneighbors(self._X_voting, self.n_neighbors,
                                    self.metrics_used_for_voting))
        if return_distance:
            if return_metrics:
                return self.distances_, self.indexes_, self.metrics_
            else:
                return self.distances_, self.indexes_
        if return_metrics:
            return self.distances_, self.metrics_
        return self.indexes_
