# These functions were introduced or adapted from the hmmlearn python package
# in order to enable sparse matrix support.
""" Helper Functions
"""
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.utils import check_array
from sklearn.utils import check_random_state


def _check_array(X, accept_sparse=True, aslist=False):
    if aslist and not isinstance(X, list):
        X = [X]
    if isinstance(X, list):
        return [check_array(X_, accept_sparse=accept_sparse) for X_ in X]
    else:
        return check_array(X, accept_sparse=accept_sparse)

def get_nsamples(X):
    if isinstance(X, list):
        x = X[0]
    else:
        x = X
    return x.shape[0]


def get_batch(X, i, j):

    if isinstance(X, list):
        return [x[i:j] for x in X]
    return X[i:j]


def _to_list(objs):
    if not isinstance(objs, list):
        return [objs]
    return objs


def iter_from_X_lengths(X, lengths, state=None):
    if isinstance(X, list):
        x = X[0]
    else:
        x = X
    if lengths is None:
        yield 0, x.shape[0]
    elif isinstance(lengths, int):
        # this chunk is for mini-batch learning
        if state is None:
            starts = np.arange(0, x.shape[0], lengths)
        else:
            starts = state.permutation(np.arange(0, x.shape[0], lengths))
        for i, start in enumerate(starts):
            yield start, min(start + lengths, x.shape[0])
    elif isinstance(lengths, list):
        n_samples = x.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {:d} samples in lengths array {!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]

def fragmentlength_by_state(model, fmat):
    df =  pd.DataFrame(fmat.cmat.toarray(), columns=fmat.cannot.cell)

    df['name'] = model._segments.name
    adf = df.groupby('name').aggregate('sum')
    adf = adf.div(adf.sum(axis=1), axis=0).rename({'cell': 'Fragment size'})

    return adf
