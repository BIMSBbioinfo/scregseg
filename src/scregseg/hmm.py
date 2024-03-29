# Hidden Markov Models
#
# Author: Ron Weiss <ronweiss@gmail.com>
#         Shiqiao Du <lucidfrontier.45@gmail.com>
# API changes: Jaques Grobler <jaquesgrobler@gmail.com>
# Modifications to create of the HMMLearn module: Gael Varoquaux
# More API changes: Sergei Lebedev <superbobry@gmail.com>
# Modified to support lists of datasets: Wolfgang Kopp <wolfgang.kopp@mdc-berlin.de>
#
#
# This module implements the Dirichlet-Multinomial HMM based on the hmmlearn python package.


"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""
import logging
from multiprocessing import Pool
from datetime import datetime
import numpy as np
from scipy.special import logsumexp
from scipy.special import digamma
from scipy.special import gammaln
from scipy.sparse import issparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import hstack
from scipy.stats import nbinom
from sklearn import cluster
from sklearn.utils import check_random_state
import pandas as pd
import os
import time
import json

from hmmlearn import _utils
from .base import _BaseHMM, MinibatchMonitor
from .utils import iter_from_X_lengths, _to_list, get_nsamples, get_batch, _check_array
from .utils import fit_mixture
from hmmlearn.utils import normalize, log_normalize
from ._utils import _fast_dirmul_loglikeli_sp

__all__ = ["DirMulHMM"]


def dirmul_loglikeli_naive(x, alpha):
    """ Dirichlet-Multinomial loglikelihood

    Slow implementation. Only for testing.
    """
    alpha0 = alpha.sum(1)[None,:] # state x cell
    n = np.asarray(x.sum(1)) # region x cell
    res = gammaln(alpha0) - gammaln(n+alpha0)
    for i in range(res.shape[0]): # sum over datapoints / regions
        for j in range(alpha.shape[0]): # sum over states
            res[i,j] = 0
            res[i,j] += gammaln(alpha[j].sum())
            res[i,j] += gammaln(x[i].toarray() + alpha[j]).sum()
            res[i,j] -= gammaln(alpha[j]).sum()
            res[i,j] -= gammaln(x[i].sum(1)+alpha[j].sum())
    return res

def dirmul_loglikeli(x, alpha, maxcounts=3):
    alpha0 = alpha.sum(1)[None,:] # state x cell
    n = np.asarray(x.sum(1)) # region x cell
    res = np.zeros((x.shape[0], alpha.shape[0]))
    res += gammaln(alpha0)
    res -= gammaln(n+alpha0) # region x 1
#    res -= gammaln(alpha).sum(1, keepdims=True).T # 1 x state
    precomp = np.zeros((maxcounts, ) + alpha.shape)
    precomp[1:] = gammaln(alpha[None,:,:] + np.arange(1,maxcounts)[:,None,None]) - gammaln(alpha)[None,:,:]
    for i, j in iter_from_X_lengths(x, 10000): # sum over regions in batches of 10000
        xbatch = x[i:j].toarray().astype(np.int64)
        res[i:j] += precomp[xbatch, :, np.arange(xbatch.shape[-1])].sum(-2)
    return res


def dirmul_loglikeli_sp(x, alpha):
    """
    x : np.array
      regions x cell count matrix
    alpha : np.array
      state x cell parameter matrix
    """
    alpha0 = alpha.sum(1)[None,:] # state x 1
    n = np.asarray(x.sum(1)) # region x 1
    res = gammaln(alpha0) - gammaln(n + alpha0)
    # x = 1, 2, 3 .. number of counts
    # n states
    # n cells
    maxcounts = x.max() + 1
    precomp = gammaln(alpha[None,:,:] + np.arange(1, maxcounts)[:,None, None]) - gammaln(alpha)[None,:,:]
    for idx in range(x.shape[0]):
        if issparse(x):
            ids = x.indices[x.indptr[idx]:x.indptr[idx+1]]
            #cnts = np.where(x.data[x.indptr[idx]:x.indptr[idx+1]].astype(np.int64) >= maxcounts, maxcounts-1, x.data[x.indptr[idx]:x.indptr[idx+1]].astype(np.int64))
            cnts = x.data[x.indptr[idx]:x.indptr[idx+1]].astype(np.int64)
        else:
            ids = np.nonzero(x[idx, :])[0]
            cnts = x[idx, ids]
            #cnts = (cnts >= maxcounts, maxcounts - 1, cnts)
        res[idx] += precomp[cnts-1, :, ids].sum(0)
    return res


def dirmul_loglikeli_sp_mincov(x, alpha, maxcounts=3, mincov = 100):
    """
    x : np.array
      regions x cell count matrix
    alpha : np.array
      state x cell parameter matrix
    """
    alpha0 = alpha.sum(1)[None,:] # state x 1
    n = np.asarray(x.sum(1)) # region x 1
    res = gammaln(alpha0) - gammaln(n + alpha0)
    # x = 1, 2, 3 .. number of counts
    # n states
    # n cells
    precomp = gammaln(alpha[None,:,:] + np.arange(1, maxcounts)[:,None, None]) - gammaln(alpha)[None,:,:]
    for idx in range(x.shape[0]):
        if n[idx, 0] >= mincov:
            ids = x.indices[x.indptr[idx]:x.indptr[idx+1]]
            cnts = np.where(x.data[x.indptr[idx]:x.indptr[idx+1]].astype(np.int64) >= maxcounts, maxcounts-1, x.data[x.indptr[idx]:x.indptr[idx+1]].astype(np.int64))
            res[idx] += precomp[cnts-1, :, ids].sum(0)
        else:
            res[idx] = 0.
    return res

def fast_dirmul_loglikeli_sp(x, alpha):
    result = np.zeros((x.shape[0], alpha.shape[0]))
    if type(x) != csr_matrix:
        try:
            x = csr_matrix(x)
        except TypeError:
            print('x cannot be converted to a scipy.sparse.csr.csr_matrix which is the required input of '
                  '_fast_dirmul_loglikeli_sp.')
            raise
    _fast_dirmul_loglikeli_sp(x.indices, x.indptr, x.data.astype('int'),
                             alpha, x.shape[0], result)
    return result

def get_region_cnts(dat):
    return pd.Series(np.asarray(dat.sum(1)).flatten())


def get_breaks(cnts, qstepsize=.05):
    qs = np.linspace(qstepsize, 1.-qstepsize, int(1./qstepsize)-1)
    cth = []
    for q in qs:
        cth.append(cnts.quantile(q))
    return np.asarray(cth)

def cnts2bins(cnts, br):
    return np.sum(cnts.values[:,None] > br[None,:], axis=1)

def init_cnt_probs(components, nbins):
    return np.ones((components, nbins))/nbins

def cntbin_loglikelihood(cntbins, probs):
    #x 14 is out of bounds fo
    #Z = csc_matrix((np.ones(len(cntbins)),
    #               (np.arange(len(cntbins), dtype='int'),
    #                cntbins.astype('int')
    #               )))
    #Z.dot(np.log(probs).T)
    return np.log(probs[:,cntbins]).T

def cntbin_suffstats(cntbins, posterior):
    Z = csc_matrix((np.ones(len(cntbins)),
                   (
                    cntbins.astype('int'),
                    np.arange(len(cntbins), dtype='int'),
                   )))
    return Z.dot(posterior).T


class DirMulHMM(_BaseHMM):
    """Hidden Markov Model with dirichlet-ultinomial (discrete) emissions
    Parameters
    ----------
    n_components : int
        Number of states.
    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.
    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.
    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".
    random_state: RandomState or an int seed, optional
        A random number generator instance.
    n_iter : int, optional
        Maximum number of iterations to perform.
    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.
    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.
    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'e' for emissionprob.
        Defaults to all parameters.
    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'e' for emissionprob.
        Defaults to all parameters.
    Attributes
    ----------
    n_features : int
        Number of possible symbols emitted by the model (in the samples).
    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.
    emissionprob\_ : array, shape (n_components, n_features)
        Probability of emitting a given symbol when in each state.
    Examples
    --------
    >>> from scregseg.hmm import DirMulHMM
    >>> DirMulHMM(n_components=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DirMulHMM(n_components=2)
    """
    # TODO: accept the prior on emissionprob_ for consistency.
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 emission_prior=1,
                 algorithm="map", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste",
                 n_jobs=1,
                 ):

        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params,
                          n_jobs=n_jobs, emission_prior=emission_prior,
                          )

    def _trim_array(self, X):
        return X

    def _init(self, X, lengths=None):

        super(DirMulHMM, self)._init(X, lengths=lengths)
        self.random_state_ = check_random_state(self.random_state)

        X = _to_list(X)

        if 'e' in self.init_params:
            self.emission_suffstats_ = []
            self.emission_prior_ = []

            z = coo_matrix(self.random_state_.multinomial(1,
                           np.ones(self.n_components)/self.n_components, X[0].shape[0]))
            for modi in range(len(X)):
                # random init but with read depth offset
                _, n_features = X[modi].shape

                # prior
                x = np.array(X[modi].sum(0)) + 1.
                normalize(x)
                x *= self.emission_prior

                self.emission_prior_.append(x)

                r = z.T.dot(X[modi]).toarray()

                self.emission_suffstats_.append(r)
        self.n_features = [x.shape[1] for x in X]

    def _check(self):
        super(DirMulHMM, self)._check()

        for nfeat, ep, es in zip(self.n_features, self.emission_prior_, self.emission_suffstats_):
            ep = np.atleast_2d(ep)
            if ep.shape != (1, nfeat):
                raise ValueError(
                    "emission_prior_ must have shape (n_components, n_features)")
            es = np.atleast_2d(es)
            if es.shape != (self.n_components, nfeat):
                raise ValueError(
                    "emission_suffstats_ must have shape (n_components, n_features) got {}".format(es.shape))

    @classmethod
    def load(cls, path):
        npzfile = np.load(os.path.join(path, 'modelparams', 'dirmulhmm.npz'))

        trans = npzfile['arr_0']
        start = npzfile['arr_1']
        es = npzfile['arr_2']
        ep = npzfile['arr_3']
        #emissions = [npzfile[file] for file in npzfile.files[2:]]
        emissions = [npzfile[file] for file in npzfile.files[2:]]

        en = len(emissions)//2
        es = emissions[:en]
        ep = emissions[en:]

        parfile = os.path.join(path, 'modelparams', 'dirmulhmm.json')
        if os.path.exists(parfile):
            with open(parfile, 'r') as f:
                params = json.load(f)
            model = cls(**params)
        else:
            model = cls(len(start))

        #model = cls(len(start))

        model.transmat_ = trans
        model.startprob_ = start
        model.emission_suffstats_ = es
        model.emission_prior_ = ep

        model.n_features = [e.shape[1] for e in es]
        return model

    def print_progress(self):
        if not self.verbose:
            return

    def save(self, path):
        """
        saves current model parameters
        """

        os.makedirs(path, exist_ok=True)
        #path = os.path.join(path, 'dirmulhmm.npz')

        savelist = [self.transmat_, self.startprob_] + self.emission_suffstats_ + self.emission_prior_
        np.savez(os.path.join(path, 'dirmulhmm.npz'), *savelist)

        # save hyper parameters
        with open(os.path.join(path, 'dirmulhmm.json'), 'w') as f:
            json.dump(self.get_params(), f)


    def _compute_log_likelihood_one(self, X, i):

        # loop over datasets each represented via a multinomial
        #for i, (ep, es, x) in enumerate(zip(self.emission_prior_, self.emission_suffstats_, X)):
            # compute the marginal likelihood with the current posterior parameters
        return fast_dirmul_loglikeli_sp(X[i], self.emission_prior_[i] + self.emission_suffstats_[i])

    def _compute_log_likelihood(self, X):
        res = np.zeros((len(X), get_nsamples(X), self.n_components))
        # loop over datasets each represented via a multinomial
        for i in range(len(X)):
            res[i] += self._compute_log_likelihood_one(X, i)

        # add independent replicates
        res = res.sum(0)

        return res

    def _generate_sample_from_state(self, state, random_state=None):
        # this is just a dummy return
        # as we are only interested in
        # the sequence of states.
        return np.asarray([[0.]])

    def _initialize_sufficient_statistics(self):
        stats = super(DirMulHMM, self)._initialize_sufficient_statistics()
        stats['obs'] = [np.zeros((self.n_components, feat)) for feat in self.n_features]
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(DirMulHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'e' in self.params:
            for i, x in enumerate(X):
                stats['obs'][i] += x.T.dot(posteriors).T

    def _do_mstep(self, stats):
        super(DirMulHMM, self)._do_mstep(stats)

        if 'e' in self.params:
            for i, suffstats in enumerate(stats['obs']):
                self.emission_suffstats_[i] = suffstats

