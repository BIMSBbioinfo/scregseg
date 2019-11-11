# Hidden Markov Models
#
# Author: Ron Weiss <ronweiss@gmail.com>
#         Shiqiao Du <lucidfrontier.45@gmail.com>
# API changes: Jaques Grobler <jaquesgrobler@gmail.com>
# Modifications to create of the HMMLearn module: Gael Varoquaux
# More API changes: Sergei Lebedev <superbobry@gmail.com>
# Modified to support lists of datasets: Wolfgang Kopp <wolfgang.kopp@

"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import numpy as np
from scipy.special import logsumexp
from scipy.special import digamma
from scipy.special import gammaln
from scipy.sparse import issparse
from scipy.stats import nbinom
from sklearn import cluster
from sklearn.utils import check_random_state
import os

from hmmlearn import _utils
from .base import _BaseHMM
from .utils import iter_from_X_lengths, _to_list, get_nsamples, get_batch
from hmmlearn.utils import normalize

__all__ = ["MultiModalMultinomialHMM", "MultiModalMixHMM", "MultiModalDirMulHMM"]


def dirmul_loglikeli_naive(x, alpha):
    x = x.tocsr()
    x.data[x.data>=maxcounts] = maxcounts - 1
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
    x = x.tocsr().copy()
    x.data[x.data>=maxcounts] = maxcounts - 1
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


def dirmul_loglikeli_sp(x, alpha, maxcounts=3):
    """
    x : np.array
      regions x cell count matrix
    alpha : np.array
      state x cell parameter matrix
    """
    x = x.tocsr().copy()
    x.data[x.data>=maxcounts] = maxcounts - 1
    alpha0 = alpha.sum(1)[None,:] # state x 1
    n = np.asarray(x.sum(1)) # region x 1
    res = gammaln(alpha0) - gammaln(n + alpha0)
    # x = 1, 2, 3 .. number of counts
    # n states
    # n cells
    precomp = gammaln(alpha[None,:,:] + np.arange(1, maxcounts)[:,None, None]) - gammaln(alpha)[None,:,:]
    for idx in range(x.shape[0]):
        ids = x.indices[x.indptr[idx]:x.indptr[idx+1]]
        cnts = x.data[x.indptr[idx]:x.indptr[idx+1]].astype(np.int64)
        res[idx] += precomp[cnts-1, :, ids].sum(0)
    return res


def test():
    #10 regions, 3 cells, 4 states
    x = csr_matrix(np.random.choice(4,(1000,300)))
    alpha = np.random.rand(4,300)
    dirmul_loglikeli_naive(x,alpha)
    dirmul_loglikeli(x,alpha)
    dirmul_loglikeli_sp(x,alpha)
    # the batched version


class MultiModalDirMulHMM(_BaseHMM):
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
    with_nbinom : boolean
        Whether to model the cross-cell type profile counts as negative binomial
        in addition to the multinomial.
    batch_size : int, optional
        Mini-batch size. Default: 10000.
    minibatchlearning : bool, optional
        Whether to perform mini-batch learning. Default: False.
    learningrate : float, optional
        Learning rate for mini-batch learning. Default: 0.05
    momentum : float, optional
        Momentum term for mini-batch learning. Default: 0.85
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
    >>> from hmmlearn.hmm import MultiModalMultinomialHMM
    >>> MultiModalMultinomialHMM(n_components=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    MultiModalMultinomialHMM(algorithm='viterbi',...
    """
    # TODO: accept the prior on emissionprob_ for consistency.
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 emission_prior=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste",
                 batch_size=10000,
                 minibatchlearning=False,
                 learningrate=0.05, momentum=0.85,
                 n_jobs=1, 
                 decay=0.1, schedule_steps=10):
        self.prior_obs = emission_prior
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params,
                          batch_size=batch_size,
                          minibatchlearning=minibatchlearning,
                          learningrate=learningrate,
                          n_jobs=n_jobs,
                          momentum=momentum)

    def _init(self, X, lengths=None):

        super(MultiModalDirMulHMM, self)._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        X = _to_list(X)

        if 'e' in self.init_params:
            self.n_features = []
            self.emission_suffstats_ = []
            self.emission_prior_ = []

            for modi in range(len(X)):
                # random init but with read depth offset
                _, n_features = X[modi].shape

                self.n_features.append(n_features)

                # prior
                x = np.array(X[modi].sum(0))
                normalize(x)
                x *= self.prior_obs

                self.emission_prior_.append(x)

                r = self.random_state.rand(self.n_components, n_features)

                normalize(r)
                #x = .9*x + .1*r
                self.emission_suffstats_.append(r)
                    #.rand(self.n_components, n_features))
                #normalize(self.emissionprob_[-1], axis=1)

    def _check(self):
        super(MultiModalDirMulHMM, self)._check()

        for nfeat, ep, es in zip(self.n_features, self.emission_prior_, self.emission_suffstats_):
            ep = np.atleast_2d(ep)
            if ep.shape != (1, nfeat):
                raise ValueError(
                    "emission_prior_ must have shape (n_components, n_features)")
            es = np.atleast_2d(es)
            if es.shape != (self.n_components, nfeat):
                raise ValueError(
                    "emission_suffstats_ must have shape (n_components, n_features)")

    @classmethod
    def load(cls, path):
        npzfile = np.load(os.path.join(path, 'modelparams', 'dirmulhmm.npz'))

        trans = npzfile['arr_0']
        start = npzfile['arr_1']
        emissions = [npzfile[file] for file in npzfile.files[2:]]

        elen = len(emissions) // 2
        es = emissions[:elen]
        ep = emissions[elen:]

        model = cls(len(start))

        model.transmat_ = trans
        model.startprob_ = start
        model.emission_suffstats_ = es
        model.emission_prior_ = ep

        model.n_features = [e.shape[1] for e in es]
        return model

    def save(self, path):
        """
        saves current model parameters
        """
        path = os.path.join(path, 'dirmulhmm.npz')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        savelist = [self.transmat_, self.startprob_] + self.emission_suffstats_ + self.emission_prior_
        np.savez(path, *savelist)

    def _compute_log_likelihood(self, X):
        res = np.zeros((get_nsamples(X), self.n_components))
        # loop over datasets each represented via a multinomial
        for ep, es, x in zip(self.emission_prior_, self.emission_suffstats_, X):
            # compute the marginal likelihood with the current posterior parameters
            res += dirmul_loglikeli_sp(x, ep+es)

        return res

    def _generate_sample_from_state(self, state, random_state=None):
        # this is just a dummy return
        # as we are only interested in
        # the sequence of states.
        return np.asarray([[0.]])

    def _initialize_sufficient_statistics(self):
        stats = super(MultiModalDirMulHMM, self)._initialize_sufficient_statistics()
        stats['obs'] = [np.zeros((self.n_components, feat)) for feat in self.n_features]
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(MultiModalDirMulHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'e' in self.params:
            for i, x in enumerate(X):
                stats['obs'][i] += x.T.tocsr().dot(posteriors).T

    def _do_mstep(self, stats):
        super(MultiModalDirMulHMM, self)._do_mstep(stats)

        if 'e' in self.params:
            for i, suffstats in enumerate(stats['obs']):
                self.emission_suffstats_[i] = suffstats


class MultiModalMultinomialHMM(_BaseHMM):
    """Hidden Markov Model with multinomial (discrete) emissions
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
    with_nbinom : boolean
        Whether to model the cross-cell type profile counts as negative binomial
        in addition to the multinomial.
    batch_size : int, optional
        Mini-batch size. Default: 10000.
    minibatchlearning : bool, optional
        Whether to perform mini-batch learning. Default: False.
    learningrate : float, optional
        Learning rate for mini-batch learning. Default: 0.05
    momentum : float, optional
        Momentum term for mini-batch learning. Default: 0.85
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
    >>> from hmmlearn.hmm import MultiModalMultinomialHMM
    >>> MultiModalMultinomialHMM(n_components=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    MultiModalMultinomialHMM(algorithm='viterbi',...
    """
    # TODO: accept the prior on emissionprob_ for consistency.
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 emission_prior=1e-5,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste",
                 batch_size=10000,
                 minibatchlearning=False,
                 learningrate=0.05, momentum=0.85,
                 n_jobs=1, 
                 decay=0.1, schedule_steps=10):
        self.emission_prior=emission_prior
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params,
                          batch_size=batch_size,
                          minibatchlearning=minibatchlearning,
                          learningrate=learningrate,
                          n_jobs=n_jobs,
                          momentum=momentum)

    def _init(self, X, lengths=None):

        super(MultiModalMultinomialHMM, self)._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        X = _to_list(X)

        if 'e' in self.init_params:
            self.n_features = []
            self.emissionprob_ = []
            for modi in range(len(X)):
                # random init but with read depth offset
                _, n_features = X[modi].shape

                self.n_features.append(n_features)

#                self.emissionprob_.append(self.random_state \
#                    .rand(self.n_components, n_features))

                x = np.array(X[modi].sum(0))
                normalize(x)
                r = self.random_state.rand(self.n_components, n_features)

                normalize(r)
                x = .9*x + .1*r
                self.emissionprob_.append(x)
                    #.rand(self.n_components, n_features))
                normalize(self.emissionprob_[-1], axis=1)

    def _check(self):
        super(MultiModalMultinomialHMM, self)._check()

        for nfeat, emission in zip(self.n_features, self.emissionprob_):
            emission = np.atleast_2d(emission)
            if emission.shape != (self.n_components, nfeat):
                raise ValueError(
                    "emissionprob_ must have shape (n_components, n_features)")
            if not np.allclose(emission.sum(axis=1), 1.0):
                raise ValueError("emissionprobs must sum to 1.0 (got {})"
                                         .format(emission.sum(axis=1)))
            assert np.all(emission > 0.0), "emissionprobs must be positive"

    @classmethod
    def load(cls, path):
        npzfile = np.load(os.path.join(path, 'modelparams', 'hmm.npz'))

        trans = npzfile['arr_0']
        start = npzfile['arr_1']
        emissions = [npzfile[file] for file in npzfile.files[2:]]

        model = cls(len(start))

        model.transmat_ = trans
        model.startprob_ = start
        model.emissionprob_ = emissions
        model.n_features = [e.shape[1] for e in emissions]
        return model

    def save(self, path):
        """
        saves current model parameters
        """
        path = os.path.join(path, 'hmm.npz')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        #mpath = os.path.join(path, 'modelparams', 'hmm.npz')
        savelist = [self.transmat_, self.startprob_] + self.emissionprob_
        np.savez(path, *savelist)

    def _compute_log_likelihood(self, X):
        res = np.zeros((get_nsamples(X), self.n_components))
        # loop over datasets each represented via a multinomial
        for e, x in zip(self.emissionprob_, X):
            res += x.dot(np.log(e.T))

        return res

    def _generate_sample_from_state(self, state, random_state=None):
        # this is just a dummy return
        # as we are only interested in
        # the sequence of states.
        return np.asarray([[0.]])

    def _initialize_sufficient_statistics(self):
        stats = super(MultiModalMultinomialHMM, self)._initialize_sufficient_statistics()
        stats['obs'] = [np.zeros((self.n_components, feat)) for feat in self.n_features]
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(MultiModalMultinomialHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'e' in self.params:
            for i, x in enumerate(X):
                stats['obs'][i] += x.T.tocsr().dot(posteriors).T

    def _do_mstep(self, stats):
        super(MultiModalMultinomialHMM, self)._do_mstep(stats)

        if 'e' in self.params:
            for i, suffstats in enumerate(stats['obs']):
                suffstats += self.emission_prior
                self.emissionprob_[i] = (suffstats / suffstats.sum(axis=1)[:, np.newaxis])

    def _do_mstep_minibatch(self, stats):
        super(MultiModalMultinomialHMM, self)._do_mstep_minibatch(stats)

        if 'e' in self.params:
            for i, suffstats in enumerate(stats['obs']):

                emissionprob_ = suffstats.copy()

                normalize(emissionprob_, axis=1)
                self.emissionprob_[i] = self.delta_weight(self.emissionprob_[i],
                    emissionprob_, 'em{}velocity'.format(i))


class MultiModalMixHMM(_BaseHMM):
    """Hidden Markov Model with multinomial (discrete) emissions
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
    with_nbinom : boolean
        Whether to model the cross-cell type profile counts as negative binomial
        in addition to the multinomial.
    batch_size : int, optional
        Mini-batch size. Default: 10000.
    minibatchlearning : bool, optional
        Whether to perform mini-batch learning. Default: False.
    learningrate : float, optional
        Learning rate for mini-batch learning. Default: 0.05
    momentum : float, optional
        Momentum term for mini-batch learning. Default: 0.85
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
    >>> from hmmlearn.hmm import MultiModalMixHMM
    >>> MultiModalMixHMM(n_components=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    MultiModalMixHMM(algorithm='viterbi',...
    """
    # TODO: accept the prior on emissionprob_ for consistency.
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 emission_prior=1e-5,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste",
                 batch_size=10000,
                 minibatchlearning=False,
                 learningrate=0.05, momentum=0.85,
                 n_jobs=1, 
                 decay=0.1, schedule_steps=10):
        self.emission_prior=emission_prior
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params,
                          batch_size=batch_size,
                          minibatchlearning=minibatchlearning,
                          learningrate=learningrate,
                          n_jobs=n_jobs,
                          momentum=momentum)

    def _init(self, X, lengths=None):
        super(MultiModalMixHMM, self)._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        X = _to_list(X)

        if 'e' in self.init_params:
            self.alpha_ = np.ones(self.n_components)*.1
            self.n_features = []
            self.emissionprob_ = []
            self.emissionbackground_ = []
            for modi in range(len(X)):
                _, n_features = X[modi].shape

                self.n_features.append(n_features)

                x = np.array(X[modi].sum(0))
                normalize(x)
                self.emissionbackground_.append(x)
                
                r = self.random_state.rand(self.n_components, n_features)

                normalize(r)
                x = .9*x + .1*r
                self.emissionprob_.append(x)
                normalize(self.emissionprob_[-1], axis=1)
               # self.emissionprob_.append(self.random_state \
               #     .rand(self.n_components, n_features))
               # normalize(self.emissionprob_[-1], axis=1)
                #self.emissionbackground_.append(np.array(X[modi].sum(0)))
                #normalize(self.emissionbackground_[-1])

    def _check(self):
        super(MultiModalMixHMM, self)._check()

        assert np.all(self.alpha_ > 0.0), 'Alphas must be positiv'
        assert np.all(self.alpha_ <= 1.0), 'Alphas must be smaller than one'
        for nfeat, emission in zip(self.n_features, self.emissionprob_):
            emission = np.atleast_2d(emission)
            if emission.shape != (self.n_components, nfeat):
                raise ValueError(
                    "emissionprob_ must have shape (n_components, n_features)")
            if not np.allclose(emission.sum(axis=1), 1.0):
                raise ValueError("emissionprobs must sum to 1.0 (got {})"
                                         .format(emission.sum(axis=1)))
            assert np.all(emission > 0.0), "emissionprobs must be positive"

    @classmethod
    def load(cls, path):
        npzfile = np.load(os.path.join(path, 'modelparams', 'mixhmm.npz'))

        trans = npzfile['arr_0']
        start = npzfile['arr_1']
        alpha = npzfile['arr_2']
        emissions = [npzfile[file] for file in npzfile.files[3:]]
        elen = len(emissions) // 2
        em = emissions[:elen]
        eb = emissions[elen:]

        model = cls(len(start))
        model.transmat_ = trans
        model.startprob_ = start
        model.alpha_ = alpha
        model.emissionprob_ = em
        model.emissionbackground_ = eb
        model.n_features = [e.shape[1] for e in em]

    def save(self, path):
        """
        saves current model parameters
        """
        path = os.path.join(path, 'mixhmm.npz')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        #mpath = os.path.join(path, 'modelparams', 'hmm.npz')
        savelist = [self.transmat_, self.startprob_, self.alpha_] + self.emissionprob_ + \
                   self.emissionbackground_
        np.savez(path, *savelist)

    def _compute_complete_log_likelihood(self, X):
        res = np.zeros((2, get_nsamples(X), self.n_components))
        res[0, :, :] = np.log(1.-self.alpha_)[None,:]
        res[1, :, :] = np.log(self.alpha_)[None,:]
        # loop over datasets each represented via a multinomial
        for b, e, x in zip(self.emissionbackground_, self.emissionprob_, X):
            res[0] += x.dot(np.log(b.T))
            res[1] += x.dot(np.log(e.T))

        return res

    def _compute_log_likelihood(self, X):
        res = self._compute_complete_log_likelihood(X)
        res = logsumexp(res, axis=0)
        return res

    def _generate_sample_from_state(self, state, random_state=None):
        # this is just a dummy return
        # as we are only interested in
        # the sequence of states.
        return np.asarray([[0.]])

    def _initialize_sufficient_statistics(self):
        stats = super(MultiModalMixHMM, self)._initialize_sufficient_statistics()
        stats['obs'] = [np.zeros((self.n_components, feat)) for feat in self.n_features]
        stats['alphas'] = np.zeros(self.n_components)
        stats['norm'] = np.zeros(self.n_components)

        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(MultiModalMixHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)

        res = self._compute_complete_log_likelihood(X)
        res -= logsumexp(res, axis=0, keepdims=True)
        np.testing.assert_allclose(np.exp(res).sum(0), np.ones_like(res[0]))
        stats['alphas'] = (np.exp(res[1])*posteriors).sum(axis=0)
        stats['norm'] = posteriors.sum(axis=0)

        if 'e' in self.params:
            for i, x in enumerate(X):
                stats['obs'][i] += x.T.tocsr().dot(posteriors*np.exp(res[1])).T

    def _do_mstep(self, stats):
        super(MultiModalMixHMM, self)._do_mstep(stats)

        self.alpha_ = stats['alphas'] / stats['norm']
        print('alpha', self.alpha_)
        if stats['trim-alpha']:
            self.alpha_ = np.where(self.alpha_ >.99, self.alpha_, .99) 
        #    self.alpha_ = np.ones_like(self.alpha_)
            print('but trimmed', self.alpha_)

        if 'e' in self.params:
            
            #np.testing.assert_allclose(stats['alphas'], np.ones_like(res[0]))
            #assert all(stats['alphas'] < 1.), stats['alphas']
            #assert all(stats['alphas'] > 0.), stats['alphas']
            #print(self.alpha_)
            for i, suffstats in enumerate(stats['obs']):
                suffstats += self.emission_prior
                self.emissionprob_[i] = (suffstats / suffstats.sum(axis=1)[:, np.newaxis])
        self._check()

    def _do_mstep_minibatch(self, stats):
        super(MultiModalMixHMM, self)._do_mstep_minibatch(stats)

        if 'e' in self.params:
            for i, suffstats in enumerate(stats['obs']):

                emissionprob_ = suffstats.copy()

                normalize(emissionprob_, axis=1)
                self.emissionprob_[i] = self.delta_weight(self.emissionprob_[i],
                    emissionprob_, 'em{}velocity'.format(i))

