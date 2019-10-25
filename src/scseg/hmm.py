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

from hmmlearn import _utils
from .base import _BaseHMM
from .utils import iter_from_X_lengths, _to_list, get_nsamples, get_batch
from hmmlearn.utils import normalize

__all__ = ["MultiModalMultinomialHMM"]


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
                          momentum=momentum)

    def _init(self, X, lengths=None):

        super(MultiModalMultinomialHMM, self)._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        X = _to_list(X)

        if 'e' in self.init_params:
            self.n_features = []
            self.emissionprob_ = []
            for modi in range(len(X)):
                _, n_features = X[modi].shape

                self.n_features.append(n_features)

                self.emissionprob_.append(self.random_state \
                    .rand(self.n_components, n_features))
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
                stats['obs'][i] += self.emission_prior
                stats['obs'][i] += x.T.tocsr().dot(posteriors).T

    def _do_mstep(self, stats):
        super(MultiModalMultinomialHMM, self)._do_mstep(stats)

        if 'e' in self.params:
            for i, suffstats in enumerate(stats['obs']):
                self.emissionprob_[i] = (suffstats / suffstats.sum(axis=1)[:, np.newaxis])

    def _do_mstep_minibatch(self, stats):
        super(MultiModalMultinomialHMM, self)._do_mstep_minibatch(stats)

        if 'e' in self.params:
            for i, suffstats in enumerate(stats['obs']):

                emissionprob_ = suffstats.copy()

                normalize(emissionprob_, axis=1)
                self.emissionprob_[i] = self.delta_weight(self.emissionprob_[i],
                    emissionprob_, 'em{}velocity'.format(i))

