# This file was adopted from the original hmmlearn implementation.
# We have modified the original implemention primarily to enable parallel
# processing for the training and inference procedures.

from __future__ import print_function

import string
import sys
from collections import deque
import logging

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from hmmlearn import _hmmc
from hmmlearn.base import ConvergenceMonitor
from hmmlearn.utils import normalize, log_normalize
from .utils import iter_from_X_lengths
from .utils import _check_array, get_nsamples, get_batch
from sklearn.utils._joblib import Parallel, delayed, effective_n_jobs

#: Supported decoder algorithms.
DECODER_ALGORITHMS = frozenset(("viterbi", "map", "robust_map"))

EPS = 1e-6

def batch_compute_loglikeli(self, X):
    framelogprob = self._compute_log_likelihood(X)
    logprob, fwdlattice = self._do_forward_pass(framelogprob)
    return framelogprob, fwdlattice, logprob

def batch_compute_posterior_robust(self, X):
    posteriors = np.zeros((len(X), get_nsamples(X), self.n_components))
    for i in range(len(X)):
        framelogprob, fwdlattice, logprob = batch_compute_loglikeli(self, [X[i]])
        bwdlattice = self._do_backward_pass(framelogprob)
        posteriors[i] = self._compute_posteriors(fwdlattice, bwdlattice)
    posteriors_mean = posteriors.mean(0)
    posteriors_sd = posteriors.std(0)
    return posteriors_mean, posteriors_sd

def batch_compute_posterior(self, X):
    framelogprob, fwdlattice, logprob = batch_compute_loglikeli(self, X)
    bwdlattice = self._do_backward_pass(framelogprob)
    posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
    return framelogprob, posteriors, fwdlattice, bwdlattice, logprob

def batch_accumulate_suff_state(self, X):
    stats = self._initialize_sufficient_statistics()
    framelogprob, posteriors, fwdlattice, bwdlattice, logprob = batch_compute_posterior(self, X)
    self._accumulate_sufficient_statistics(
        stats, X, framelogprob, posteriors, fwdlattice,
        bwdlattice)
    return framelogprob, posteriors, fwdlattice, bwdlattice, logprob, stats

class MinibatchMonitor(ConvergenceMonitor):
    @property
    def converged(self):
        return (self.iter == self.n_iter)

class _BaseHMM(BaseEstimator):
    r"""Base class for Hidden Markov Models.

    This class allows for easy evaluation of, sampling from, and
    maximum a posteriori estimation of the parameters of a HMM.

    See the instance documentation for details specific to a
    particular object.

    Parameters
    ----------
    n_components : int
        Number of states in the model.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi", "map" or "robust_map".
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
        't' for transmat, and other characters for subclass-specific
        emission parameters. Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, and other characters for
        subclass-specific emission parameters. Defaults to all
        parameters.

    Attributes
    ----------
    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    """
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters,
                 n_jobs=1, emission_prior=1,
                 replicate='sum'
                 ):
        self.n_components = n_components
        self.params = params
        self.init_params = init_params
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.monitor_ = MinibatchMonitor(self.tol, self.n_iter, self.verbose)
        self.check_fitted = "transmat_"
        self.emission_prior = emission_prior
        self.replicate = replicate

    def get_stationary_distribution(self):
        """Compute the stationary distribution of states.
        """
        # The stationary distribution is proportional to the left-eigenvector
        # associated with the largest eigenvalue (i.e., 1) of the transition
        # matrix.
        check_is_fitted(self, self.check_fitted)
        eigvals, eigvecs = np.linalg.eig(self.transmat_.T)
        eigvec = np.real_if_close(eigvecs[:, np.argmax(eigvals)])
        return eigvec / eigvec.sum()

    def _trim_array(self, X):
        return X

    def print_progress(self):
        pass

    def robust_predict_proba(self, X):
        """Compute the robust posteriors across replicates.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        Returns
        -------

        posteriors_mean : array, shape (n_samples, n_components)
            Average State-membership probabilities for each sample in ``X`` across replicates.

        posteriors_std : array, shape (n_samples, n_components)
            Std. dev. State-membership probabilities for each sample in ``X`` across replicates.

        See Also
        --------
        score : Compute the log probability under the model.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        X = self._trim_array(X)
        check_is_fitted(self, self.check_fitted)
        self._check()

        X = _check_array(X)

        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs, verbose=max(0,
                            self.verbose - 1))

        lengths = X[0].shape[0]//n_jobs

        results = parallel(delayed(batch_compute_posterior_robust)(self, get_batch(X, i, j))
                           for i, j in iter_from_X_lengths(X, lengths))

        posteriors_means, posteriors_stds = zip(*results)

        posteriors_means = np.vstack(posteriors_means)
        posteriors_stds = np.vstack(posteriors_stds)
        return posteriors_means, posteriors_stds

    def score_samples(self, X, lengths=None):
        """Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample in ``X``.

        See Also
        --------
        score : Compute the log probability under the model.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        X = self._trim_array(X)
        check_is_fitted(self, self.check_fitted)
        self._check()

        X = _check_array(X)

        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs, verbose=max(0,
                            self.verbose - 1))

        lengths = X[0].shape[0]//n_jobs

        results = parallel(delayed(batch_compute_posterior)(self, get_batch(X, i, j))
                           for i, j in iter_from_X_lengths(X, lengths))

        _, posteriors, _, _, logprob_ = zip(*results)

        logprob = sum(logprob_)
        posteriors = np.vstack(posteriors)
        return logprob, posteriors

    def score(self, X, lengths=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        X = self._trim_array(X)
        check_is_fitted(self, self.check_fitted)
        self._check()

        X = _check_array(X)

        curr_logprob = 0

        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs, verbose=max(0,
                            self.verbose - 1))

        lengths = X[0].shape[0]//n_jobs

        results = parallel(delayed(batch_compute_loglikeli)(self, get_batch(X, i, j))
                           for i, j in iter_from_X_lengths(X, lengths))

        _, _, logprob_ = zip(*results)

        logprob = sum(logprob_)
        return logprob

    def _decode_viterbi(self, X):
        framelogprob = self._compute_log_likelihood(X)
        return self._do_viterbi_pass(framelogprob)

    def _decode_map(self, X):
        _, posteriors = self.score_samples(X)
        logprob = np.log(np.max(posteriors, axis=1)).sum()
        state_sequence = np.argmax(posteriors, axis=1)
        return logprob, state_sequence

    def _robust_decode_map(self, X):
        posteriors, _ = self.robust_predict_proba(X)
        logprob = np.log(np.max(posteriors, axis=1)).sum()
        state_sequence = np.argmax(posteriors, axis=1)
        return logprob, state_sequence

    def decode(self, X, lengths=None, algorithm=None):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        algorithm : string
            Decoder algorithm. Must be one of "viterbi" or "map".
            If not given, :attr:`decoder` is used.

        Returns
        -------
        logprob : float
            Log probability of the produced state sequence.

        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X`` obtained via a given
            decoder ``algorithm``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        score : Compute the log probability under the model.
        """
        X = self._trim_array(X)
        check_is_fitted(self, self.check_fitted)
        self._check()

        algorithm = algorithm or self.algorithm
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError("Unknown decoder {!r}".format(algorithm))

        decoder = {
            "viterbi": self._decode_viterbi,
            "map": self._decode_map,
            #"robust_map": self._robust_decode_map
        }[algorithm]

        X = _check_array(X)
        n_samples = get_nsamples(X)
        logprob = 0
        state_sequence = np.empty(n_samples, dtype=int)
        for i, j in iter_from_X_lengths(X, lengths):
            # XXX decoder works on a single sample at a time!
            logprobij, state_sequenceij = decoder(get_batch(X, i, j))
            logprob += logprobij
            state_sequence[i:j] = state_sequenceij

        return logprob, state_sequence

    def predict(self, X, lengths=None, algorithm=None):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X``.
        """
        _, state_sequence = self.decode(X, lengths, algorithm)
        return state_sequence

    def predict_proba(self, X, lengths=None,  algorithm=None):
        """Compute the posterior probability for each state in the model.

        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample from ``X``.
        """

        algorithm = algorithm or self.algorithm

        if algorithm == 'robust_map':
            posteriors, _ = self.robust_predict_proba(X)
        else:
            _, posteriors = self.score_samples(X, lengths)
        return posteriors

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        random_state : RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Feature matrix.

        state_sequence : array, shape (n_samples, )
            State sequence produced by the model.
        """
        check_is_fitted(self, self.check_fitted)
        self._check()

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        startprob_cdf = np.cumsum(self.startprob_)
        transmat_cdf = np.cumsum(self.transmat_, axis=1)

        currstate = (startprob_cdf > random_state.rand()).argmax()
        state_sequence = [currstate]
        X = [self._generate_sample_from_state(
            currstate, random_state=random_state)]

        for t in range(n_samples - 1):
            currstate = (transmat_cdf[currstate] > random_state.rand()) \
                .argmax()
            state_sequence.append(currstate)
            X.append(self._generate_sample_from_state(
                currstate, random_state=random_state))

        return np.atleast_2d(X), np.array(state_sequence, dtype=int)

    def fit(self, X, lengths=None):
        """Estimate model parameters.

        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._trim_array(X)
        X = _check_array(X)
        self._init(X, lengths=lengths)
        self._check()

        self.monitor_._reset()
        if True:
            n_jobs = effective_n_jobs(self.n_jobs)
            parallel = Parallel(n_jobs=n_jobs, verbose=max(0,
                                self.verbose - 1))

            lengths = X[0].shape[0]//n_jobs

            for iter_ in range(self.n_iter):
                #stats = self._initialize_sufficient_statistics()

                curr_logprob = 0

                results = parallel(delayed(batch_accumulate_suff_state)(self, get_batch(X, i, j))
                                   for i, j in iter_from_X_lengths(X, lengths))

                _, _, _, _, logprob, statssub = zip(*results)
                n = 0
                stats = self._initialize_sufficient_statistics()
                for i, j in iter_from_X_lengths(X, lengths):
                    for k in stats:
                        if isinstance(stats[k], list):
                            for i, _ in enumerate(stats[k]):
                                stats[k][i] += statssub[n][k][i]
                        else:
                            stats[k] += statssub[n][k]
                    curr_logprob += logprob[n]
                    n += 1

                # XXX must be before convergence check, because otherwise
                #     there won't be any updates for the case ``n_iter=1``.
                self._do_mstep(stats)

                self.print_progress()
                delta = curr_logprob - self.monitor_.history[-1] if self.monitor_.history else np.nan
                logging.debug(self.monitor_._template.format(iter=iter_+1, log_prob=curr_logprob, delta=delta))
                self.monitor_.report(curr_logprob)
                if self.monitor_.converged:
                    break

        return self

    def _do_viterbi_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        state_sequence, logprob = _hmmc.viterbi(self.startprob_,
                                                 self.transmat_,
                                                 framelogprob)
        return logprob, state_sequence

    def _do_forward_pass(self, framelogprob):
        log_prob, fwdlattice = _hmmc.forward_log(self.startprob_,
                                                 self.transmat_,
                                                 framelogprob)
        return log_prob, fwdlattice

    def _do_backward_pass(self, framelogprob):
        bwdlattice = _hmmc.backward_log(self.startprob_,
                           self.transmat_,
                           framelogprob)
        return bwdlattice

    def _compute_posteriors(self, fwdlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma = fwdlattice + bwdlattice
        log_normalize(log_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(log_gamma)

    def _init(self, X, lengths):
        """Initializes model parameters prior to fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        """
        init = 1. / self.n_components
        if 's' in self.init_params or not hasattr(self, "startprob_"):
            self.startprob_ = np.full(self.n_components, init)
        if 't' in self.init_params or not hasattr(self, "transmat_"):
            self.transmat_ = np.full((self.n_components, self.n_components),
                                     init)

    def _check(self):
        """Validates model parameters prior to fitting.

        Raises
        ------

        ValueError
            If any of the parameters are invalid, e.g. if :attr:`startprob_`
            don't sum to 1.
        """
        self.startprob_ = np.asarray(self.startprob_)
        if len(self.startprob_) != self.n_components:
            raise ValueError("startprob_ must have length n_components")
        if not np.allclose(self.startprob_.sum(), 1.0):
            raise ValueError("startprob_ must sum to 1.0 (got {:.4f})"
                             .format(self.startprob_.sum()))
        assert np.all(self.startprob_ > 0.0), "negative start detected"

        self.transmat_ = np.asarray(self.transmat_)
        if self.transmat_.shape != (self.n_components, self.n_components):
            raise ValueError(
                "transmat_ must have shape (n_components, n_components)")
        if not np.allclose(self.transmat_.sum(axis=1), 1.0):
            raise ValueError("rows of transmat_ must sum to 1.0 (got {})"
                             .format(self.transmat_.sum(axis=1)))
        assert np.all(self.transmat_ > 0.0), "negative transitions detected"

    def _compute_log_likelihood(self, X):
        """Computes per-component log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        Returns
        -------
        logprob : array, shape (n_samples, n_components)
            Log probability of each sample in ``X`` for each of the
            model states.
        """

    def _generate_sample_from_state(self, state, random_state=None):
        """Generates a random sample from a given component.

        Parameters
        ----------
        state : int
            Index of the component to condition on.

        random_state: RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.

        Returns
        -------
        X : array, shape (n_features, )
            A random sample from the emission distribution corresponding
            to a given component.
        """

    # Methods used by self.fit()

    def _initialize_sufficient_statistics(self):
        """Initializes sufficient statistics required for M-step.

        The method is *pure*, meaning that it doesn't change the state of
        the instance.  For extensibility computed statistics are stored
        in a dictionary.

        Returns
        -------
        nobs : int
            Number of samples in the data.

        start : array, shape (n_components, )
            An array where the i-th element corresponds to the posterior
            probability of the first sample being generated by the i-th
            state.

        trans : array, shape (n_components, n_components)
            An array where the (i, j)-th element corresponds to the
            posterior probability of transitioning between the i-th to j-th
            states.
        """
        stats = {'nobs': 0,
                 'start': np.zeros(self.n_components),
                 'trans': np.zeros((self.n_components, self.n_components))}
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        """Updates sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~base._BaseHMM._initialize_sufficient_statistics`.

        X : array, shape (n_samples, n_features)
            Sample sequence.

        framelogprob : array, shape (n_samples, n_components)
            Log-probabilities of each sample under each of the model states.

        posteriors : array, shape (n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.

        fwdlattice, bwdlattice : array, shape (n_samples, n_components)
            Log-forward and log-backward probabilities.
        """
        stats['nobs'] += 1
        if 's' in self.params:
            stats['start'] += posteriors[0]
        if 't' in self.params:
            n_samples, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return

            log_xi_sum = _hmmc.compute_log_xi_sum(fwdlattice,
                                                  self.transmat_,
                                                  bwdlattice, framelogprob)
            with np.errstate(under="ignore"):
                stats['trans'] += np.exp(log_xi_sum)

    def _do_mstep(self, stats):
        """Performs the M-step of EM algorithm.

        Parameters
        ----------
        stats : dict
            Sufficient statistics updated from all available samples.
        """
        # The ``np.where`` calls guard against updating forbidden states
        # or transitions in e.g. a left-right HMM.
        if 's' in self.params:
            startprob_ = self.startprob_prior - 1.0 + stats['start'] + EPS
            self.startprob_ = np.where(self.startprob_ == 0.0,
                                       self.startprob_, startprob_)
            normalize(self.startprob_)
        if 't' in self.params:
            transmat_ = self.transmat_prior - 1.0 + stats['trans'] + EPS
            self.transmat_ = np.where(self.transmat_ == 0.0,
                                      self.transmat_, transmat_)
            normalize(self.transmat_, axis=1)


