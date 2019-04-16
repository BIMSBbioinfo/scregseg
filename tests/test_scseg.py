
import numpy as np
from scseg.cli import main
from scseg._hmm import _forward
from scseg._hmm import _backward
from scseg._hmm import _suffstats_hmm
from scipy.special import logsumexp

np.seterr(divide='ignore')

def logsigmoid(x):
    return -np.log(1. + np.exp(-x))

def test_forward():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([-.3, -.5])
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    results = np.zeros((n_samples, n_topics, 2))

    log_same = np.log(np.eye(n_topics))

    _forward(n_samples, n_topics, log_theta, log_beta, sigarg, results)

    # init step
    np.testing.assert_equal(results[0], np.array([[-2.1, -np.Inf], [-2.5, -np.Inf]]))

    p = 0
    r = np.zeros((n_topics, n_topics, 2))
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
            r[i, j, 0] = logsumexp(results[p], axis=1)[i] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
            r[i, j, 1] = logsumexp(results[p], axis=1)[i] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
    np.testing.assert_allclose(logsumexp(r, axis=0), results[p+1])

    p = 1
    r = np.zeros((n_topics, n_topics, 2))
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
            r[i, j, 0] = logsumexp(results[p], axis=1)[i] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
            r[i, j, 1] = logsumexp(results[p], axis=1)[i] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
    np.testing.assert_allclose(logsumexp(r, axis=0), results[p+1])

    j= 0
    logsumexp(logsumexp(results[p], axis=1) + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1])
    for p in range(2):
        r = np.zeros((n_topics, n_topics, 2))
        for i in range(r.shape[0]):
            for j in range(r.shape[0]):
                r[i, j, 0] = logsumexp(results[p], axis=1)[i] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
                r[i, j, 1] = logsumexp(results[p], axis=1)[i] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
        np.testing.assert_allclose(logsumexp(r, axis=0), results[p+1])


def test_forward_almost_lda():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([20., 20])
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    results = np.zeros((n_samples, n_topics, 2))

    log_same = np.log(np.eye(n_topics))

    _forward(n_samples, n_topics, log_theta, log_beta, sigarg, results)

    # init step
    np.testing.assert_equal(results[0], np.array([[-2.1, -np.Inf], [-2.5, -np.Inf]]))

    p = 0
    r = np.zeros((n_topics, n_topics, 2))
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
            r[i, j, 0] = logsumexp(results[p], axis=1)[i] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
            r[i, j, 1] = logsumexp(results[p], axis=1)[i] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
    np.testing.assert_allclose(logsumexp(r, axis=0), results[p+1])

    p = 1
    r = np.zeros((n_topics, n_topics, 2))
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
            r[i, j, 0] = logsumexp(results[p], axis=1)[i] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
            r[i, j, 1] = logsumexp(results[p], axis=1)[i] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
    np.testing.assert_allclose(logsumexp(r, axis=0), results[p+1])

    j= 0
    logsumexp(logsumexp(results[p], axis=1) + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1])
    for p in range(2):
        r = np.zeros((n_topics, n_topics, 2))
        for i in range(r.shape[0]):
            for j in range(r.shape[0]):
                r[i, j, 0] = logsumexp(results[p], axis=1)[i] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
                r[i, j, 1] = logsumexp(results[p], axis=1)[i] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
        np.testing.assert_allclose(logsumexp(r, axis=0), results[p+1])

    # test if total log likelihood is the same as of the original LDA model
    np.testing.assert_allclose(logsumexp(results[-1]), np.log(np.dot(np.exp(log_theta), np.exp(log_beta))).sum())



def test_backward():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([-.3, -.5])
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    results = np.zeros((n_samples, n_topics))
    log_same = np.log(np.eye(n_topics))

    _backward(n_samples, n_topics, log_theta, log_beta, sigarg, results)

    # init step
    np.testing.assert_equal(results[-1], np.zeros(2))

    p = 1
    r = np.zeros((n_topics, n_topics))
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
        #    r[i, j, 0] = logsumexp(results[p], axis=1)[i] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
            r[i, j] = results[p+1, j] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
    np.testing.assert_allclose(logsumexp(r, axis=0), results[p])

    for p in [1, 0]:
        r = np.zeros((n_topics, n_topics))
        for i in range(r.shape[0]):
            for j in range(r.shape[0]):
            #    r[i, j, 0] = logsumexp(results[p], axis=1)[i] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
                r[i, j] = results[p+1, j] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
        np.testing.assert_allclose(logsumexp(r, axis=0), results[p])


def test_backward_almost_lda():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([20., 20])
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    results = np.zeros((n_samples, n_topics))
    log_same = np.log(np.eye(n_topics))

    _backward(n_samples, n_topics, log_theta, log_beta, sigarg, results)

    # init step
    np.testing.assert_equal(results[-1], np.zeros(2))

    p = 1
    r = np.zeros((n_topics, n_topics))
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
        #    r[i, j, 0] = logsumexp(results[p], axis=1)[i] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
            r[i, j] = results[p+1, j] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
    np.testing.assert_allclose(logsumexp(r, axis=0), results[p])

    for p in [1, 0]:
        r = np.zeros((n_topics, n_topics))
        for i in range(r.shape[0]):
            for j in range(r.shape[0]):
            #    r[i, j, 0] = logsumexp(results[p], axis=1)[i] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
                r[i, j] = results[p+1, j] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
        np.testing.assert_allclose(logsumexp(r, axis=0), results[p])

def test_forward_backward_update_stats():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([-.3, -.5])
    counts = np.ones(3)
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    bresults = np.zeros((n_samples, n_topics))
    fresults = np.zeros((n_samples, n_topics, 2))
    #log_same = np.log(np.eye(n_topics))
    log_theta_stats = np.zeros(2)
    dist_targets = np.zeros(2)

    _backward(n_samples, n_topics, log_theta, log_beta, sigarg, bresults)
    _forward(n_samples, n_topics, log_theta, log_beta, sigarg, fresults)

    _suffstats_hmm(n_samples, n_topics, counts, fresults, bresults, log_theta_stats, dist_targets)


def test_forward_backward_update_stats_lda():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([20., 20.])
    counts = np.ones(3)
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    bresults = np.zeros((n_samples, n_topics))
    fresults = np.zeros((n_samples, n_topics, 2))
    #log_same = np.log(np.eye(n_topics))
    log_theta_stats = np.zeros(2)
    dist_targets = np.zeros(2)

    _backward(n_samples, n_topics, log_theta, log_beta, sigarg, bresults)
    _forward(n_samples, n_topics, log_theta, log_beta, sigarg, fresults)

    _suffstats_hmm(n_samples, n_topics, counts, fresults, bresults, log_theta_stats, dist_targets)

    np.testing.assert_allclose(log_theta_stats.sum(), counts.sum())
    np.testing.assert_allclose(dist_targets, np.ones(2))

    # with more counts
    counts[1] = 3
    _suffstats_hmm(n_samples, n_topics, counts, fresults, bresults, log_theta_stats, dist_targets)

    np.testing.assert_allclose(log_theta_stats.sum(), counts.sum())
    np.testing.assert_allclose(dist_targets, np.ones(2))
