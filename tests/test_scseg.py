
import numpy as np
from scseg.cli import main
from scseg._hmm import _forward
from scseg._hmm import _backward
from scseg._hmm import _compute_theta_sstats
from scseg._hmm import _compute_beta_sstats
from scseg._hmm import _compute_log_reg_targets
from scipy.special import logsumexp
from sklearn.utils import check_random_state
#from sklearn.decomposition._online_lda import _dirichlet_expectation_1d
from sklearn.decomposition._online_lda import _dirichlet_expectation_2d
from sklearn.decomposition._online_lda import mean_change
from sklearn.decomposition.online_lda import _update_doc_distribution
from sklearn.decomposition import LatentDirichletAllocation
from scseg.mlda import _update_doc_distribution_markovlda
from scseg.mlda import _update_doc_distribution_lda

from scseg import CountMatrix
from scseg import Scseg

np.seterr(divide='ignore')

def logsigmoid(x):
    return -np.log(1. + np.exp(-x))

def test_forward():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([-.3, -.5])
    counts = np.ones(3)
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    results = np.zeros((n_samples, n_topics, 2))

    log_same = np.log(np.eye(n_topics))

    _forward(n_samples, n_topics, counts, log_theta, log_beta, sigarg, results)

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
    counts = np.ones(3)
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    results = np.zeros((n_samples, n_topics, 2))

    log_same = np.log(np.eye(n_topics))

    _forward(n_samples, n_topics, counts, log_theta, log_beta, sigarg, results)

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

def test_forward_almost_hmm():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([-20., -20])
    counts = np.ones(3)
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    results = np.zeros((n_samples, n_topics, 2))

    log_same = np.log(np.eye(n_topics))

    _forward(n_samples, n_topics, counts, log_theta, log_beta, sigarg, results)

    # init step
    np.testing.assert_equal(results[0], np.array([[-2.1, -np.Inf], [-2.5, -np.Inf]]))

    for p in range(2):
        r = np.zeros((n_topics, n_topics, 2))
        for i in range(r.shape[0]):
            for j in range(r.shape[0]):
                r[i, j, 0] = logsumexp(results[p], axis=1)[i] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
                r[i, j, 1] = logsumexp(results[p], axis=1)[i] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
        np.testing.assert_allclose(logsumexp(r, axis=0), results[p+1])

    # test if total log likelihood is the same as of the original LDA model
    #np.testing.assert_allclose(logsumexp(results[-1]), np.log(np.dot(np.exp(log_theta), np.exp(log_beta))).sum())



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
    r = np.zeros((n_topics, n_topics,2))
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
            r[i, j, 0] = results[p+1, j] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
            r[i, j, 1] = results[p+1, j] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
    #r = logsumexp(r, axis=2)
    np.testing.assert_allclose(logsumexp(r, axis=(1,2)), results[p])

    for p in [1, 0]:
        r = np.zeros((n_topics, n_topics, 2))
        for i in range(r.shape[0]):
            for j in range(r.shape[0]):
                r[i, j, 0] = results[p+1, j] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
                r[i, j, 1] = results[p+1, j] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
        np.testing.assert_allclose(logsumexp(r, axis=(1,2)), results[p])


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
    r = np.zeros((n_topics, n_topics,2))
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
            r[i, j, 0] = results[p+1, j] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
            r[i, j, 1] = results[p+1, j] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
    #r = logsumexp(r, axis=2)
    np.testing.assert_allclose(logsumexp(r, axis=(1,2)), results[p])

    for p in [1, 0]:
        r = np.zeros((n_topics, n_topics, 2))
        for i in range(r.shape[0]):
            for j in range(r.shape[0]):
                r[i, j, 0] = results[p+1, j] + logsigmoid(-sigarg[p]) + log_same[i, j] + log_beta[j, p+1]
                r[i, j, 1] = results[p+1, j] + logsigmoid(sigarg[p]) + log_theta[j] + log_beta[j, p+1]
        np.testing.assert_allclose(logsumexp(r, axis=(1,2)), results[p])

    np.testing.assert_allclose(results[:,0], results[:,1])


def test_forward_backward_logreg_stats():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([-.3, -.5])
    counts = np.ones(3)
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    bresults = np.zeros((n_samples, n_topics))
    fresults = np.zeros((n_samples, n_topics, 2))

    dist_targets = np.zeros(2)


    _backward(n_samples, n_topics, log_theta, log_beta, sigarg, bresults)
    _forward(n_samples, n_topics, counts, log_theta, log_beta, sigarg, fresults)


    _compute_log_reg_targets(n_samples, n_topics, counts, fresults, bresults, dist_targets)
    assert np.all(dist_targets>0.)
    assert np.all(dist_targets<1.)
    #_compute_beta_sstats(n_samples, n_topics, counts, fresults, bresults, log_beta_stats)


def test_forward_backward_theta_stats():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([-.3, -.5])
    counts = np.ones(3)
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    bresults = np.zeros((n_samples, n_topics))
    fresults = np.zeros((n_samples, n_topics, 2))
    log_theta_stats = np.zeros(2)


    _backward(n_samples, n_topics, log_theta, log_beta, sigarg, bresults)
    _forward(n_samples, n_topics, counts, log_theta, log_beta, sigarg, fresults)

    _compute_theta_sstats(n_samples, n_topics, counts, fresults, bresults, log_theta_stats)

    assert np.all(log_theta_stats>0.)
    assert log_theta_stats.sum() < counts.sum()


def test_forward_backward_beta_stats():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([-.3, -.5])
    counts = np.ones(3)
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    bresults = np.zeros((n_samples, n_topics))
    fresults = np.zeros((n_samples, n_topics, 2))

    log_beta_stats = np.zeros((2,3))

    _backward(n_samples, n_topics, log_theta, log_beta, sigarg, bresults)
    _forward(n_samples, n_topics, counts, log_theta, log_beta, sigarg, fresults)

    _compute_beta_sstats(n_samples, n_topics, counts, fresults, bresults, log_beta_stats)
    np.testing.assert_allclose(log_beta_stats.sum(), 3)
    np.testing.assert_allclose(log_beta_stats.sum(0), counts)


def test_forward_backward_logreg_stats_lda():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([20., 20.])
    counts = np.ones(3)
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    bresults = np.zeros((n_samples, n_topics))
    fresults = np.zeros((n_samples, n_topics, 2))

    dist_targets = np.zeros(2)


    _backward(n_samples, n_topics, log_theta, log_beta, sigarg, bresults)
    _forward(n_samples, n_topics, counts, log_theta, log_beta, sigarg, fresults)


    _compute_log_reg_targets(n_samples, n_topics, counts, fresults, bresults, dist_targets)

    np.testing.assert_allclose(dist_targets, np.ones(2))

    # with more counts
    counts[1] = 3

    _compute_log_reg_targets(n_samples, n_topics, counts, fresults, bresults, dist_targets)

    np.testing.assert_allclose(dist_targets, np.ones(2))


def test_forward_backward_theta_stats_lda():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([20., 20.])
    counts = np.ones(3)
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    bresults = np.zeros((n_samples, n_topics))
    fresults = np.zeros((n_samples, n_topics, 2))

    log_theta_stats = np.zeros(2)

    _backward(n_samples, n_topics, log_theta, log_beta, sigarg, bresults)
    _forward(n_samples, n_topics, counts, log_theta, log_beta, sigarg, fresults)

    _compute_theta_sstats(n_samples, n_topics, counts, fresults, bresults, log_theta_stats)

    np.testing.assert_allclose(log_theta_stats.sum(), counts.sum())

    # with more counts
    counts[1] = 3
    _compute_theta_sstats(n_samples, n_topics, counts, fresults, bresults, log_theta_stats)

    np.testing.assert_allclose(log_theta_stats.sum(), counts.sum())


def test_forward_backward_theta_stats_hmm():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([-20., -20.])
    counts = np.ones(3)
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    bresults = np.zeros((n_samples, n_topics))
    fresults = np.zeros((n_samples, n_topics, 2))

    log_theta_stats = np.zeros(2)

    _backward(n_samples, n_topics, log_theta, log_beta, sigarg, bresults)
    _forward(n_samples, n_topics, counts, log_theta, log_beta, sigarg, fresults)

    np.testing.assert_allclose(logsumexp(fresults, -1) + bresults, np.dot(np.ones((3,1)), np.array([[-2.4, -3.8]])))

    _compute_theta_sstats(n_samples, n_topics, counts, fresults, bresults, log_theta_stats)

    # should be only 1, because top down evidence is induced only once in the
    # beginning. Subsequently, the hmm is locked into its states and it can't
    # switch to another state.
    np.testing.assert_allclose(log_theta_stats.sum(), 1)
    np.testing.assert_allclose(log_theta_stats, np.array([0.80218389, 0.19781611]))

    # with more counts
    counts[1] = 3
    _compute_theta_sstats(n_samples, n_topics, counts, fresults, bresults, log_theta_stats)

    #the same is true here. bottom down evidence is observed only once.
    np.testing.assert_allclose(log_theta_stats.sum(), 1)
    np.testing.assert_allclose(log_theta_stats, np.array([0.80218389, 0.19781611]))


def test_forward_backward_beta_stats_lda():
    n_topics = 2
    n_samples = 3

    sigarg = np.array([20., 20.])
    counts = np.ones(3)
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])
    bresults = np.zeros((n_samples, n_topics))
    fresults = np.zeros((n_samples, n_topics, 2))
    log_beta_stats = np.zeros((2,3))

    _backward(n_samples, n_topics, log_theta, log_beta, sigarg, bresults)
    _forward(n_samples, n_topics, counts, log_theta, log_beta, sigarg, fresults)

    _compute_beta_sstats(n_samples, n_topics, counts, fresults, bresults, log_beta_stats)

    np.testing.assert_allclose(log_beta_stats.sum(), counts.sum())

    # with more counts
    counts[1] = 3
    _compute_beta_sstats(n_samples, n_topics, counts, fresults, bresults, log_beta_stats)

    np.testing.assert_allclose(log_beta_stats.sum(), counts.sum())


def test_fit_lda_legacy():

    countmatrixfile = '/local/wkopp/source/markovlda/genomebins_binsize2000.minmapq10.mincount0.tab.gz'
    bedfile =  '/local/wkopp/source/markovlda/genomebins_binsize2000.minmapq10.mincount0.bed.gz'
    minreadsincells = 1000
    minreadsinpeaks = 20
    maxreadsincells = 30000

    def trans(x):
        return np.log10(x/2000.)

    cm = CountMatrix.create_from_countmatrix(countmatrixfile, bedfile,
                               transform=trans)

    cm.filter_count_matrix(minreadsincells, maxreadsincells,
                           minreadsinpeaks, binarize=False)

    data = cm.cmat.T

    np.testing.assert_equal(data.shape, (1139, 219633))


    sseg = Scseg(10, max_iter=1)
    sseg.fit(data[:1])


def test_fit_mlda():

    countmatrixfile = '/local/wkopp/source/markovlda/genomebins_binsize2000.minmapq10.mincount0.tab.gz'
    bedfile =  '/local/wkopp/source/markovlda/genomebins_binsize2000.minmapq10.mincount0.bed.gz'
    minreadsincells = 1000
    minreadsinpeaks = 20
    maxreadsincells = 30000

    def trans(x):
        return np.log10(x/2000.)

    cm = CountMatrix.create_from_countmatrix(countmatrixfile, bedfile,
                               transform=trans)

    cm.filter_count_matrix(minreadsincells, maxreadsincells,
                           minreadsinpeaks, binarize=False)

    data = cm.cmat.T
    dists = cm.get_distance_matrix()

    np.testing.assert_equal(data.shape, (1139, 219633))
    np.testing.assert_equal(dists.shape, (1139, 16410))

    sseg = Scseg(10, max_iter=1)

    sseg.fit(data[:1], dists[:1])


def test_e_step():
    countmatrixfile = '/local/wkopp/source/markovlda/genomebins_binsize2000.minmapq10.mincount0.tab.gz'
    bedfile =  '/local/wkopp/source/markovlda/genomebins_binsize2000.minmapq10.mincount0.bed.gz'
    minreadsincells = 1000
    minreadsinpeaks = 20
    maxreadsincells = 30000

    def trans(x):
        return np.log10(x/2000.)

    cm = CountMatrix.create_from_countmatrix(countmatrixfile, bedfile,
                               transform=trans)

    cm.filter_count_matrix(minreadsincells, maxreadsincells,
                           minreadsinpeaks, binarize=False)

    data = cm.cmat.T
    dists = cm.get_distance_matrix()

    np.testing.assert_equal(data.shape, (1139, 219633))
    np.testing.assert_equal(dists.shape, (1139, 16410))

    data = data[:1]
    dists = dists[:1]

    n_components = 2
    sseg = Scseg(n_components)
    n_features = data.shape[1]
    doc_topic_prior = 1./n_components
    # push offset so that sigmoid responds with approx 1
    reg_weights = np.array([100., 0.])
    max_iters = sseg.max_doc_update_iter
    max_iters = 10
    max_dist = 100
    mean_change_tol = sseg.mean_change_tol
    cal_sstats = True

    random_state = check_random_state(0)
    components = random_state.gamma(
        100., 0.01, (n_components, n_features))
    #sseg.fit(data, dists)
    exp_dirichlet_component_ = _dirichlet_expectation_2d(components)

    exp_exp_dirichlet_component_ = np.exp(exp_dirichlet_component_)

    tdd1, sstat_1, _ = _update_doc_distribution_markovlda(data, dists, exp_dirichlet_component_, doc_topic_prior,
                                       reg_weights, max_dist, max_iters,
                                       mean_change_tol, cal_sstats, None)

    tdd2, sstat_2, _ = _update_doc_distribution_lda(data, exp_exp_dirichlet_component_, doc_topic_prior,
                                       max_iters,
                                       mean_change_tol, cal_sstats, None)


    tdd3, sstat_3 = _update_doc_distribution(data, exp_exp_dirichlet_component_, doc_topic_prior,
                                   max_iters,
                                   mean_change_tol, cal_sstats, None)

    np.testing.assert_allclose(tdd1, tdd2)
    np.testing.assert_allclose(tdd1, tdd3)
    sstat_3 *= exp_exp_dirichlet_component_
    np.testing.assert_allclose(sstat_1.sum(), sstat_3.sum())
    np.testing.assert_allclose(sstat_1.sum(), sstat_2.sum())
    np.testing.assert_equal(sstat_1.nonzero(), sstat_3.nonzero())
    np.testing.assert_equal(sstat_1.nonzero(), sstat_2.nonzero())
    np.testing.assert_allclose(sstat_2[:,[93,    259,    314]], sstat_3[:,[93,    259,    314]])
    np.testing.assert_allclose(sstat_1[:,[93,    259,    314]], sstat_3[:,[93,    259,    314]])
    np.testing.assert_allclose(sstat_1[:,[93,    259,    314]], sstat_2[:,[93,    259,    314]])
    np.testing.assert_allclose(sstat_1[sstat_1.nonzero()], sstat_3[sstat_3.nonzero()])
    np.testing.assert_allclose(sstat_1[sstat_1.nonzero()], sstat_2[sstat_2.nonzero()])


def test_score():
    countmatrixfile = '/local/wkopp/source/markovlda/genomebins_binsize2000.minmapq10.mincount0.tab.gz'
    bedfile =  '/local/wkopp/source/markovlda/genomebins_binsize2000.minmapq10.mincount0.bed.gz'
    minreadsincells = 1000
    minreadsinpeaks = 20
    maxreadsincells = 30000

    def trans(x):
        return np.log10(x/2000.)

    for binarize in [False, True]:
        print('using binarize={}'.format(binarize))

        cm = CountMatrix.create_from_countmatrix(countmatrixfile, bedfile,
                                                 transform=trans)

        cm.filter_count_matrix(minreadsincells, maxreadsincells,
                               minreadsinpeaks, binarize=True)

        data = cm.cmat.T
        dists = cm.get_distance_matrix()

        data = data[:10]
        dists = dists[:10]

        n_components = 5
        sseg = Scseg(n_components, random_state=0)
        sseg_markov = Scseg(n_components, random_state=0, no_regression=True,
                            reg_weights=np.array([100., 0.]))
        lda = LatentDirichletAllocation(n_components, random_state=0)

        # get score of original lda
        lda.fit(data)
        oscore = lda.score(data)

        # get score of lda version of the new code
        sseg.fit(data)
        new_lda_score = sseg.score(data)

        sseg_markov.fit(data, dists)
        mlda_score = sseg_markov.score(data, dists)

        np.testing.assert_allclose(oscore, new_lda_score)

        np.testing.assert_allclose(oscore, mlda_score)

        np.testing.assert_allclose(lda.components_, sseg.components_)
        np.testing.assert_allclose(sseg_markov.components_, sseg.components_)


def test_score_toyexample():
    from scseg import Scseg
    from sklearn.datasets import make_multilabel_classification


    for binarize in [True, False]:
        # This produces a feature matrix of token counts, similar to what
        # CountVectorizer would produce on text.
        data, _ = make_multilabel_classification(random_state=0)
        dists = np.zeros_like(data)
        print('using binarize={}'.format(binarize))

        if binarize:
            data[data>0] = 1

        n_components = 5
        sseg = Scseg(n_components, random_state=0)
        sseg_markov = Scseg(n_components, random_state=0, no_regression=True, reg_weights=np.array([100., 0.]))
        lda = LatentDirichletAllocation(n_components, random_state=0)

        # get score of original lda
        lda.fit(data)
        oscore = lda.score(data)

        # get score of lda version of the new code
        sseg.fit(data)
        new_lda_score = sseg.score(data)

        sseg_markov.fit(data, dists)
        mlda_score = sseg_markov.score(data, dists)
        doc_topic_distr1 = sseg._unnormalized_transform(data, None)
        doc_topic_distr2 = sseg_markov._unnormalized_transform(data, dists)
        np.testing.assert_allclose(doc_topic_distr1, doc_topic_distr2)

        np.testing.assert_allclose(lda.components_, sseg.components_)
        np.testing.assert_allclose(sseg_markov.components_, sseg.components_)

        np.testing.assert_allclose(oscore, new_lda_score)
        np.testing.assert_allclose(oscore, mlda_score)

        lll1 = sseg.compute_likelihood(data, None, doc_topic_distr1)
        lll2 = sseg_markov.compute_likelihood(data, dists, doc_topic_distr2)

        np.testing.assert_allclose(lll1, lll2)


def test_likelihood():
    # test if the likelihoods are identical between the forward
    # algorithm and the lda
    n_topics = 2
    n_samples = 3

    sigarg = np.array([100., 100.])
    counts = np.ones(3)
    log_beta = np.array([[-.1, -.1, -.2], [-.2, -1, -.3]])
    log_theta = np.array([-2, -2.3])

    fwdlattice = np.zeros((n_samples, n_topics, 2))

    _forward(n_samples, n_topics, counts, log_theta, log_beta, sigarg, fwdlattice)

    temp = (log_theta[:, np.newaxis]
            + log_beta[:])
    score = logsumexp(temp, axis=0).sum()
    np.testing.assert_allclose(score, logsumexp(fwdlattice[-1]))


def test_transform():
    countmatrixfile = '/local/wkopp/source/markovlda/genomebins_binsize2000.minmapq10.mincount0.tab.gz'
    bedfile =  '/local/wkopp/source/markovlda/genomebins_binsize2000.minmapq10.mincount0.bed.gz'
    minreadsincells = 1000
    minreadsinpeaks = 20
    maxreadsincells = 30000

    def trans(x):
        return np.log10(x/2000.)

    cm = CountMatrix.create_from_countmatrix(countmatrixfile, bedfile,
                               transform=trans)

    cm.filter_count_matrix(minreadsincells, maxreadsincells,
                           minreadsinpeaks, binarize=False)

    data = cm.cmat.T
    dists = cm.get_distance_matrix()

    np.testing.assert_equal(data.shape, (1139, 219633))
    np.testing.assert_equal(dists.shape, (1139, 16410))

    data = data[:1]
    dists = dists[:1]

    n_components = 2
    sseg = Scseg(n_components, random_state=0)
    sseg_markov = Scseg(n_components, random_state=0, no_regression=True, reg_weights=np.array([100., 0.]))
    lda = LatentDirichletAllocation(n_components, random_state=0)

    # get score of original lda
    lda.fit(data)
    otrans = lda.transform(data)

    # get score of lda version of the new code
    sseg.fit(data)
    new_lda_trans = sseg.transform(data)

    sseg_markov.fit(data, dists)
    mlda_trans = sseg_markov.transform(data, dists)

    np.testing.assert_allclose(otrans, new_lda_trans)

    np.testing.assert_allclose(otrans, mlda_trans)
