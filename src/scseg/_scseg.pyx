# cython: language_level=3, boundscheck=False, wraparound=False

from cython cimport view
from numpy.math cimport expl, logl, tanhl, log1pl, isinf, fabsl, INFINITY, PI

import numpy as np

ctypedef double dtype_t


cdef inline int _argmax(dtype_t[:] X) nogil:
    cdef dtype_t X_max = -INFINITY
    cdef int pos = 0
    cdef int i
    for i in range(X.shape[0]):
        if X[i] > X_max:
            X_max = X[i]
            pos = i
    return pos


cdef inline dtype_t _max(dtype_t[:] X) nogil:
    return X[_argmax(X)]


cdef inline dtype_t _logsumexp(dtype_t[:] X) nogil:
    cdef dtype_t X_max = _max(X)
    if isinf(X_max):
        return -INFINITY

    cdef dtype_t acc = 0
    for i in range(X.shape[0]):
        acc += expl(X[i] - X_max)

    return logl(acc) + X_max


cdef inline dtype_t _logaddexp(dtype_t a, dtype_t b) nogil:
    if isinf(a) and a < 0:
        return b
    elif isinf(b) and b < 0:
        return a
    else:
        return max(a, b) + log1pl(expl(-fabsl(a - b)))

cdef inline dtype_t _logsigmoid(dtype_t x) nogil:
    return -logl( 1. + expl(-x))

cdef inline dtype_t _kappa(dtype_t x) nogil:
    return (1. + PI * x / 8.) ** (-1/2.)

cdef inline dtype_t Lambda(dtype_t xi) nogil:
    return tanhl(xi / 2.) / (4.*xi)

def compute_precision_suffstats(int n_elem,
                                dtype_t[:] dist,
                                dtype_t[:] xi,
                                dtype_t[:,:] precision):
    cdef int t, i, j
    cdef dtype_t[::view.contiguous] phi = np.ones(2)

    with nogil:
        for t in range(n_elem):
            phi[1] = dist[t]
            for i in range(2):
                for j in range(2):
                    precision[i, j] += 2 * Lambda(xi[t]) * phi[i] * phi[j]


#def compute_mean_suffstats(int n_elem, dtype_t[:] dist, dtype_t[:] ptarget, dtype_t[:] mean):
#    cdef int t, i, j
#
#    with nogil:
#        for t in range(n_elem):
#            mean[0] += ptarget[t] - 0.5
#            mean[1] += dist[t] * (ptarget[t] - 0.5)
#
#
#def log_posterior_predictive_sigmoid(int n_elem,
#                                 dtype_t[:] dist,
#                                 dtype_t[:] mu,
#                                 dtype_t[:,:] cov,
#                                 dtype_t[:] log_p):
#    cdef int t, i, j
#    cdef dtype_t mu_a, sigma_a
#    cdef dtype_t[::view.contiguous] work_buffer = np.ones(2)
#
#    with nogil:
#        for t in range(n_elem):
#            work_buffer[1] = dist[t]
#            mu_a = np.dot(mu, work_buffer)
#            sigma_a = np.dot(work_buffer, np.dot(cov, work_buffer))
#            log_p[t] = _logsigmoid(kappa(sigma_a)*mu_a)
#
#

def log_posterior_predictive_sigmoid_arg(int n_samples,
                                 dtype_t[:] dist,
                                 dtype_t[:] mu,
                                 dtype_t[:,:] cov,
                                 dtype_t[:] log_p_arg):
    cdef int t, i, j
    cdef dtype_t mu_a, sigma_a
    cdef dtype_t[::view.contiguous] work_buffer = np.ones(2)


    with nogil:
        for t in range(n_samples):
            sigma_a = 0.0
            work_buffer[1] = dist[t]
            mu_a = mu[0] * work_buffer[0] + mu[1] * work_buffer[1]
            for i in range(2):
                for j in range(2):
                    sigma_a += work_buffer[i] * cov[i,j] * work_buffer[j]
            #sigma_a = np.dot(work_buffer, np.dot(cov, work_buffer))
            log_p_arg[t] = _kappa(sigma_a)*mu_a
