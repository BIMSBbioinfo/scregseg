# cython: language_level=3, boundscheck=False, wraparound=False

from cython cimport view
from numpy.math cimport expl, logl, tanhl, log1pl, isinf, fabsl, INFINITY, PI

import numpy as np

ctypedef double dtype_t

cdef inline dtype_t _max2d(dtype_t[:,:] X) nogil:
    cdef dtype_t X_max = -INFINITY
    cdef int i, j
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        if X[i,j] > X_max:
            X_max = X[i,j]

    return X_max


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


cdef inline dtype_t _logaddexp(dtype_t a, dtype_t b) nogil:
    if isinf(a) and a < 0:
        return b
    elif isinf(b) and b < 0:
        return a
    else:
        return max(a, b) + log1pl(expl(-fabsl(a - b)))


cdef inline dtype_t _logsigmoid(dtype_t x) nogil:
    return -logl( 1. + expl(-x))


cdef inline dtype_t _logsumexp2d(dtype_t[:,:] X) nogil:
    cdef dtype_t X_max = _max2d(X)
    if isinf(X_max):
        return -INFINITY

    cdef dtype_t acc = 0
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        acc += expl(X[i,j] - X_max)

    return logl(acc) + X_max


cdef inline dtype_t _logsumexp(dtype_t[:] X) nogil:
    cdef dtype_t X_max = _max(X)
    if isinf(X_max):
        return -INFINITY

    cdef dtype_t acc = 0
    for i in range(X.shape[0]):
        acc += expl(X[i] - X_max)

    return logl(acc) + X_max

def _forward(int n_samples, int n_components,
             dtype_t[:] log_theta,
             dtype_t[:, :] log_beta,
             dtype_t[:] sigmoid_arg,
             dtype_t[:, :, :] fwdlattice):

    cdef int t, i, j
    cdef dtype_t[::view.contiguous] wb0 = np.zeros(n_components)
    cdef dtype_t[::view.contiguous] wb1 = np.zeros(n_components)
    cdef dtype_t[:, ::view.contiguous] log_same = \
        np.full((n_components, n_components), -INFINITY)
    cdef dtype_t merged_fw

    with nogil:
        # initalize lattice
        for i in range(n_components):
            fwdlattice[0, i, 0] = log_theta[i] + log_beta[i, 0]
            fwdlattice[0, i, 1] = -INFINITY
            log_same[i,i] = 0.0

        for t in range(1, n_samples):
            for j in range(n_components):
                for i in range(n_components):
                    # sum over indep or same topic
                    merged_fw = _logsumexp(fwdlattice[t - 1, i])

                    #if an independent topic emerges:
                    wb0[i] = merged_fw

                    # if the same topic is extended:
                    wb1[i] = merged_fw + log_same[i, j]

                fwdlattice[t, j, 0] = _logsumexp(wb0) + log_beta[j, t] + log_theta[j] + _logsigmoid(sigmoid_arg[t - 1])
                fwdlattice[t, j, 1] = _logsumexp(wb1) + log_beta[j, t] + _logsigmoid(-sigmoid_arg[t - 1])


def _backward(int n_samples, int n_components,
             dtype_t[:] log_theta,
             dtype_t[:, :] log_beta,
             dtype_t[:] sigmoid_arg,
             dtype_t[:, :] bwdlattice):

    cdef int t, i, j
    cdef dtype_t[::view.contiguous] wb0 = np.zeros(n_components)
    cdef dtype_t[::view.contiguous] wb1 = np.zeros(n_components)
    cdef dtype_t[:,::view.contiguous] log_same = \
        np.full((n_components, n_components), -INFINITY)
    cdef dtype_t merged_fw

    with nogil:
        for i in range(n_components):
            bwdlattice[n_samples - 1, i] = 0.0
            log_same[i,i] = 0.0

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_components):
                for j in range(n_components):
                    wb1[j] = log_same[i, j] + _logsigmoid(-sigmoid_arg[t]) + log_beta[j, t + 1] + bwdlattice[t + 1, j]
                bwdlattice[t, i] = _logsumexp(wb1)

def _suffstats_hmm(int n_samples, int n_components,
                   dtype_t[:] num_words,
                   dtype_t[:,:,:] fwdlattice,
                   dtype_t[:,:] bwdlattice,
                   dtype_t[:] log_theta_stats,
                   dtype_t[:] reg_target_stats):

    cdef int t, i, j
    cdef dtype_t[:,::view.contiguous] wb = np.zeros((n_components, 2))
    cdef dtype_t partition
    cdef dtype_t val

    with nogil:
      for i in range(n_components):
        log_theta_stats[i] = 0.0

      for t in range(n_samples - 1):
        reg_target_stats[t] = 0.0

      for t in range(n_samples):
        for i in range(n_components):
          wb[i, 0] = fwdlattice[t, i, 0] + bwdlattice[t, i]
          wb[i, 1] = fwdlattice[t, i, 1] + bwdlattice[t, i]
        partition = _logsumexp2d(wb)
        for i in range(n_components):
          val = expl(wb[i, 0] - partition)
          log_theta_stats[i] += val * num_words[t]
          if t > 0:
            reg_target_stats[t-1] += val
