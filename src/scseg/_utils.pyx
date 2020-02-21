
from cython.parallel import prange
import numpy as np
cimport numpy as np
from cython cimport view, boundscheck, wraparound
from libc.math cimport lgamma
from libc.stdio cimport printf

@boundscheck(False)
@wraparound(False)
def _fast_dirmul_loglikeli_sp(const int[:] x_indices,
                              const int[:] x_indptr,
                              const long[:] x_cnts,
                              const double[:,:] alpha,
                              int n_regions, double[:,:] result):
    """
    x : np.array
      regions x cell count matrix
    alpha : np.array
      state x cell parameter matrix
    """
    cdef int n_states = alpha.shape[0]
    cdef int n_cells = alpha.shape[1]
    cdef int r, c, s, m, ci, maxcounts
    cdef double[::view.contiguous] n = np.zeros(n_regions, dtype=np.float)
    cdef double[::view.contiguous] alpha0 = np.zeros(n_states, dtype=np.float)
    cdef double[:, :, :] precomp
    # = np.zeros((maxcounts, n_cells, n_states), dtype=np.float)

    maxcounts = 0
    with nogil:
    #if True:
        # init array
        for r in range(n_regions):
            for c in range(x_indptr[r], x_indptr[r+1]):
                n[r] += x_cnts[c]
                if maxcounts < x_cnts[c]:
                    maxcounts = x_cnts[c]
    
    precomp = np.zeros((maxcounts, n_cells, n_states), dtype=np.float)
    with nogil:
        for s in range(n_states):
            for c in range(n_cells):
                alpha0[s] += alpha[s, c]
    
        for r in range(n_regions):
            for s in range(n_states):
                result[r, s] = lgamma(alpha0[s]) - lgamma(n[r] + alpha0[s])
    
        # precompute gammas
        for m in range(1, maxcounts + 1):
            for c in range(n_cells):
                for s in range(n_states):
                    precomp[m-1, c, s] += lgamma(alpha[s, c] + m) - lgamma(alpha[s,c])
    
        # finally compute likelihood per datapoint and state
        for r in range(n_regions):
            #printf("%i\n", r)
            #for c in range(x_indices[x_indptr[i], x_indptr[i+1]]):
            for c in range(x_indptr[r], x_indptr[r+1]):
                ci = x_indices[c]
                m = x_cnts[c]# if x_cnts[c] < maxcounts else maxcounts - 1
                #m = x_cnts[c]
                for s in range(n_states):
                    result[r, s] += precomp[m-1, ci, s]
    return result    
