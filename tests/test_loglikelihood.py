import time
import numpy as np
from scipy.sparse import csr_matrix

from scregseg.hmm import dirmul_loglikeli_naive
from scregseg.hmm import dirmul_loglikeli
from scregseg.hmm import dirmul_loglikeli_sp
from scregseg.hmm import fast_dirmul_loglikeli_sp
from scregseg.hmm import _fast_dirmul_loglikeli_sp

def test_loglikelihood():
    #10 regions, 3 cells, 4 states
    x = csr_matrix(np.random.choice(3,(1000,300)))
    alpha = np.random.rand(4,300)
    r1 = dirmul_loglikeli_naive(x,alpha)

    r2 = dirmul_loglikeli(x,alpha)
    r3 = dirmul_loglikeli_sp(x,alpha)
    r4 = fast_dirmul_loglikeli_sp(x, alpha)

    np.testing.assert_allclose(r1, r2)
    np.testing.assert_allclose(r1, r3)
    np.testing.assert_allclose(r1, r4)

def test_loglikelihood2():
    #10 regions, 3 cells, 4 states
    x = csr_matrix(np.random.choice(5,(1000,300)))
    alpha = np.random.rand(4,300)
    r1 = dirmul_loglikeli_sp(x,alpha)
    r2 = fast_dirmul_loglikeli_sp(x, alpha)

    np.testing.assert_allclose(r1, r2)

def test_loglikelihood3():
    #10 regions, 3 cells, 4 states
    x = csr_matrix(np.random.choice(5,(1000,300)))
    maxcount = 3
    x.data[x.data>=maxcount] = maxcount -1
    alpha = np.random.rand(4,300)
    r1 = dirmul_loglikeli_sp(x,alpha)
    r2 = fast_dirmul_loglikeli_sp(x, alpha)

    np.testing.assert_allclose(r1, r2)

def test_loglikelihood4():
    #10 regions, 3 cells, 4 states
    x = csr_matrix(np.random.choice(5,(1000,300)))
    alpha = np.random.rand(4,300)
    r1 = dirmul_loglikeli_sp(x,alpha)
    r3 = fast_dirmul_loglikeli_sp(x, alpha)

    np.testing.assert_allclose(r1, r3)
