#import numpy as np
#from scseg import CountMatrix
#
#countmatrixfile = '/local/wkopp/source/markovlda/genomebins_binsize2000.minmapq10.mincount0.tab.gz'
#bedfile =  '/local/wkopp/source/markovlda/genomebins_binsize2000.minmapq10.mincount0.bed.gz'
#minreadsincells = 1000
#minreadsinpeaks = 20
#maxreadsincells = 30000
#
#cm = CountMatrix.create_from_countmatrix(countmatrixfile, bedfile,
#                           transform=trans)
#cm.filter_count_matrix(minreadsincells, maxreadsincells,
#                       minreadsinpeaks, binarize=False)
#
#def test_countmatrix():
#    np.testing.assert_equal(cm.shape, (219633, 1139))
#    np.testing.assert_equal(len(cm.get_peaks(0)), 1614)
#    np.testing.assert_equal(len(cm.get_distances(0)), 1613)
#    np.testing.assert_equal(cm.n_cells, 1139)
#    np.testing.assert_equal(cm.n_regions, 219633)
#
#def test_countmatrix_subset():
#    idxs = np.random.permutation(cm.n_cells)
#    ntest = 300
#    cmtest = cm.subset(idxs[:ntest])
#    cmtrain = cm.subset(idxs[ntest:])
#
#    np.testing.assert_equal(cmtest.n_cells, 300)
#    np.testing.assert_equal(cmtest.n_regions, 219633)
#    np.testing.assert_equal(cmtrain.n_cells, 839)
#    np.testing.assert_equal(cmtrain.n_regions, 219633)
#    np.testing.assert_equal(cmtrain.n_cells + cmtest.n_cells, cm.n_cells)
