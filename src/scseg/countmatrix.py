import copy
import gzip
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix


def get_count_matrix_(filename, shape, header=True, offset=0):
    if header:
        spdf = pd.read_csv(filename, sep='\t', skiprows=1)
    else:
        spdf = pd.read_csv(filename, sep='\t', skiprows=1, header=None, names=['region','cell','count'])
    if offset > 0:
        spdf.region -= offset
        spdf.cell -= offset
    smat = csc_matrix((spdf['count'], (spdf.region, spdf.cell)),
                      shape=shape, dtype='float')
    return smat

def get_cell_annotation_first_row_(filename):

    open_ = gzip.open if filename.endswith('.gz') else open
    with open_(filename, 'r') as f:
        line = f.readline()
        if hasattr(line, 'decode'):
            line = line.decode('utf-8')
        line = line.split('\n')[0]
        line = line.split(' ')[-1]
        line = line.split('\t')[-1]
        annot = line.split(';')
    return annot

def get_regions_from_bed_(filename):
    regions = pd.read_csv(filename, sep='\t', names=['chrom', 'start', 'end'], usecols=[0,1,2])
    return regions


class CountMatrix:

    @classmethod
    def create_from_countmatrix(cls, countmatrixfile, regionannotation, transform=None, header=True, index_offset=0):

        cannot = get_cell_annotation_first_row_(countmatrixfile)
        rannot = get_regions_from_bed_(regionannotation)
        shape = rannot.shape[0], len(cannot)
        cmat = get_count_matrix_(countmatrixfile, shape, header=header, offset=index_offset)
        return cls(cmat, rannot, cannot, transform)

    def __init__(self, countmatrix, regionannotation, cellannotation,
                 transform=None):

        self.cmat = countmatrix
        self.cannot = cellannotation
        self.regions = regionannotation
        #self.filter_count_matrix()
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    @property
    def chroms(self):
        return self.regions.chrom.values

    @property
    def starts(self):
        return self.regions.start.values

    @property
    def ends(self):
        return self.regions.end.values

    def filter_count_matrix(self, minreadsincells=1000, maxreadsincells=30000,
                            minreadsinpeaks=20,
                            binarize=True):
        if binarize:
            self.cmat.data[self.cmat.data > 0] = 1
        cellcounts = self.cmat.sum(axis=0)
        keepcells = np.where((cellcounts>=minreadsincells) & (cellcounts<maxreadsincells))[1]

        self.cmat = self.cmat[:, keepcells]
        self.cannot = [self.cannot[kid] for kid in keepcells]

        regioncounts = self.cmat.sum(axis=1)
        keepregions = np.where(regioncounts>=minreadsinpeaks)[0]

        self.cmat = self.cmat[keepregions, :]
        self.regions = self.regions.iloc[keepregions]

    def __call__(self, icell=None):
        if icell is None:
            return self.cmat.toarray()
        elif isinstance(icell, int):
            return self.cmat[:, icell].toarray()
        elif isinstance(icell, slice):
            return self.cmat[:, icell].toarray()
        raise ValueError("indexing not supported")

    def get_peaks(self, cell):
        return self.cmat[:, cell].nonzero()[0]

    def __getitem__(self, cell):
        return self.cmat[:, cell].data

    def get_distances(self, cell):
        reg = self.regions.iloc[self.get_peaks(cell)]

        dists = np.diff(reg.start.values)

        # no negative distances allowed, which might be due to
        # traversing chromosome boundaries.
        dists = np.abs(dists)

        dists = self.transform(dists)
        return dists

    def __repr__(self):
        return "{} x {} CountMatrix".format(self.cmat.shape[0], self.cmat.shape[1])

    @property
    def n_cells(self):
        return self.cmat.shape[1]

    @property
    def n_regions(self):
        return self.cmat.shape[0]

    @property
    def shape(self):
        return (self.n_regions, self.n_cells)

    @property
    def __len__(self):
        return self.n_regions

    @property
    def n_peaks_total(self):
        return self.cmat.nnz

    def subset(self, indices):
        return CountMatrix(self.cmat[:, indices], copy.copy(self.regions),
                    [self.cannot[kid] for kid in indices],
                    transform=self.transform)

    def get_distance_matrix(self):
        maxp = 0
        L = np.zeros(self.n_cells, dtype='int')
        for i in range(self.n_cells):
            L[i] = len(self.get_distances(i))
            if maxp < int(L[i]):
                maxp = int(L[i])
        R = np.zeros((self.n_cells, maxp))
        for i in range(self.n_cells):
            R[i, :L[i]] = self.get_distances(i)

        return R

def split_train_test(data):
    idxs = np.random.permutation(data.n_cells)
    ntest = int(data.n_cells * val_split)
    val_data = data.subset(idxs[:ntest])
    train_data = data.subset(idxs[ntest:])
    return train_data, val_data
