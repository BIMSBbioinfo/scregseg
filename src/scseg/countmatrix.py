import os
import copy
import gzip
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.sparse import hstack
from scipy.io import mmread, mmwrite
from pybedtools import BedTool, Interval
from pysam import AlignmentFile
from collections import Counter
from scipy.sparse import dok_matrix

class Barcoder:
    def __init__(self, tag):
        print('Barcodes determined from {} tag'.format(tag))
        self.tag = tag

    def __call__(self, aln):
        if aln.has_tag(self.tag):
            rg = aln.get_tag(self.tag)
        else:
            rg = 'dummy'
        return rg
        
def make_counting_bins(bamfile, binsize, storage=None):
    """ Genome intervals for binsize.

    For a given genome and binsize,
    this function creates a bed-file containing all intervals.
    """
    # Obtain the header information
    afile = AlignmentFile(bamfile, 'rb')

    # extract genome size

    genomesize = {}
    for chrom, length in zip(afile.references, afile.lengths):
        genomesize[chrom] = length
    print('found {} chromosomes'.format(len(genomesize)))
    bed_content = [] #pd.DataFrame(columns=['chr', 'start', 'end'])

    for chrom in genomesize:

        nbins = genomesize[chrom]//binsize + 1 if (genomesize[chrom] % binsize > 0) else 0
        starts = [int(i*binsize) for i in range(nbins)]
        ends = [min(int((i+1)*binsize), genomesize[chrom]) for i in range(nbins)]
        chr_ = [chrom] * nbins

        bed_content += [Interval(c, s, e) for c, s, e in zip(chr_, starts, ends)]
    regions = BedTool(bed_content)
    print('found {} regions'.format(len(regions)))
    if storage is not None:
        regions.moveto(storage)
    return regions



def sparse_count_reads_in_regions(bamfile, regions, storage,
                                  barcodetag, flank=0, log=None,
                                  count_both_ends=False):
    """ This function obtains the counts per bins of equal size
    across the genome.

    The function automatically extracts the genome size from the
    bam file header.
    If group tags are available, they will be used to extract
    the indices from.
    Finally, the function autmatically detects whether the bam-file
    contains paired-end or single-end reads.
    Paired-end reads are counted once at the mid-point between the two
    pairs while single-end reads are counted at the 5' end.
    For paired-end reads it is optionally possible to count both read ends
    by setting count_both_ends=True.

    Parameters
    ----------
    bamfile :  str
        Path to a bamfile. The bamfile must be indexed.
    regions : str
        BED or GFF file containing the regions of interest.
    storage : str
        Path to the output hdf5 file, which contains the counts per chromsome.
    flank : int
        Extension of the regions in base pairs. Default: 0
    template_length : int
        Assumed template length. This is used when counting paired-end reads
        at the mid-point and the individual reads do not overlap with
        the given region, but the mid-point does.
    count_both_ends : bool
        Indicates whether for paired-end sequences, the ends of both mates should
        be counted separately. Default: False.
    """

    # Obtain the header information
    afile = AlignmentFile(bamfile, 'rb')

    # extract genome size
    genomesize = {}
    for chrom, length in zip(afile.references, afile.lengths):
        genomesize[chrom] = length

    regfile = BedTool(regions)

    nreg = len(regfile)
    barcoder = Barcoder(barcodetag)

    barcodecounter = Counter()
    for aln in afile.fetch():
        bar = barcoder(aln)
        #if bar == 'dummy':
       #    continue
        barcodecounter[bar] += 1

    barcodemap = {key: i for i, key in enumerate(barcodecounter)}

    print('found {} barcodes'.format(len(barcodemap)))

    # barcode string for final table
    barcode_string = ';'.join([bar for bar in barcodemap])

    sdokmat = dok_matrix((nreg, len(barcodemap)), dtype='int32')

    template_length = 3000

    if count_both_ends:
        # if both ends are counted, template_length is irrelevant
        tlen = 0
    else:
        tlen = template_length

    for idx, iv in enumerate(regfile):

        iv.start -= flank
        iv.end += flank

        if iv.chrom not in genomesize:
            # skip over peaks/ regions from chromosomes
            # that are not contained in the bam file
            continue

        fetchstart = max(iv.start - tlen, 0)
        fetchend =  min(iv.end + tlen, genomesize[iv.chrom])

        for aln in afile.fetch(iv.chrom, fetchstart, fetchend):
            bar = barcoder(aln)
            #if bar == 'dummy':
            #    continue

            if aln.is_proper_pair and aln.is_read1 and not count_both_ends:

                pos = min(aln.reference_start, aln.next_reference_start)

                # count paired end reads at midpoint
                midpoint = pos + abs(aln.template_length)//2
                if midpoint >= iv.start and midpoint < iv.end:
                   sdokmat[idx, barcodemap[bar]] += 1

            if not aln.is_paired or count_both_ends:
                # count single-end reads at 5p end
                if not aln.is_reverse:
                    if aln.reference_start >= iv.start and aln.reference_start < iv.end:
                        sdokmat[idx, barcodemap[bar]] += 1
                else:
                    if aln.reference_start + aln.reference_length - 1 >= iv.start and \
                       aln.reference_start + aln.reference_length - 1 < iv.end:
                        sdokmat[idx, barcodemap[bar]] += 1

    afile.close()

    # store the results in COO sparse matrix format
    save_sparsematrix(storage, sdokmat, barcode_string.split(';'))

        
def save_sparsematrix(filename, mat, barcodes):
    spcoo = mat.tocoo()
    mmwrite(filename, spcoo)

    if isinstance(barcodes, pd.DataFrame):
        df = barcodes
    else:
        df = pd.DataFrame({'barcodes': barcodes})
    df.to_csv(filename + '.bct', sep='\t', header=True, index=False)

def get_count_matrix_(filename, shape, header=True, offset=0):
    """
    read count matrix in sparse format
    """
    if filename.endswith(".mtx"):
        return mmread(filename).tocsr()

    if header:
        spdf = pd.read_csv(filename, sep='\t', skiprows=1)
    else:
        spdf = pd.read_csv(filename, sep='\t', skiprows=1, header=None, names=['region','cell','count'])
    if offset > 0:
        spdf.region -= offset
        spdf.cell -= offset
    smat = csr_matrix((spdf['count'], (spdf.region, spdf.cell)),
                      shape=shape, dtype='float')
    return smat

def get_cell_annotation_first_row_(filename):
    """
    extract cell ids from the comment line in the count matrix csv
    """

    open_ = gzip.open if filename.endswith('.gz') else open
    with open_(filename, 'r') as f:
        line = f.readline()
        if hasattr(line, 'decode'):
            line = line.decode('utf-8')
        line = line.split('\n')[0]
        line = line.split(' ')[-1]
        line = line.split('\t')[-1]
        annot = line.split(';')
    return pd.DataFrame(annot, columns=['cell'])

def get_cell_annotation(filename):
    if os.path.exists(filename + '.bct'):
        return pd.read_csv(filename + '.bct', sep='\t')
    #if os.path.exists(filename + '.cannot.tsv'):
    #    return pd.read_csv(filename + '.cannot.tsv', sep='\t')
    #return get_cell_annotation_first_row_(filename)

def get_regions_from_bed_(filename):
    """
    load a bed file
    """
    regions = pd.read_csv(filename, sep='\t', names=['chrom', 'start', 'end'], usecols=[0,1,2])
    return regions


def write_cannot_table(filename, table):
    table.to_csv(filename + '.bct', sep='\t', index=False)

class CountMatrix:

    @classmethod
    def create_from_countmatrix(cls, countmatrixfile, regionannotation, header=True, index_offset=0):
        """
        constructor for loading a count matrix with associated regions
        """
            
        cannot = get_cell_annotation(countmatrixfile)
        
        cannot['cell'] = cannot[cannot.columns[0]]
        rannot = get_regions_from_bed_(regionannotation)
        shape = rannot.shape[0], len(cannot)
        cmat = get_count_matrix_(countmatrixfile, shape, header=header, offset=index_offset)
        return cls(cmat, rannot, cannot)

    def __init__(self, countmatrix, regionannotation, cellannotation):

        if not issparse(countmatrix):
            countmatrix = csr_matrix(countmatrix)

        self.cmat = countmatrix.tocsr()
        self.cannot = cellannotation
        self.regions = regionannotation
        assert self.cmat.shape[0] == len(self.regions)
        assert self.cmat.shape[1] == len(self.cannot)

    @property
    def counts(self):
        """
        count matrix property
        """
        return self.cmat

    @classmethod
    def merge(cls, cms, samplelabel=None):
        newcannot = []
        for i, cm in enumerate(cms):
            ca = cm.cannot.copy()
            if samplelabel is not None:
                ca['sample'] = samplelabel[i]
            else:
                ca['sample'] = 'sample_{}'.format(i)
            newcannot.append(ca)
        cannot = pd.concat(newcannot, axis=0)
        return cls(hstack([cm.cmat for cm in cms]), cms[0].regions, cannot)

    def filter_count_matrix(self, minreadsincells=1000, maxreadsincells=30000,
                            minreadsinpeaks=20,
                            binarize=True, maxcount=None):
        """
        Applies filtering to the count matrix
        """

        if binarize:
            self.cmat.data[self.cmat.data > 0] = 1
        if maxcount is not None:
            self.cmat.data[self.cmat.data > maxcount] = maxcount 

        cellcounts = self.cmat.sum(axis=0)
        keepcells = np.where((cellcounts>=minreadsincells) & (cellcounts<maxreadsincells) & (self.cannot.cell.values!='dummy'))[1]

        self.cmat = self.cmat[:, keepcells]
        self.cannot = self.cannot.iloc[keepcells]

        regioncounts = self.cmat.sum(axis=1)
        keepregions = np.where(regioncounts>=minreadsinpeaks)[0]

        self.cmat = self.cmat[keepregions, :]
        self.regions = self.regions.iloc[keepregions]

    def pseudobulk(self, cell, group):

        grouplabels = list(set(group))

        cnts = np.zeros((self.n_regions, len(grouplabels)))

        for i, glab in enumerate(grouplabels):
            ids = self.cannot.cell.isin(cell[group == glab])
            ids = np.arange(self.cannot.shape[0])[ids]
            cnts[:, i:(i+1)] = self.cmat[:, ids].sum(1)

        cannot = pd.DataFrame(grouplabels, columns=['cell'])
        return CountMatrix(csr_matrix(cnts), self.regions, cannot)
        
    def __call__(self, icell=None):
        if icell is None:
            return self.cmat.toarray()
        elif isinstance(icell, int):
            return self.cmat[:, icell].toarray()
        elif isinstance(icell, slice):
            return self.cmat[:, icell].toarray()
        raise ValueError("indexing not supported")

    def __getitem__(self, ireg):
        return self.cmat[ireg]

    def __repr__(self):
        return "{} x {} CountMatrix with {} entries".format(self.cmat.shape[0], self.cmat.shape[1], self.cmat.nnz)

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

    def subset(self, indices):
        return CountMatrix(self.cmat[:, indices], copy.copy(self.regions),
                    self.cannot.iloc[indices])

    def export_regions(self, filename):
        """
        Exports the associated regions to a bed file.
        """
        self.regions.to_csv(filename,
                            columns=['chrom', 'start', 'end'],
                            sep='\t', index=False, header=False)

    def export_counts(self, filename):
        """
        Exports the countmatrix in sparse format to a csv file
        """

        save_sparsematrix(filename, self.cmat, self.cannot)
        #mmwrite(filename, self.cmat.tocoo())

        #df = pd.DataFrame({'barcodes': self.cannot.cell.values})
        #df.to_csv(filename + '.bct', sep='\t', header=True, index=False)

#        spcoo = self.cmat.tocoo()
#        # sort lexicographically
#       
#        order_ = np.lexsort((spcoo.col, spcoo.row))
#        indices = np.asarray([x for x in zip(spcoo.row, spcoo.col)], dtype=np.int64)[order_]
#        values = spcoo.data.astype(np.float32)[order_]
#        cont = {'region': indices[:,0], 'cell': indices[:, 1], 'count': values}
#       
#        df = pd.DataFrame(cont)
#        with open(filename, 'w') as title:
#            title.write('#  ' + ';'.join(self.cannot.cell.values) + '\n')
#       
#        df.to_csv(filename, mode = 'a', sep='\t',
#                  header=True, index=False,
#                  columns=['region', 'cell', 'count'])
#
#        write_cannot_table(filename, self.cannot)
