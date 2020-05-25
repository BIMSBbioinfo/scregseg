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

from scseg.bam_utils import Barcoder

def make_counting_bins(bamfile, binsize, storage=None):
    """ Genome intervals for binsize.

    For a given bam-file and binsize,
    this function creates a bed-file containing all intervals.
    The genome size is extracted from the bam header.

    Parameters
    ----------
    bamfile : str
       Path to bamfile
    binsize : int
       Bin size
    storage : path or None
       Output path of the BED file.

    Returns
    -------
    BedTool object:
       Output BED file is returned as BedTool object.
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



def sparse_count_reads_in_regions(bamfile, regions,
                                  barcodetag, flank=0, log=None, mapq=30,
                                  mode='midpoint', only_with_barcode=True):
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
    by setting mode='both' or mode='eitherend'.

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
    mapq : int
        Minimum mapping quality
    mode : str
        For paired-end sequences reads can be counted at the midpoint,
        by counting both ends (like they came from single-ended sequencing)
        or by counting if either 5'-end is in the bin.
        These options are indicated by mode=['midpoint', 'countboth', 'eitherend'].
        Default: mode='midpoint'
    only_with_barcode : bool
        This indicates that reads without barcodes should be skipped.
        Use False for bulk or pseudobulk aggregation.
        Default: True.
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
        if only_with_barcode and bar == 'dummy':
            continue
        barcodecounter[bar] += 1

    barcodemap = {key: i for i, key in enumerate(barcodecounter)}

    print('found {} barcodes'.format(len(barcodemap)))

    # barcode string for final table
    barcode_string = ';'.join([bar for bar in barcodemap])

    sdokmat = dok_matrix((nreg, len(barcodemap)), dtype='int32')

    template_length = 3000

    if mode != 'midpoint':
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
            if only_with_barcode and bar == 'dummy':
                continue
            if aln.mapping_quality < mapq:
                continue

            if aln.is_proper_pair and aln.is_read1 and mode == 'midpoint':

                pos = min(aln.reference_start, aln.next_reference_start)

                # count paired end reads at midpoint
                midpoint = pos + abs(aln.template_length)//2
                if midpoint >= iv.start and midpoint < iv.end:
                   sdokmat[idx, barcodemap[bar]] += 1

            if aln.is_proper_pair and mode == 'eitherend':

                minpos = min(aln.reference_start + aln.template_length, aln.reference_start)
                maxpos = max(aln.reference_start + aln.template_length, aln.reference_start)
                #minpos = min(aln.reference_start, aln.next_reference_start)
                #maxpos = max(aln.reference_start, aln.next_reference_start)

                if minpos >= iv.start and minpos < iv.end and maxpos >= iv.start and maxpos < iv.end and aln.is_read2:
                    pass
                    #sdokmat[idx, barcodemap[bar]] += 1
                else:
                    sdokmat[idx, barcodemap[bar]] += 1
                   

            if not aln.is_paired or mode == 'countboth':
                # count single-end reads at 5p end
                if not aln.is_reverse:
                    if aln.reference_start >= iv.start and aln.reference_start < iv.end:
                        sdokmat[idx, barcodemap[bar]] += 1
                else:
                    if aln.reference_start + aln.reference_length - 1 >= iv.start and \
                       aln.reference_start + aln.reference_length - 1 < iv.end:
                        sdokmat[idx, barcodemap[bar]] += 1

    afile.close()

    return sdokmat.tocsr(), pd.DataFrame({'cell': barcode_string.split(';')})
    # store the results in COO sparse matrix format
    #save_sparsematrix(storage, sdokmat, barcode_string.split(';'))

        
def save_sparsematrix(filename, mat, barcodes):
    """ Save sparse count matrix and annotation 

    Parameters
    ----------
    filename : str
        Filename of the matrix market output file.
        The associated cell annotation is stored with the additional prefix '.bct'.
    mat : sparse matrix
        Matrix to store.
    barcodes: list(str) or pandas.DataFrame
        Cell annotation to store in the '.bct' file.

    """
    spcoo = mat.tocoo()
    mmwrite(filename, spcoo)

    if isinstance(barcodes, pd.DataFrame):
        df = barcodes
    else:
        df = pd.DataFrame({'cell': barcodes})
    df.to_csv(filename + '.bct', sep='\t', header=True, index=False)

def get_count_matrix_(filename):
    """ Read count matrix in sparse format

    This function also loads the associated cell/barcode information from
    the .bct file.

    Parameters
    ----------
    filename : str
       Path to input matrix in matrix market format.
    shape : tuple(int)
       (Obsolete parameter) Target shape. Was used in an earlier version, before matrix market format was supported.
    header : bool
       (Obsolete parameter) header information
    offset : int
       (Obsolete parameter) offset
    """
    if filename.endswith(".mtx"):
        return mmread(filename).tocsr()
#
#    if header:
#        spdf = pd.read_csv(filename, sep='\t', skiprows=1)
#    else:
#        spdf = pd.read_csv(filename, sep='\t', skiprows=1, header=None, names=['region','cell','count'])
#    if offset > 0:
#        spdf.region -= offset
#        spdf.cell -= offset
#    smat = csr_matrix((spdf['count'], (spdf.region, spdf.cell)),
#                      shape=shape, dtype='float')
#    return smat

#def get_cell_annotation_first_row_(filename):
#    """
#    extract cell ids from the comment line in the count matrix csv
#    """
#
#    open_ = gzip.open if filename.endswith('.gz') else open
#    with open_(filename, 'r') as f:
#        line = f.readline()
#        if hasattr(line, 'decode'):
#            line = line.decode('utf-8')
#        line = line.split('\n')[0]
#        line = line.split(' ')[-1]
#        line = line.split('\t')[-1]
#        annot = line.split(';')
#    return pd.DataFrame(annot, columns=['cell'])

def get_cell_annotation(filename):
    """ Load Cell/barcode information from '.bct' file 

    Parameter
    ---------
    filename : str
       Filename prefix (without the .bct file ending)
    """
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
    def create_from_countmatrix(cls, countmatrixfile, regionannotation):
        """ Load Countmatrix from matrix market format file.

        Parameters
        ----------
        countmatrixfile : str
            Matrix market file
        regionannotation : str
            Region anntation in bed format

        Returns
        -------
        CountMatrix object
        """
        cannot = get_cell_annotation(countmatrixfile)
        
        if 'cell' not in cannot.columns:
            cannot['cell'] = cannot[cannot.columns[0]]
        rannot = get_regions_from_bed_(regionannotation)
        cmat = get_count_matrix_(countmatrixfile)
        return cls(cmat, rannot, cannot)

    @classmethod
    def create_from_bam(cls, bamfile, regions, barcodetag='CB', mode='eitherend', mapq=30, no_barcode=False):
        """ Creates a countmatrix from a given bam file and pre-specified target regions.

        Parameters
        ----------
        bamfile : str
            Path to the input bam file.
        regions : str
            Path to the input bed files with the target regions.
        barcodetag : str or callable
            Barcode tag or callable for extracting the barcode from the alignment.
            Default: 'CB'
        mode : str
            Specifies the counting mode for paired end data.
            'bothends' counts each 5' end, 'midpoint' counts the fragment once at the midpoint
            and 'eitherend' counts once if either end is present in the interval, but if 
            both ends are inside of the interval, it is counted only once to mitigate double counting.
            Default: 'eitherend'
        mapq : int
            Only consider reads with a minimum mapping quality. Default: 30
        no_barcode : bool
            Whether the file contains barcodes or whether it contains a bulk sample. Default: False.

        Returns
        -------
        CountMatrix object

        """
        #if not os.path.exists(regions):
        #    make_counting_bins(bamfile, binsize, regions)
        rannot = get_regions_from_bed_(regions)
        cmat, cannot = sparse_count_reads_in_regions(bamfile, regions,
                                  barcodetag, flank=0, log=None,
                                  mapq=mapq,
                                  mode=mode,
                                  only_with_barcode=not no_barcode)
        
        return cls(cmat.tocsr(), rannot, cannot)
        

    def __init__(self, countmatrix, regionannotation, cellannotation):

        if not issparse(countmatrix):
            countmatrix = csr_matrix(countmatrix)

        self.cmat = countmatrix.tocsr()
        self.cannot = cellannotation
        self.regions = regionannotation
        assert self.cmat.shape[0] == len(self.regions)
        assert self.cmat.shape[1] == len(self.cannot)

    def remove_chroms(self, chroms):
        """Remove chromsomes."""
        idx = self.regions.chrom[~self.regions.chrom.isin(chroms)].index
        self.regions = self.regions[~self.regions.chrom.isin(chroms)]
        self.cmat = self.cmat[idx]

    @property
    def counts(self):
        """
        count matrix property
        """
        return self.cmat

    @classmethod
    def merge(cls, cms, samplelabel=None):
        """ Merge several countmatices.

        Matrices must have the same row dimensionality

        Parameters
        ----------
        cms : list(CountMatrix objects)
            List of count matrices
        samplelabel : list(str) or None
            Associated sample labels. If None, a default sample name is used 'sample_x'.
        
        Returns
        -------
        CountMatrix object
        """
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
                            binarize=True, trimcount=None):
        """
        Applies quality filtering to the count matrix.

        Parameters
        ----------
        minreadsincells : int
            Minimum counts in cells to remove poor quality cells with too few reads.
            Default: 1000
        maxreadsincells : int
            Maximum counts in cells to remove poor quality cells with too many reads.
            Default: 30000
        minreadsinpeaks : int
            Minimum counts in region to remove low coverage regions.
            Default: 20
        binarize : bool
            Whether to binarize the count matrix. Default: True
        trimcounts : int or None
            Whether to trim the maximum number of reads per cell and region.
            This is a generalization to the binarize option.
            Default: None (No trimming performed)

        """

        if binarize:
            self.cmat.data[self.cmat.data > 0] = 1

        if trimcount is not None and trimcount > 0:
            self.cmat.data[self.cmat.data > trimcount] = trimcount 

        cellcounts = self.cmat.sum(axis=0)
        keepcells = np.where((cellcounts>=minreadsincells) & (cellcounts<maxreadsincells) & (self.cannot.cell.values!='dummy'))[1]

        self.cmat = self.cmat[:, keepcells]
        self.cannot = self.cannot.iloc[keepcells]

        regioncounts = self.cmat.sum(axis=1)
        keepregions = np.where(regioncounts>=minreadsinpeaks)[0]

        self.cmat = self.cmat[keepregions, :]
        self.regions = self.regions.iloc[keepregions]

    def pseudobulk(self, cell, group):
        """ Compute pseudobulk counts.

        Given a matchin list of cells and a list of group association (of the same length)
        The pseudobulk is computed across cells in each group.

        Parameters
        ----------
        cell : list of cells
            List of cell names. These must match with the cell names in the countmatrix
        group : list of groups
            List of group names. Defines which cells correspond to which group.

        Returns
        -------
        CountMatrix object
        """
        grouplabels = list(set(group))

        cnts = np.zeros((self.n_regions, len(grouplabels)))

        for i, glab in enumerate(grouplabels):
            ids = self.cannot.cell.isin(cell[group == glab])
            ids = np.arange(self.cannot.shape[0])[ids]
            cnts[:, i:(i+1)] = self.cmat[:, ids].sum(1)

        cannot = pd.DataFrame(grouplabels, columns=['cell'])
        return CountMatrix(csr_matrix(cnts), self.regions, cannot)
        
    def subset(self, cell):
        """ Subset countmatrix

        Returns a new count matrix containing only the given cell names.

        Parameters
        ----------
        cell : list(str)
            List of cell names

        Returns
        -------
        CountMatrix object
        """
        ids = self.cannot.cell.isin(cell)
        ids = np.arange(self.cannot.shape[0])[ids]

        cannot = self.cannot[self.cannot.cell.isin(cell)]

        #ids = np.asarray(cannot.cell.isin(cell).index.tolist())
        #cannot = cannot[cannot.cell.isin(cell)]

        cmat = self.cmat.tocsc()
        cnts = cmat[:, ids]

        return CountMatrix(csr_matrix(cnts), self.regions, cannot)
        
#    def __call__(self, icell=None):
#        if icell is None:
#            return self.cmat.toarray()
#        elif isinstance(icell, int):
#            return self.cmat[:, icell].toarray()
#        elif isinstance(icell, slice):
#            return self.cmat[:, icell].toarray()
#        raise ValueError("indexing not supported")
#
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

#    def subset(self, indices):
#        return CountMatrix(self.cmat[:, indices], copy.copy(self.regions),
#                    self.cannot.iloc[indices])
#
    def export_regions(self, filename):
        """
        Exports the associated regions to a bed file.

        Parameters
        ----------
        filename : str
            Output bed file.
        """
        self.regions.to_csv(filename,
                            columns=['chrom', 'start', 'end'],
                            sep='\t', index=False, header=False)

    def export_counts(self, filename):
        """
        Exports the countmatrix in matrix market format

        Parameters
        ----------
        filename : str
            Output file name.
        """

        save_sparsematrix(filename, self.cmat, self.cannot)
