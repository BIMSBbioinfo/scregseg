import logging
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
from collections import Counter, OrderedDict
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from anndata import AnnData
from anndata import read_h5ad
import anndata as ad

from scregseg.bam_utils import Barcoder
from scregseg.bam_utils import fragmentlength_in_regions
from scregseg.bam_utils import fragmentlength_from_bam, fragmentlength_from_bed

def load_count_matrices(countfiles, bedfile, mincounts,
                        maxcounts, trimcounts, minregioncounts):
    """ Load count matrices.

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
    data = []
    for cnt in countfiles:
        cm = CountMatrix.load(cnt, bedfile)
        cm = cm.filter(mincounts, maxcounts,
                  binarize=False, trimcount=trimcounts)

        data.append(cm)

    if minregioncounts is None:
        minregioncounts = 0

    # make sure the same regions are used afterwards
    if minregioncounts > 0:
        regioncounts = np.zeros((data[0].shape[0], ))
        for datum in data:
            regioncounts += np.asarray(datum.cmat.sum(axis=1)).flatten()
        maxregioncounts = regioncounts.max()
        logging.debug('removing region with x < {} and x >= {}'.format(minregioncounts, maxregioncounts))
        for i, _ in enumerate(data):
            keepregions = np.where((regioncounts >= minregioncounts) & (regioncounts <= maxregioncounts))[0]

            data[i].cmat = data[i].cmat[keepregions, :]
            data[i].regions = data[i].regions.iloc[keepregions]
    return data


def get_genome_size_from_bam(file):
    """ Extract chromosome sizes from a bam-file.

    Parameters
    ----------
    file : str
       bam-file

    Returns
    -------
    dict
        Dict with keys and values corresponding to chromosome names and lengths, respectively.
    """
    afile = AlignmentFile(file, 'rb')

    # extract genome size

    genomesize = {}
    for chrom, length in zip(afile.references, afile.lengths):
        genomesize[chrom] = length
    afile.close()
    return genomesize


def get_genome_size_from_tsv(file):
    """ Extract chromosome sizes from a bed-file.

    Parameters
    ----------
    file : str
       bed-file or table

    Returns
    -------
    dict
        Dict with keys and values corresponding to chromosome names and lengths, respectively.
    """
    df = BedTool(file).to_dataframe()
    chroms = df.chrom.unique()
    genomesize = {}
    for chrom in chroms:
        genomesize[chrom] = df[df.chrom==chrom].end.max()
    return genomesize


def get_genome_size(file):
    """ Extract chromosome sizes from a bed-file or bam-files.

    Parameters
    ----------
    file : str
       bam-file, bed-file or table

    Returns
    -------
    dict
        Dict with keys and values corresponding to chromosome names and lengths, respectively.
    """
    if file.endswith('.bam'):
        gs = get_genome_size_from_bam(file)
    elif file.endswith('.tsv.gz') or file.endswith('.tsv'):
        gs = get_genome_size_from_tsv(file)
    else:
        raise ValueError(f'Unknown file type for: {file}')
    return gs


def make_counting_bins(file, binsize, storage=None, remove_chroms=[]):
    """ Genome intervals for binsize.

    For a given bam-file and binsize,
    this function creates a bed-file containing all intervals.
    The genome size is extracted from the bam header.

    Parameters
    ----------
    file : str
       Path to bamfile or a fragments.tsv file
    binsize : int
       Bin size
    storage : path or None
       Output path of the BED file.
    remove_chroms : list
       List of chromosomes to remove. Default=[]

    Returns
    -------
    BedTool object:
       Output BED file is returned as BedTool object.
    """
    genomesize = get_genome_size(file)
    bed_content = [] #pd.DataFrame(columns=['chr', 'start', 'end'])

    for chrom in genomesize:
        ignore_chr=False
        for rmchr in remove_chroms:
            if rmchr in chrom:
                ignore_chr=True
        if ignore_chr:
            continue

        nbins = genomesize[chrom]//binsize + 1 if (genomesize[chrom] % binsize > 0) else 0
        starts = [int(i*binsize) for i in range(nbins)]
        ends = [min(int((i+1)*binsize), genomesize[chrom]) for i in range(nbins)]
        chr_ = [chrom] * nbins

        bed_content += [Interval(c, s, e) for c, s, e in zip(chr_, starts, ends)]
    regions = BedTool(bed_content)
    if storage is not None:
        regions.moveto(storage)
    return regions


def sparse_count_fragments_in_regions(fragmentfile, regions):
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

    Returns
    -------
        Sparse matrix and cell annotation as pd.DataFrame
    """

    regionbed = BedTool([Interval(chrom=iv.chrom, start=iv.start, end=iv.end, name=str(i)) for i, iv in enumerate(BedTool(regions))])

    fragments = BedTool(fragmentfile)

    mapping = Counter()
    for i, frag  in enumerate(fragments):
        mapping[frag.name] = 1

    mapping = {k:i for i, k in enumerate(mapping)}

    counts = regionbed.intersect(fragments, wo=True)

    rows = np.asarray([[int(c.fields[3]), mapping[c.fields[-3]]] for c in counts])

    smat = coo_matrix((np.ones(rows.shape[0]), (rows[:,0], rows[:,1])),
                      shape=(len(regionbed), len(mapping)),
                      dtype='int32')

    barcodes = pd.DataFrame({'barcode': [k for k in mapping]})
    barcodes.set_index('barcode', inplace=True)
    return smat.tocsr(), barcodes

def sparse_count_reads_in_regions(bamfile, regions,
                                  barcodetag, flank=0, log=None, mapq=30,
                                  mode='midpoint', only_with_barcode=True,
                                  maxfraglen=2000):
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
    maxfraglen : int
        Maximum fragment length to consider. Default: 2000 [bp]

    Returns
    -------
        Sparse matrix and cell annotation as pd.DataFrame
    """

    warnings.warn('sparse_count_reads_in_regions deprecated. Please use sparse_count_reads_in_regions2',
                  category=DeprecationWarning)
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
            if abs(aln.template_length) > maxfraglen:
                continue
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

                if minpos >= iv.start and minpos < iv.end and maxpos >= iv.start and maxpos < iv.end and aln.is_read2:
                    pass
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

    barcodes = pd.DataFrame({'barcode': barcode_string.split(';')})
    barcodes.set_index('barcode', inplace=True)
    return sdokmat.tocsr(), barcodes


def sparse_count_reads_in_regions2(bamfile, regions,
                                  barcodetag, flank=0, log=None, mapq=30,
                                  mode='midpoint', only_with_barcode=True,
                                  maxfraglen=2000):
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
    maxfraglen : int
        Maximum fragment length to consider. Default: 2000 [bp]

    Returns
    -------
        Sparse matrix and cell annotation as pd.DataFrame
    """
    # Obtain the header information
    afile = AlignmentFile(bamfile, 'rb')

    # extract genome size

    regfile = BedTool(regions).to_dataframe()
    nreg = regfile.shape[0]
    regfile.name = [str(x) for x in range(nreg)]
    regfile = BedTool.from_dataframe(regfile)

    print(f'found {nreg} regions')
    barcoder = Barcoder(barcodetag)

    barcodemap = OrderedDict()

    chroms = []
    starts =  []
    ends = []
    barcodes = []
    for i, aln in enumerate(afile.fetch()):
        bar = barcoder(aln)
        if bar not in barcodemap:
            barcodemap[bar] = len(barcodemap)
        if abs(aln.template_length) > maxfraglen:
            continue
        if only_with_barcode and bar == 'dummy':
            continue
        if aln.mapping_quality < mapq:
            continue

        if aln.is_proper_pair and aln.is_read1 and mode == 'midpoint':

            chroms.append(aln.reference_name)
            barcodes.append(barcodemap[bar])

            pos = min(aln.reference_start, aln.next_reference_start)

            # count paired end reads at midpoint
            midpoint = pos + abs(aln.template_length)//2
            pos = midpoint
            starts.append(pos)

        if not aln.is_paired or mode == 'countboth':
            # count single-end reads at 5p end
            if not aln.is_reverse:
                pos = aln.reference_start
            else:
                pos = aln.reference_start + aln.inferred_length - 1
            chroms.append(aln.reference_name)
            barcodes.append(barcodemap[bar])
            starts.append(pos)

    afile.close()
    print(f'parsed {i} reads from bam')
    print(f'found {len(barcodemap)} barcodes')

    df = pd.DataFrame({'chrom':chroms, 'start':starts})
    df.loc[:,'end'] = df.start+1
    df.loc[:,'name'] = barcodes

    reads=BedTool.from_dataframe(df)
    
    nfreg = len(regfile[0].fields)
    counts = regfile.intersect(reads, wo=True)

    rows = np.asarray([[int(c.fields[3]), int(c.fields[nfreg+3])] for c in counts])

    smat = coo_matrix((np.ones(rows.shape[0]), (rows[:,0], rows[:,1])),
                      shape=(len(regfile), len(barcodemap)),
                      dtype='int32')

    print(f'made countmatrix {smat.shape}')
    barcode_string = ';'.join([bar for bar in barcodemap])

    barcodes = pd.DataFrame({'barcode': barcode_string.split(';')})
    barcodes.set_index('barcode', inplace=True)
    return smat.tocsr(), barcodes


def sparse_count_reads_in_regions_fast(bamfile, regions,
                                  barcodetag, flank=0, log=None, mapq=30,
                                  mode='midpoint', only_with_barcode=True,
                                  maxfraglen=2000):
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
    maxfraglen : int
        Maximum fragment length to consider. Default: 2000 [bp]

    Returns
    -------
        Sparse matrix and cell annotation as pd.DataFrame
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


    temp_smats = {}
    for chrom, length in zip(afile.references, afile.lengths):
        temp_smats[chrom] = dok_matrix((length, len(barcodemap)), dtype='int32')

    # barcode string for final table
    barcode_string = ';'.join([bar for bar in barcodemap])

    for aln in afile.fetch():
        bar = barcoder(aln)
        if abs(aln.template_length) > maxfraglen:
            continue
        if only_with_barcode and bar == 'dummy':
            continue
        if aln.mapping_quality < mapq:
            continue

        if aln.is_proper_pair and aln.is_read1 and mode == 'midpoint':

            pos = min(aln.reference_start, aln.next_reference_start)

            # count paired end reads at midpoint
            midpoint = pos + abs(aln.template_length)//2
            temp_smats[aln.reference_name][midpoint, barcodemap[bar]] += 1

        if not aln.is_paired or mode == 'countboth':
            # count single-end reads at 5p end
            if not aln.is_reverse:
                temp_smats[aln.reference_name][aln.reference_start, barcodemap[bar]] += 1
            else:
                temp_smats[aln.reference_name][aln.reference_start + aln.reference_length - 1, barcodemap[bar]] += 1

    for k in temp_smats:
        temp_smats[k] = temp_smats[k].tocsr()
    afile.close()

    sdokmat = lil_matrix((nreg, len(barcodemap)), dtype='int32')
    for idx, iv in enumerate(regfile):

        iv.start -= flank
        iv.end += flank

        if iv.chrom not in genomesize:
            # skip over peaks/ regions from chromosomes
            # that are not contained in the bam file
            continue
        sdokmat[idx,:] += temp_smats[iv.chrom][iv.start:iv.end, :].sum(0)


    barcodes = pd.DataFrame({'barcode': barcode_string.split(';')})
    barcodes.set_index('barcode', inplace=True)
    return sdokmat.tocsr(), barcodes

def save_cellannotation(filename, barcodes):
    """ Save cell annotation

    Parameters
    ----------
    filename : str
        Filename of the matrix market output file.
        The associated cell annotation is stored with the additional prefix '.bct'.
    barcodes: list(str) or pandas.DataFrame
        Cell annotation to store in the '.bct' file.

    Returns
    -------
        None
    """
    if isinstance(barcodes, pd.DataFrame):
        df = barcodes
    else:
        df = pd.DataFrame({'barcode': barcodes})
        df.set_index('barcode', inplace=True)
    df.to_csv(filename + '.bct', sep='\t', header=True, index=True)


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

    Returns
    -------
        None
    """
    spcoo = mat.tocoo()
    mmwrite(filename, spcoo)
    save_cellannotation(filename,  barcodes)


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

    Returns
    -------
        Sparse matrix in CSR format
    """
    if filename.endswith(".mtx"):
        return mmread(filename).tocsr()
    if filename.endswith('.npz'):
        files = np.load(filename)
        return csr_matrix((files['arr_0'], files['arr_1'], files['arr_2']))
    else:
        raise ValueError('unknown file format. Counts must be in mtx for npz format')

def get_cell_annotation(filename, suffix='.bct'):
    """ Load Cell/barcode information from '.bct' file

    Parameter
    ---------
    filename : str
       Filename prefix (without the .bct file ending)

    Returns
    -------
        Cell annotation as pd.DataFrame
    """
    df = pd.read_csv(filename + suffix, sep='\t')
    if 'barcode' not in df.columns:
        df.rename(columns={df.columns[0]: 'barcode'}, inplace=True)
    df.set_index('barcode', inplace=True)
    return df

def get_regions_from_bed_(filename):
    """
    load a bed file

    Parameter
    ---------
    filename : str
       BED file

    Returns
    -------
        Region annotation from bed file as pd.DataFrame
    """
    regions = pd.read_csv(filename, sep='\t',
                          names=['chrom', 'start', 'end'],
                          usecols=[0,1,2])
    regions.loc[:, 'name'] = regions.apply(lambda row: f'{row.chrom}_{row.start}_{row.end}', axis=1)
    regions.set_index('name', inplace=True)
    return regions


class CountMatrix:

    @classmethod
    def load(clt, countfile, *args, **kwargs):
        """ Load Countmatrix from a h5ad or mtx file.

        Parameters
        ----------
        countmatrixfile : str
            h5ad or mtx file
        regionannotation : str or None
            Associated regions. Only required for mtx file.
        cellannotation : str or None
            Associated barcode annotation. Optionally used for mtx file only.

        Returns
        -------
        CountMatrix object
        """
        if countfile.endswith('.h5ad'):
            return clt.from_h5ad(countfile)
        else:
            return clt.from_mtx(countfile, *args,**kwargs)

    @classmethod
    def from_h5ad(cls, countmatrixfile):
        """ Load Countmatrix from h5ad file.

        Parameters
        ----------
        countmatrixfile : str
            anndata storage

        Returns
        -------
        CountMatrix object
        """
        adata = read_h5ad(countmatrixfile)
        cl = cls(adata.X, adata.obs, adata.var, adata.uns, adata.obsm, adata.varm)
        return cl

    @classmethod
    def from_mtx(cls, countmatrixfile,
                 regionannotation=None, cellannotation=None):
        """ Load Countmatrix from matrix market format file.

        Parameters
        ----------
        countmatrixfile : str
            Matrix market file
        regionannotation : str or None
            Region anntation in bed format
        cellannotation : str or None
            Cell anntation in tsv format

        Returns
        -------
        CountMatrix object
        """
        if cellannotation is None:
            # try to infer cell annotation file
            cannot = get_cell_annotation(countmatrixfile)
        else:
            cannot = get_cell_annotation(cellannotation, suffix='')

        if regionannotation is None:
            # try to infer region annotation file
            rannot = get_regions_from_bed_(countmatrixfile + '.bed')
        else:
            rannot = get_regions_from_bed_(regionannotation)

        cmat = get_count_matrix_(countmatrixfile)
        return cls(cmat, rannot, cannot)

    @classmethod
    def create_from_countmatrix(cls, countmatrixfile,
                                regionannotation=None, cellannotation=None):
        """ Load Countmatrix from matrix market format file.

        Parameters
        ----------
        countmatrixfile : str
            Matrix market file
        regionannotation : str or None
            Region anntation in bed format
        cellannotation : str or None
            Cell anntation in tsv format

        Returns
        -------
        CountMatrix object
        """
        warnings.warn('create_from_countmatrix deprecated. Please use from_mtx',
                      category=DeprecationWarning)
        return cls.from_mtx(countmatrixfile,
                            regionannotation=regionannotation,
                            cellannotation=cellannotation)

    @classmethod
    def from_bam(cls, bamfile, regions, barcodetag='CB',
                        mode='eitherend', mapq=30, no_barcode=False,
                        maxfraglen=2000, with_fraglen=False):
        """ Creates a countmatrix from a given bam file and pre-specified target regions.

        Parameters
        ----------
        bamfile : str
            Path to the input bam file.
        regions : str
            Path to the input bed files with the target regions.
        barcodetag : str or callable
            Barcode tag or callable for extracting the barcode
            from the alignment.
            Default: 'CB'
        mode : str
            Specifies the counting mode for paired end data.
            'bothends' counts each 5' end,
            'midpoint' counts the fragment once at the midpoint
            and 'eitherend' counts once if either end is
            present in the interval, but if
            both ends are inside of the interval,
            it is counted only once to mitigate double counting.
            Default: 'eitherend'
        mapq : int
            Only consider reads with a minimum mapping quality. Default: 30
        no_barcode : bool
            Whether the file contains barcodes or whether it
            contains a bulk sample. Default: False.
        maxfraglen : int
            Maximum fragment length to consider. Default: 2000 [bp]
        with_fraglen : bool
            Load fragment lengths in addtion. Default=False.
        Returns
        -------
        CountMatrix object

        """
        rannot = get_regions_from_bed_(regions)
        cmat, cannot = sparse_count_reads_in_regions2(bamfile, regions,
                                                     barcodetag,
                                                     flank=0, log=None,
                                                     mapq=mapq,
                                                     mode=mode,
                                                     only_with_barcode=not no_barcode,
                                                     maxfraglen=maxfraglen)

        obj = cls(cmat.tocsr(), rannot, cannot)
        if with_fraglen:
            fragments = fragmentlength_from_bam(bamfile, regions, mapq, maxfraglen)
            obj.adata.obsm['frag_lens'] = fragments.tocsr()
        return obj

    @classmethod
    def from_fragments(cls, fragmentfile, regions, with_fraglen=False):
        """ Creates a countmatrix from a given fragments file (output by cellranger) and pre-specified target regions.

        Parameters
        ----------
        fragmentfile : str
            Path to the input bed files with the target regions.
        regions : str
            BED file containing features for which to obtain the counts
        with_fraglen : bool
            Load fragment lengths in addtion. Default=False.

        Returns
        -------
        CountMatrix object

        """
        rannot = get_regions_from_bed_(regions)
        cmat, cannot = sparse_count_fragments_in_regions(fragmentfile, regions)
        obj = cls(csr_matrix(cmat), rannot, cannot)

        if with_fraglen:
            fragments = fragmentlength_from_bed(fragmentfile, regions, 2000)
            obj.adata.obsm['frag_lens'] = fragments.tocsr()
        return obj

    @classmethod
    def create_from_fragments(cls, fragmentfile, regions, with_fraglen=False):
        return cls.from_fragments(fragmentfile, regions, with_fraglen)

    @classmethod
    def create_from_bam(cls, bamfile, regions, barcodetag='CB',
                        mode='eitherend', mapq=30, no_barcode=False,
                        maxfraglen=2000):
        """ Creates a countmatrix from a given bam file and pre-specified target regions.

        Parameters
        ----------
        bamfile : str
            Path to the input bed files with the target regions.
        barcodetag : str or callable
            Barcode tag or callable for extracting the barcode
            from the alignment.
            Default: 'CB'
        mode : str
            Specifies the counting mode for paired end data.
            'bothends' counts each 5' end,
            'midpoint' counts the fragment once at the midpoint
            and 'eitherend' counts once if either end is
            present in the interval, but if
            both ends are inside of the interval,
            it is counted only once to mitigate double counting.
            Default: 'eitherend'
        mapq : int
            Only consider reads with a minimum mapping quality. Default: 30
        no_barcode : bool
            Whether the file contains barcodes or whether it
            contains a bulk sample. Default: False.
        maxfraglen : int
            Maximum fragment length to consider. Default: 2000 [bp]

        Returns
        -------
        CountMatrix object

        """
        warnings.warn('create_from_bam deprecated. Please use from_bam',
                      category=DeprecationWarning)
        return cls.from_bam(bamfile, regions, barcodetag,
                            mode, mapq, no_barcode, maxfraglen)

    @classmethod
    def create_from_fragmentsize(cls, bamfile, regions, mapq=30,
                                 maxlen=1000, resolution=50):
        """ Creates a countmatrix from a given bam file and pre-specified target regions.

        Parameters
        ----------
        bamfile : str
            Path to the input bam file.
        regions : str
            Path to the input bed files with the target regions.
        mapq : int
            Only consider reads with a minimum mapping quality. Default: 30
        maxlen : int
            Maximum fragment length to consider. Default: 1000 bp
        resolution : int
            Resolution in base-pairs to construct the matrix

        Returns
        -------
        CountMatrix object

        """
        warnings.warn('create_from_fragmentsize deprecated. Fragmentsize is determined automatically and stored along with the count matrix',
                      category=DeprecationWarning)
        rannot = get_regions_from_bed_(regions)
        cmat, cannot = fragmentlength_in_regions(bamfile, regions,
                                                 mapq=mapq, maxlen=maxlen,
                                                 resolution=resolution)
        return cls(csr_matrix(cmat), rannot, cannot)


    def __init__(self, countmatrix, regionannotation, cellannotation, uns=None, obsm=None, varm=None):

        if not issparse(countmatrix):
            countmatrix = csr_matrix(countmatrix)

        self.adata = AnnData(countmatrix.tocsr(), regionannotation, cellannotation, uns, obsm, varm)

    @property
    def cmat(self):
        return self.adata.X

    @property
    def cannot(self):
        return self.adata.var

    @property
    def regions(self):
        return self.adata.obs

    def remove_chroms(self, chroms):
        """Remove chromsomes."""
        self.adata = self.adata[~self.adata.obs.chrom.isin(chroms),:]
        return self

    @property
    def counts(self):
        """
        count matrix property
        """
        return self.cmat

    @classmethod
    def merge(cls, cms, samplenames=None):
        """ Merge several countmatices.

        Matrices must have the same row dimensionality

        Parameters
        ----------
        cms : list(CountMatrix objects)
            List of count matrices
        samplenames : list(str) or None
            Associated sample labels. If None, a default sample name is used 'sample_x'.

        Returns
        -------
        CountMatrix object
        """

        if samplenames is None:
            samplenames = [f'sample_{i}' for i, _ in enumerate(cms)]

        for i, cm in enumerate(cms):
            if 'sample' not in cm.adata.var.columns:
                if samplenames is not None:
                    cm.adata.var.loc[:,'sample'] = samplenames[i]

        adata = ad.concat([cm.adata for cm in cms], axis=1)
        adata.obs = cms[0].adata.obs

        for i, cm in enumerate(cms):
            for k in dict(cm.adata.obsm):
                adata.obsm[f'{k}_{samplenames[i]}'] = cm.adata.obsm[k]

        return cls(adata.X, adata.obs, adata.var, adata.uns, adata.obsm, adata.varm)

    def filter(self, minreadsincell=None, maxreadsincell=None,
                            minreadsinregion=None, maxreadsinregion=None,
                            binarize=True, trimcount=None):
        """
        Applies in-place quality filtering to the count matrix.

        Parameters
        ----------
        minreadsincell : int or None
            Minimum counts in cells to remove poor quality cells with too few reads.
            Default: None
        maxreadsincell : int or None
            Maximum counts in cells to remove poor quality cells with too many reads.
            Default: None
        minreadsinregion : int or None
            Minimum counts in region to remove low coverage regions.
            Default: None
        maxreadsinregion : int or None
            Maximum counts in region to remove low coverage regions.
            Default: None
        binarize : bool
            Whether to binarize the count matrix. Default: True
        trimcounts : int or None
            Whether to trim the maximum number of reads per cell and region.
            This is a generalization to the binarize option.
            Default: None (No trimming performed)

        """

        if minreadsincell is None:
            minreadsincell = 0

        if maxreadsincell is None:
            maxreadsincell = sys.maxsize

        if minreadsinregion is None:
            minreadsinregion = 0

        if maxreadsinregion is None:
            maxreadsinregion = sys.maxsize

        adata = self.adata.copy()
        if binarize:
            adata.X.data[adata.X.data > 0] = 1

        if trimcount is not None and trimcount > 0:
            adata.X.data[adata.X.data > trimcount] = trimcount

        cellcounts = np.asarray(adata.X.sum(axis=0)).flatten()
        adata.var.loc[:,'nFrags'] = cellcounts

        adata = adata[:, (adata.var.nFrags >= minreadsincell) &
                         (adata.var.nFrags <= maxreadsincell) &
                         (~adata.var.index.isin(['dummy']))].copy()

        regioncounts = np.asarray(adata.X.sum(axis=1)).flatten()
        adata.obs.loc[:,'nFrags'] = regioncounts

        adata = adata[(adata.obs.nFrags >= minreadsinregion) &
                      (adata.obs.nFrags <= maxreadsinregion), :].copy()
        return CountMatrix(adata.X, adata.obs, adata.var, adata.uns, adata.obsm, adata.varm)

    def filter_count_matrix(self, minreadsincell=None, maxreadsincell=None,
                            minreadsinregion=None, maxreadsinregion=None,
                            binarize=True, trimcount=None):
        """
        Applies in-place quality filtering to the count matrix.

        Parameters
        ----------
        minreadsincell : int or None
            Minimum counts in cells to remove poor quality cells with too few reads.
            Default: None
        maxreadsincell : int or None
            Maximum counts in cells to remove poor quality cells with too many reads.
            Default: None
        minreadsinregion : int or None
            Minimum counts in region to remove low coverage regions.
            Default: None
        maxreadsinregion : int or None
            Maximum counts in region to remove low coverage regions.
            Default: None
        binarize : bool
            Whether to binarize the count matrix. Default: True
        trimcounts : int or None
            Whether to trim the maximum number of reads per cell and region.
            This is a generalization to the binarize option.
            Default: None (No trimming performed)

        """
        warnings.warn('filter_count_matrix deprecated. Please use filter',
                      category=DeprecationWarning)
        return self.filter(minreadsincell=minreadsincell,
                           maxreadsincell=maxreadsincell,
                           minreadsinregion=minreadsinregion,
                           maxreadsinregion=maxreadsinregion,
                           binarize=binarize, trimcount=trimcount)

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

        if not isinstance(cell, np.ndarray):
            cell = np.asarray(cell)
        if not isinstance(group, np.ndarray):
            group = np.asarray(group)
        for i, glab in enumerate(grouplabels):
            adata = self.adata[:, self.adata.var.index.isin(cell[group == glab])]
            cnts[:, i:(i+1)] = adata.X.sum(1)

        cannot = pd.DataFrame(grouplabels, columns=['barcode'])
        cannot.set_index('barcode', inplace=True)
        return CountMatrix(csr_matrix(cnts), adata.obs, cannot)

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
        if not isinstance(cell, np.ndarray):
            cell = np.asarray(cell)
        adata = self.adata[:, self.adata.var.index.isin(cell)]

        return CountMatrix(adata.X.tocsr(), adata.obs, adata.var, adata.uns, adata.obsm, adata.varm)

#    def split(self, samplename):
#        cms = []
#        names = self.adata.var.loc[:,samplename].unique()
#        for name in names:
#            ada = self.adata[:, self.adata.var[samplename]==name]
#            cms.append(CountMatrix(ada.X, ada.obs, ada.var, ada.uns, ada.obsm, ada.varm))
#        return cms
#
    def __repr__(self):
        return "{} x {} CountMatrix with {} entries".format(self.adata.shape[0], self.adata.shape[1], self.adata.X.nnz)

    @property
    def shape(self):
        return self.adata.shape

    @property
    def __len__(self):
        return self.shape[0]

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
        Exports the countmatrix

        Parameters
        ----------
        filename : str
            Output file name.
            File format is determined by the ending: mtx, npz, h5ad.
        """
        if filename.endswith('.mtx'):
            self.to_mtx(filename)
        elif filename.endswith('.npz'):
            self.to_npz(filename)
        elif filename.endswith('.h5ad'):
            self.to_h5ad(filename)
        else:
            # default to mtx format
            self.to_mtx(filename)


    def to_mtx(self, filename):
        """
        Exports the countmatrix in matrix market format

        Parameters
        ----------
        filename : str
            Output file name.
        """
        save_sparsematrix(filename, self.cmat, self.cannot)

    def to_npz(self, filename):
        """
        Exports the countmatrix in npz format

        Parameters
        ----------
        filename : str
            Output file name.
        """

        colnames = self.cannot.to_dict(orient='list')
        rownames = self.regions.to_dict(orient='list')
        np.savez(filename, self.cmat.data, self.cmat.indices, self.cmat.indptr)
        save_cellannotation(filename, self.cannot)

    def to_h5ad(self, filename):
        """
        Exports the countmatrix in h5ad format

        Parameters
        ----------
        filename : str
            Output file name.
        """
        self.adata.write(filename)
