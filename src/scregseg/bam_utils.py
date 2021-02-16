import logging
import os
from collections import Counter, OrderedDict
from copy import copy
import pysam
from pysam import AlignmentFile
from pybedtools import BedTool, Interval
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import coo_matrix
from scipy.sparse import dia_matrix
from multiprocessing import Pool
from scregseg.utils import run_commandline


class Barcoder:
    """ Class for extracting barcode from specific field

    Parameters
    ----------
    tag : str or callable
        Specifies which alignment tag should be considered as barcode.
        Alternatively, a callable can be supplied that extracts custom
        barcode encoding from the alignment.

    """

    def __init__(self, tag):
        logging.debug('Barcodes determined from {} tag'.format(tag))
        self.tag = tag

    def __call__(self, aln):
        if callable(self.tag):
            barcode = self.tag(aln)
        elif self.tag in ':.':
            barcode = aln.query_name.split(self.tag)[0]
        elif aln.has_tag(self.tag):
            barcode = aln.get_tag(self.tag)
        else:
            barcode = 'dummy'
        return barcode


def deduplicate_reads(bamin, bamout, tag='CB'):
    """Performs deduplication within barcodes/cells.

    Parameters
    ----------
    bamin : str
        Position sorted input bamfile.
    bamout : str
        Output file containing deduplicated reads.
    tag : str or callable
        Indicates the barcode tag or custom function to extract the barcode. Default: 'CB'

    Returns
    -------
    None

    """
    bamfile = AlignmentFile(bamin, 'rb')
    output = AlignmentFile(bamout, 'wb', template=bamfile)

    last_barcode = {}
    barcoder = Barcoder(tag)

    for aln in bamfile.fetch():
        # if previous hash matches the current has
        # skip the read
        val = (aln.reference_id, aln.reference_start,
               aln.is_reverse, aln.tlen)
        barcode = barcoder(aln)

        if barcode not in last_barcode:
            output.write(aln)
            # clear dictionary
            last_barcode[barcode] = val

        if val == last_barcode[barcode]:
            continue
        else:
            output.write(aln)
            last_barcode[barcode] = val


def remove_chroms(bamin, bamout, rmchroms):
    """ Removes chromosomes from bam-file.

    The function searches for matching chromosomes
    using regular expressions.
    For example, rmchroms=['chrM', '_random']
    would remove 'chrM' as well as all random chromsomes.
    E.g. chr1_KI270706v1_random.

    Parameters
    ----------
    bamin : str
       Input bam file.
    bamout : str
       Output bam file.
    rmchroms : list(str)
       List of chromosome names or name patterns to be removed.

    Returns
    -------
    None

    """

    treatment = AlignmentFile(bamin, 'rb')

    header = copy(treatment.header.as_dict())
    newheader = []
    for seq in header['SQ']:

        if not any([x in seq['SN'] for x in rmchroms]):
            newheader.append(seq)

    header['SQ'] = newheader

    tidmap = {k['SN']: i for i, k in enumerate(header['SQ'])}

    bam_writer = AlignmentFile(bamout, 'wb', header=header)

    # write new bam files containing only valid chromosomes
    for aln in treatment.fetch(until_eof=True):
        if aln.is_unmapped:
            continue
        if aln.reference_name not in tidmap or aln.next_reference_name not in tidmap:
            continue

        refid = tidmap[aln.reference_name]
        refnextid = tidmap[aln.next_reference_name]

        aln.reference_id = refid
        aln.next_reference_id = refnextid
        bam_writer.write(aln)

    bam_writer.close()
    treatment.close()


def cell_scaling_factors(file, selected_barcodes=None, *args, **kwargs):
    """ Generates pseudo-bulk tracks.

    Parameters
    ----------
    file : str
       Input bam file.
    tag : str or callable
       Barcode tag or callable to extract barcode from the alignments. Default: 'CB'
    mapq : int
       Minimum mapping quality. Default: 10

    Returns
    -------
    pd.Series
       Series containing the barcode counts per barcode.

    """
    if file.endswith('.bam'):
        return cell_scaling_factors_bam(file, selected_barcodes=selected_barcodes, *args, **kwargs)
    else:
        return cell_scaling_factors_fragments(file, selected_barcodes=selected_barcodes)

def cell_scaling_factors_bam(file, selected_barcodes=None, tag='CB', mapq=10):
    """ Generates pseudo-bulk tracks.

    Parameters
    ----------
    file : str
       Input bam file.
    tag : str or callable
       Barcode tag or callable to extract barcode from the alignments. Default: 'CB'
    mapq : int
       Minimum mapping quality. Default: 10

    Returns
    -------
    pd.Series
       Series containing the barcode counts per barcode.

    """

    barcodecount = Counter()
    afile = AlignmentFile(file, 'rb')
    barcoder = Barcoder(tag)
    for aln in afile.fetch():
        if aln.mapping_quality < mapq:
            continue
        bct = barcoder(aln)
        if selected_barcodes is not None:
            if bct not in selected_barcodes:
               continue
        barcodecount[bct] += 1
    return pd.Series(barcodecount)

def cell_scaling_factors_fragments(fragmentfile, selected_barcodes=None):
    """ Generates pseudo-bulk tracks.

    Parameters
    ----------
    fragmentfile : str
       Input fragments file.

    Returns
    -------
    pd.Series
       Series containing the barcode counts per barcode.

    """

    barcodecount = Counter()
    bed = BedTool(fragmentfile)
    for region in bed:
        bct = region.name
        if selected_barcodes is not None:
            if bct not in selected_barcodes:
               continue
        barcodecount[bct] += 1
    return pd.Series(barcodecount)

def profile_counts(inbamfile, genomicregion,
                   selected_barcodes=None,
                   tag='CB', mapq=10, binsize=50):
    """ Generates pseudo-bulk tracks.

    Parameters
    ----------
    bamin : str
       Input bam file.
    genomicregion : str
       Genomic coordinates. E.g. 'chr1:5000-10000'
    selected_barcodes : list(str) or None
       Contains a list of barcodes to consider for the profile.
       If None, all barcodes are considered. Default=None.
    tag : str or callable
       Barcode tag or callable to extract barcode from the alignments. Default: 'CB'
    mapq : int
       Minimum mapping quality. Default: 10
    binsize : int
       Resolution of the signal track in bp. Default: 50

    Returns
    -------
    anndata.AnnData
       AnnData object containing the read counts for the given locus.

    """

    afile = AlignmentFile(inbamfile, 'rb')
    
    barcoder = Barcoder(tag)

    def split_iv(gr):
        chr_, res = gr.split(':')
        start,end = res.split('-')
        return chr_, int(start), int(end)

    positions = []
    cells = []
    chrom, start, end = split_iv(genomicregion)
    barcodemap = OrderedDict()

    for aln in afile.fetch(chrom, start, end):
        bar = barcoder(aln)
        if aln.mapping_quality < mapq:
            continue
        if aln.is_unmapped:
            continue
        if selected_barcodes is not None:
            if bar not in selected_barcodes:
                # skip barcode if not in selected_barcodes list
                continue
        if not aln.is_reverse:
            pos = aln.reference_start
        else:
            pos = aln.reference_start + aln.inferred_length
        if pos > end or pos < start:
            continue
        positions.append(pos-start)
        
        if bar not in barcodemap:
            barcodemap[bar] = len(barcodemap)
        cells.append(barcodemap[bar])

    afile.close()

    smat = coo_matrix((np.ones(len(positions)), (positions, cells)),
                      shape=(end-start, len(barcodemap)),
                      dtype='int32')
    data = np.ones((binsize,smat.shape[0]))
    offsets = np.arange(binsize)
    di = dia_matrix((data,offsets), shape=(smat.shape[0],smat.shape[0]))
    smat = di.dot(smat).tocsr()
    smat = smat[::binsize]

    var = pd.DataFrame({'chrom':[chrom] *int(np.ceil((end-start)/binsize)),
                        'start':np.arange(start,end, binsize),
                        'end':np.arange(start+binsize,end+binsize, binsize)})
                        
    obs = pd.DataFrame(index=[bc for bc in barcodemap])
    adata = AnnData(smat.T.tocsr(), obs=obs, var=var)
    adata.raw = adata
    return adata


def make_pseudobulk_bam(inbamfile, outputdir,
                        cells, grouplabels,
                        tag='CB',
                        threads=10,
                        make_bigwigs=True):
    """ Generates pseudo-bulk tracks.

    Parameters
    ----------
    bamin : str
       Input bam file.
    outputdir : str
       Output folder in which the pseudo-bulk bam files are stored. Optionally,
       bigwig tracks will be stored there as well (see make_bigwigs).
    cells : list(str)
       List of cells/barcode identifiers.
    grouplabels : list
       List of group/cluster associations corresponding to the cells.
    tag : str or callable
       Barcode tag or callable to extract barcode from the alignments. Default: 'CB'
    threads : int
       Number of threads
    make_bigwigs : bool
       Whether to also prepare bigwig-files. This option will require deeptools to be installed.

    Returns
    -------
    None
    """

    os.makedirs(outputdir, exist_ok=True)

    assert len(cells) == len(grouplabels)

    barcode2groupmap = {}
    for i, _ in enumerate(cells):
        barcode2groupmap[cells[i]] = grouplabels[i]

    groups = list(set(grouplabels))

    bamreader = AlignmentFile(inbamfile, 'rb')
    barcoder = Barcoder(tag)

    bam_writer = {group: AlignmentFile(os.path.join(outputdir,
                                                    '{}.bam'.format(group)),
                  'wb', template=bamreader) for group in groups}


    for aln in bamreader.fetch(until_eof=True):
        bct = barcoder(aln)
        if bct in barcode2groupmap:
            bam_writer[barcode2groupmap[bct]].write(aln)

    bamreader.close()
    for b in bam_writer:
        bam_writer[b].close()
    for group in groups:
        pysam.index(os.path.join(outputdir,'{}.bam'.format(group)))

    if not make_bigwigs:
        return

    bwfiles = {os.path.join(outputdir,'{}.bam'.format(group)):
               os.path.join(outputdir,'{}.bigwig'.format(group)) for group in groups}

    pool = Pool(threads)
    cmd = 'bamCoverage --normalizeUsing CPM -b {} -o {}'
    results = pool.map(run_commandline, ((cmd, k, bwfiles[k]) for k in bwfiles))


def fragmentlength_from_bed(bedfile, regions, maxlen):
    """ Compute fragment length per region from a bed-file or BedTool obj

    Parameters
    ----------
    bedfile : str, BedTool
        Bed-file or BedTool object containing the fragments.
    regions : str, BedTool
        Bed-file or BedTool object containing the regions.
    maxlen : int
        Maximum fragment length.

    Returns
    -------
    scipy.sparse.coo_matrix
        Sparse regions by maxlen matrix containing the fragment counts.
    """
    if not isinstance(bedfile, BedTool):
        bedfile = BedTool(bedfile)

    if not isinstance(regions, BedTool):
        regions = BedTool(regions)

    roi = BedTool([Interval(iv.chrom, iv.start, iv.end, str(i)) for i, iv in enumerate(regions)])
    n = len(roi[0].fields)
    inter = roi.intersect(bedfile, wo=True)
    rowids = []
    colids = []
    shape=(len(roi), maxlen+1)
    for iv in inter:
        rowids.append(int(iv.name))
        colids.append(min(int(iv.fields[n+2]) - int(iv.fields[n+1]), maxlen))
    mat = coo_matrix((np.ones(len(rowids)), (rowids, colids)), shape=shape)
    return mat


def fragmentlength_from_bam(bamfile, regions, mapq, maxlen):
    """ Compute fragment length per region from a bam-file or

    Parameters
    ----------
    bamfile : str
        bam-file 
    regions : str, BedTool
        Bed-file or BedTool object containing the regions.
    mapq : int
        Minimum mapping quality.
    maxlen : int
        Maximum fragment length.

    Returns
    -------
    scipy.sparse.coo_matrix
        Sparse regions by maxlen matrix containing the fragment counts.
    """
    chroms = []
    starts = []
    ends = []
    tlens = []

    afile = AlignmentFile(bamfile, "rb")
    for aln in afile.fetch():
        if aln.mapping_quality < mapq:
            continue
        if aln.is_proper_pair and aln.is_read1:
            start = min(aln.reference_start, aln.next_reference_start)
            end = abs(aln.tlen)
            chroms.append(aln.reference_name)
            starts.append(start)
            ends.append(end)
    df = pd.DataFrame({'chrom':chroms, 'start':starts, 'end':ends})
    fragments = BedTool.from_dataframe(df)

    return fragmentlength_from_bed(fragments, regions, maxlen) 


def fragmentlength_in_regions(file, regions, mapq, maxlen, resolution):
    """ Extract fragment lengths per region.

    Deprecated.

    Parameters
    ----------
    bamfile : str
       Indexed input bam file.
    regions : str
       Regions in bed format. Must be genome-wide bins.
    mapq : int
       Mapping quality
    maxlen : int
       Maximum fragment length.
    resolution : int
       Base pair resolution.

    Return
    -------
        CountMatrix and annotation as pd.DataFrame
    """
    
    warnings.warn('fragmentlength_in_regions deprecated.',
                  category=DeprecationWarning)
    bed = BedTool(regions)
    binsize = bed[0].end - bed[0].start
    fragments = np.zeros((len(bed), maxlen//resolution))
    m = {(iv.chrom, iv.start): i for i, iv in enumerate(bed)}

    afile = AlignmentFile(bamfile, "rb")

    for aln in afile.fetch():
        if aln.mapping_quality < mapq:
            continue
        if aln.is_proper_pair and aln.is_read1:

            pos = (min(aln.reference_start, aln.next_reference_start) // binsize) * binsize

            tlen = abs(aln.tlen)//resolution
            if tlen < maxlen // resolution:
                if (aln.reference_name, pos) in m:
                    fragments[m[(aln.reference_name, pos)], tlen] += 1

    afile.close()
    cmat = fragments
    cannot = pd.DataFrame({'barcode':
                           ['{}bp'.format(bp*resolution) \
                            for bp in range(maxlen// resolution)]})

    return cmat, cannot
