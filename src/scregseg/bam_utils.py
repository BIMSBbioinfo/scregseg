import logging
import os
from copy import copy
import pysam
from pysam import AlignmentFile
from pybedtools import BedTool
import numpy as np
import pandas as pd
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
       Output folder in which the pseudo-bulk bam files are stored.
    cells : list(str)
       List of cells/barcode identifiers.
    grouplabels : list
       List of group/cluster associations corresponding to the cells.
    tag : str or callable
       Barcode tag or callable to extract barcode from the alignments. Default: 'CB'
    threads : int
       Number of threads
    make_bigwigs : bool
       Whether to also prepare bigwig-files. This will require deeptools to be installed.
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


def fragmentlength_in_regions(bamfile, regions, mapq, maxlen, resolution):
    """ Extract fragment lengths per region.

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
