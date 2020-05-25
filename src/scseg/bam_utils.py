import os
from copy import copy
import pysam
from pysam import AlignmentFile


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
        print('Barcodes determined from {} tag'.format(tag))
        self.tag = tag

    def __call__(self, aln):
        if callable(self.tag):
            rg = self.tag(aln)
        elif aln.has_tag(self.tag):
            rg = aln.get_tag(self.tag)
        else:
            rg = 'dummy'
        return rg
        
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
        rg = barcoder(aln)

        if rg not in last_barcode:
            output.write(aln)
            # clear dictionary
            last_barcode[rg] = val

        if val == last_barcode[rg]:
            continue
        else:
            output.write(aln)
            last_barcode[rg] = val


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
    nh = []
    for i, seq in enumerate(header['SQ']):
#        remove=False
        if not any([x in seq['SN'] for x in rmchroms]):
            nh.append(seq)
#        for toremove in rmchroms:
#            if toremove in seq['SN']:
#                remove=True
#        if not remove:
#            nh.append(seq)

    header['SQ'] = nh

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
#        if aln.is_paired and aln.is_proper_pair:
        aln.next_reference_id = refnextid
        bam_writer.write(aln)

    bam_writer.close()
    treatment.close()


def make_pseudobulk_bam(inbamfile, outputdir, cells, grouplabels, tag='CB'):
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
    """

    os.makedirs(outputdir, exist_ok=True)

    assert len(cells) == len(grouplabels)

    barcode2groupmap = {}
    for i in range(len(cells)):
        barcode2groupmap[cells[i]] = grouplabels[i]
    
    groups = list(set(grouplabels))

    bamreader = AlignmentFile(inbamfile, 'rb')
    barcoder = Barcoder(tag)

    bam_writer = {group: AlignmentFile(os.path.join(outputdir, '{}.bam'.format(group)), 'wb', template=bamreader) for group in groups}
    cin = 0
    cout = 0

    for aln in bamreader.fetch(until_eof=True):
        bct = barcoder(aln)
        if bct in barcode2groupmap:
            bam_writer[barcode2groupmap[bct]].write(aln)
            cin += 1
        else:
            cout += 1

    bamreader.close()
    for b in bam_writer:
        bam_writer[b].close()

