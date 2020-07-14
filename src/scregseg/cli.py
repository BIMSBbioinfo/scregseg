"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mscregseg` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``scregseg.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``scregseg.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
import glob
from scregseg.countmatrix import CountMatrix
from scregseg.countmatrix import write_cannot_table
from scregseg.countmatrix import get_cell_annotation
from scregseg.countmatrix import make_counting_bins
from scregseg.countmatrix import load_count_matrices
from scregseg.countmatrix import sparse_count_reads_in_regions
from scregseg.hmm import MultinomialHMM
from scregseg.hmm import DirMulHMM
from scregseg import Scregseg
from scregseg.scregseg import run_segmentation
from scregseg.scregseg import get_statecalls
from scregseg.utils import fragmentlength_by_state
from scregseg.scregseg import export_bed
from scregseg.motifs import MotifExtractor
from scregseg.motifs import MotifExtractor2
from scipy.sparse import hstack
from scipy.stats import zscore
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pybedtools import BedTool
from pybedtools import Interval
import argparse
import logging


parser = argparse.ArgumentParser(description='Scregseg - Single-Cell REGulatory landscape SEGmentation.')

subparsers = parser.add_subparsers(dest='program')


# dataset preprocessing and rearrangements
counts = subparsers.add_parser('bam_to_counts', description='Make countmatrix')
counts.add_argument('--bamfile', dest='bamfile', type=str, help="Location of an indexed BAM-file", required=True)
counts.add_argument('--regions', dest='regions', type=str, help="Location of regions in BED format. ", required=True)
counts.add_argument('--counts', dest='counts', type=str,
                    help="Location of the output count matrix. "
                    "The matrix will be in matrix market format. Another file will be stored with ending '.bct'"
                    " holding information associated with the barcode names.", required=True)
counts.add_argument('--barcodetag', dest='barcodetag', type=str,
                    help="Barcode encoding tag. For instance, CB or RG depending on which tag represents "
             "the barcode. If the barcode is encoded as prefix in the read name "
             "separated by '.' or ':' use '.' or ':'.", default='CB')

counts.add_argument('--cellgroup', dest='cellgroup', type=str,
                    help="(Optional) Location of table (csv or tsv) defining groups of cells. "
                         "If specified, a pseudo-bulk count matrix will be created. "
                         "The table must have at least two columns, the first specifying the barcode name "
                         " and the second specifying the group label.")

counts.add_argument('--mode', dest='mode', type=str, default='midpoint',
                    help='Indicates whether to count mid-points, both ends '
                    'independently or once if either end is located in the interval.'
                    ' Options are midpoint, countboth and eitherend. '
                    ' Default: mode=midpoint', choices=['eitherend', 'midpoint', 'countboth'])

batchannot = subparsers.add_parser('sampleannot', description='Add sample annotation to a count matrix')
batchannot.add_argument('--counts', dest='counts', type=str, help="Location of a count matrix.", required=True)
batchannot.add_argument('--batches', dest='batches', type=str, nargs='+',
                        help="Batch name:value pairs to attach", required=True)

filtering = subparsers.add_parser('filter_counts', description='Filter countmatrix to remove poor quality cells')
filtering.add_argument('--incounts', dest='incounts', type=str, help="Location of input count matrix", required=True)
filtering.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
filtering.add_argument('--outcounts', dest='outcounts', type=str, help="Location of output count matrix", required=True)
filtering.add_argument('--mincount', dest='mincounts', type=int,
                       default=0, help='Minimum number of counts per cell. Default: 0')
filtering.add_argument('--minregioncount', dest='minregioncounts', type=int,
                       default=0, help='Minimum number of counts per region. Default: 0')
filtering.add_argument('--maxcount', dest='maxcounts', type=int,
                       default=sys.maxsize, help='Maximum number of counts per cell. Default: maxint')
filtering.add_argument('--trimcount', dest='trimcounts', type=int,
                       default=sys.maxsize,
                       help='Maximum number of counts per matrix element. '
                       'This can be used to mitigate the effect of artifacts '
                       '(e.g. high read counts in a single cell at some region).'
                       '--trimcount 1 amounts to binarization. Default: no trimming')

merge = subparsers.add_parser('merge', description='Merge count matrices across cells')
merge.add_argument('--incounts', dest='incounts', type=str,
                   nargs='+', help="Location of one or more input count matrices", required=True)
merge.add_argument('--regions', dest='regions', type=str,
                   help="Location of regions in bed format", required=True)
merge.add_argument('--outcounts', dest='outcounts', type=str,
                   help="Location of the merged output count matrix", required=True)

groupcells = subparsers.add_parser('groupcells', description='Collapse cells within pre-defined groups (make pseudo-bulk).')
groupcells.add_argument('--incounts', dest='incounts', type=str, help="Location of an input count matrix", required=True)
groupcells.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
groupcells.add_argument('--outcounts', dest='outcounts', type=str,
                        help="Location of the collapsed output count matrix", required=True)
groupcells.add_argument('--cellgroup', dest='cellgroup', type=str,
                        help="Location of a table defining cell groups within which to aggregate the reads.",
                        required=True)

subset = subparsers.add_parser('subset', description='Subset cells by cell name.')
subset.add_argument('--incounts', dest='incounts', type=str, help="Location of an input count matrix", required=True)
subset.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
subset.add_argument('--outcounts', dest='outcounts', type=str, help="Location of an output count matrix", required=True)
subset.add_argument('--subset', dest='subset', type=str,
                    help="Location of a table defining "
                    "cell names which to retain for the output count matrix.",
                    required=True)

# score a model from scratch
fsegment = subparsers.add_parser('fit_segment', description='Fit a Scregseg segmentation model.')
fsegment.add_argument('--counts', dest='counts', nargs='+', type=str,
                      help="Location of one or more input count matrices. Must span the same regions.", required=True)
fsegment.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
fsegment.add_argument('--storage', dest='storage', type=str, help="Location of the output folder.", required=True)
fsegment.add_argument('--labels', dest='labels', nargs='*', type=str,
                      help="Label names for the countmatrices")
fsegment.add_argument('--mincount', dest='mincounts', type=int,
                      default=0, help='Minimum number of counts per cell. Default: 0')
fsegment.add_argument('--maxcount', dest='maxcounts', type=int, default=sys.maxsize,
                      help='Maximum number of counts per cell. Default: maxint')
fsegment.add_argument('--minregioncount', dest='minregioncounts', type=int, default=0,
                      help='Minimum number of counts per region. Default: 0')
#fsegment.add_argument('--topfrac', dest='topfrac', type=float, default=1., help='Fraction of top most covered regions to use.')
fsegment.add_argument('--trimcount', dest='trimcounts', type=int,
                      default=sys.maxsize,
                      help='Maximum number of counts per matrix element. '
                      'For instance, trimcount 1 amounts to binarization. Default: maxint')

fsegment.add_argument('--nstates', dest='nstates', type=int, default=20, help="Number of states. Default: 20")
fsegment.add_argument('--randomseed', dest='randomseed', nargs='+', type=int, default=32,
                      help='Random seed or list of seeds for reproduability. '
                      'We recommend to to specify multiple seeds which restarts '
                      'the model fitting with random initial weights for each seed.'
                      'Subsequently, the best scoring model is saved. This allows '
                      'to mitigates the issue of obtaining a model getting stuck '
                      'in a poor local optimum after model fitting. Default: 32')
fsegment.add_argument('--niter', dest='niter', type=int, default=100,
                      help='Number of EM iterations. Default: 100')
fsegment.add_argument('--n_jobs', dest='n_jobs', type=int, default=1,
                      help='Number jobs to use for parallel processing. Default: 1')
fsegment.add_argument('--modelname', dest='modelname', type=str,
                      default='dirmulhmm', help='Model name. Default: dirmulhmm')

# fitting a model from scratch
segment = subparsers.add_parser('segment', description='Re-runs state calling for an existing Scregseg model.')
segment.add_argument('--counts', dest='counts', nargs='+', type=str,
                      help="Location of one or more input count matrices. Must span the same regions.", required=True)
segment.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
segment.add_argument('--storage', dest='storage', type=str, help="Location of the output folder.", required=True)
segment.add_argument('--labels', dest='labels', nargs='*', type=str,
                      help="Label names for the countmatrices")
segment.add_argument('--mincount', dest='mincounts', type=int,
                      default=0, help='Minimum number of counts per cell. Default: 0')
segment.add_argument('--maxcount', dest='maxcounts', type=int, default=sys.maxsize,
                      help='Maximum number of counts per cell. Default: maxint')
segment.add_argument('--minregioncount', dest='minregioncounts', type=int, default=0,
                      help='Minimum number of counts per region. Default: 0')
#fsegment.add_argument('--topfrac', dest='topfrac', type=float, default=1., help='Fraction of top most covered regions to use.')
segment.add_argument('--trimcount', dest='trimcounts', type=int,
                      default=sys.maxsize,
                      help='Maximum number of counts per matrix element. '
                      'For instance, trimcount 1 amounts to binarization. Default: maxint')
segment.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name. Default: dirmulhmm')

plotannotate = subparsers.add_parser('plot_emission', description='Plot state-emissions.')
plotannotate.add_argument('--output', dest='output', type=str,
                          help="Output figure.", required=True)
plotannotate.add_argument('--storage', dest='storage', type=str,
                          help="Location for containing the pre-trained segmentation "
                          "and for storing the annotated segmentation results", required=True)
plotannotate.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name. Default: dirmulhmm')
plotannotate.add_argument('--states', dest='states', type=str, nargs='*',
                          help='Selection of states to visualize. Default: all states are visualized')


seg2bed = subparsers.add_parser('seg_to_bed', description='Export state calls in BED-format')
seg2bed.add_argument('--storage', dest='storage', type=str, help="Location of the model folder.", required=True)
seg2bed.add_argument('--output', dest='output', type=str, help='Output BED file containing the state calls.', required=True)
seg2bed.add_argument('--method', dest='method', type=str, default='rarest',
                     help='Method for selecting states for exporting:'
                     'rarest exports the --nsmallest states, '
                     'manuelselect exports a list of manually specified states given by --statenames,'
                     'nucfree exports the --nlargest states most enriched for nucleosome free reads (<=150bp),'
                     'abundancethreshold selects the states with maximum state abundance given by --max_state_abundance. Default: rarest',
                     choices=['rarest', 'nucfree', 'manuelselect', 'abundancethreshold'])
seg2bed.add_argument('--individual', dest='individualbeds', action='store_true', default=False,
                     help="Save segmentation in individual bed files. Default: False exports a single bed file containing all states.")
seg2bed.add_argument('--threshold', dest='threshold', type=float, default=0.0,
                     help="Threshold on posterior decoding probability. "
                        "Only export state calls that exceed the posterior decoding threshold. "
                        "This allows to adjust the stringency of state calls for down-stream analysis steps. Default: 0.0")
seg2bed.add_argument('--no_bookended_merging', dest='no_bookended_merging', action='store_true',
                     default=False,
                     help='Whether to merge neighboring bins representing the same state. Default=False.')
seg2bed.add_argument('--exclude_states', dest='exclude_states', nargs='*',
                     type=str, help='List of state names which should be exclued.')
seg2bed.add_argument('--max_state_abundance', dest='max_state_abundance', type=float, default=1.,
         help='Max. state abundance across the genome. '
         'This parameters allows to report only rarely occurring states. '
         'Abundant states are filtered out as they usually reflect the genomic background distribution. '
         'A good choice for this is a value that is slightly lower than 1./n_state. Default=1.')
seg2bed.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name. Default: dirmulhmm')

seg2bed.add_argument('--nsmallest', dest='nsmallest', type=int, default=-1,
                     help='Number of most rare states to export. Default: -1 (all states are considered).')
seg2bed.add_argument('--nlargest', dest='nlargest', type=int, default=-1,
                     help='Number of most nucleosome-free fragment enriched '
                     'states to export. Default: -1 (all states are considered).')
seg2bed.add_argument('--nregsperstate', dest='nregsperstate', type=int, default=-1,
                     help='Number of regions per state to export. Usually, only a subset of representative state calls'
                     'need to be exported to achieve satisfying results for the downstream clustering.')
seg2bed.add_argument('--statenames', dest='statenames', nargs='*',
                     help='List of states to export.')



annotate = subparsers.add_parser('annotate', description='Annotate Scregseg states.')
annotate.add_argument('--files', dest='files', nargs='+', type=str,
                      help="Location of a BAM-, BIGWIG or BED-files to annotate the states with.", required=True)
annotate.add_argument('--labels', dest='labels', nargs='+', type=str,
                      help="Annotation labels.", required=True)
annotate.add_argument('--storage', dest='storage', type=str,
                      help='Location for containing the pre-trained segmentation and '
                      'for storing the annotated segmentation results', required=True)
annotate.add_argument('--modelname', dest='modelname', type=str,
                      default='dirmulhmm', help='Model name. Default: dirmulhmm')

plotannotate = subparsers.add_parser('plot_annot', description='Plot state-annotation associated.')
plotannotate.add_argument('--labels', dest='labels', nargs='+', type=str,
                          help="Annotation labels.", required=True)
plotannotate.add_argument('--title', dest='title', type=str,
                          help="Plot name.", required=True)
plotannotate.add_argument('--storage', dest='storage', type=str,
                          help="Location for containing the pre-trained segmentation "
                          "and for storing the annotated segmentation results", required=True)
plotannotate.add_argument('--threshold', dest='threshold', type=float, default=0.0,
                          help="Threshold on posterior decoding probability. "
                                "Only export results that exceed the posterior decoding threshold. "
                                "This allows to adjust the stringency of state calls for down-stream analysis steps.")
plotannotate.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name. Default: dirmulhmm')
plotannotate.add_argument('--plottype', dest='plottype', type=str, default='boxplot',
                          choices=['boxplot', 'countplot', 'heatmap'],
                          help='Plot type')
plotannotate.add_argument('--groupby', dest='groupby', type=str,
                          help="Annotation label.", default=None)

enrichment = subparsers.add_parser('enrichment', description='Computes state abundance enrichment in and around the given feature sets')
enrichment.add_argument('--storage', dest='storage', type=str,
                        help="Location for containing the pre-trained segmentation "
                        "and for storing the annotated segmentation results.", required=True)
enrichment.add_argument('--title', dest='title', type=str,
                        help='Name of the state enrichment test. Default: "geneset"',
                        default='geneset')
enrichment.add_argument('--features', dest='features', type=str,
                        help='Path to a folder containing bed-files that define the feature sets.', required=True)
enrichment.add_argument('--flanking', dest='flanking', type=int,
                        default=30000,
                        help='Extention window in bp to consider for each feature/gene. Default=30000')
enrichment.add_argument('--method', dest='method', type=str, default='chisqstat',
                        help='Method to use for the state abundance enrichment in and around the features.'
                        'pvalue computes the negative log-pvalue of observing at least o state calls in a window of the size of the feature. This may be slow for long features (e.g. gene body) and large feature sets.'
                        'logfold compute the log ratio between observe fraction of state calls and expected fraction of state calls in the feature set (faster than pvalue).'
                        'chisqstat compute (o - e)^2 / e where o and e denote the observed and expected state number of state counts (faster than pvalue).'
                        'Default: chisqstat',
                        choices=['pvalue', 'logfold', 'chisqstat'])
enrichment.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name. Default: dirmulhmm')
enrichment.add_argument('--noplot', dest='noplot', action='store_true',
                        default=False, help='Whether to skip plotting the heatmap. Default: False')
enrichment.add_argument('--ntop', dest='ntop', type=int, default=5,
                        help='Report n top enriched features per state. Default: n=5')
enrichment.add_argument('--using_genebody', dest='using_genebody', action='store_true',
                        default=False, help='Uses state enrichment in gene body (+/- flank). '
                        'Otherwise, the state enrichment is determined around the TSS only (+/- flank)')
enrichment.add_argument('--output', dest='output', type=str,
                        default=None,
                        help='Alternative output directory. If not specified, '
                             'the results are stored in <storage>/<modelname>/annotation')

motifextraction = subparsers.add_parser('extract_motifs', description='Extract motifs associated with states')
motifextraction.add_argument('--storage', dest='storage', type=str,
                             help="Location for containing the pre-trained segmentation "
                             "and for storing the annotated segmentation results", required=True)
motifextraction.add_argument('--refgenome', dest='refgenome', type=str, help="Reference genome.", required=True)
motifextraction.add_argument('--ntop', dest='ntop', type=int,
                             help="Positive set size. Default: 15000", default=15000)
motifextraction.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name. Default: dirmulhmm')
motifextraction.add_argument('--ngap', dest='ngap', type=int,
                             help="Gap size between positive and negative set. Default: 70000",
                             default=70000)
motifextraction.add_argument('--nbottom', dest='nbottom', type=int,
                             help="Negative set size. Default: 15000",
                             default=15000)
motifextraction.add_argument('--flank', dest='flank', type=int,
                             help="Flank size added to each interval. Default: 250 bp",
                             default=250)
motifextraction.add_argument('--nmotifs', dest='nmotifs', type=int,
                             help="Number of motifs to report. Default: 10",
                             default=10)
motifextraction.add_argument('--output', dest='output', type=str,
                        default=None,
                        help='Alternative output directory. If not specified, '
                             'the results are stored in <storage>/<modelname>/motifs')
motifextraction.add_argument('--method', dest='method', type=str,
                        default='betweenstates',
                        help='Extraction method determines how to constitute the negative samples. betweenstates'
                             ' uses the other high-confidence state calls as contrast.')

fragmentsize = subparsers.add_parser('fragmentsize',
                                     description='Inspect fragment size distribution per state')
fragmentsize.add_argument('--storage', dest='storage', type=str,
                          help="Location for containing the pre-trained segmentation "
                          "and for storing the annotated segmentation results",
                          required=True)
fragmentsize.add_argument('--bamfiles', dest='bamfiles', type=str, nargs='+',
                          help="Location of one or more BAM file containing paired-end reads.",
                          required=True)
fragmentsize.add_argument('--label', dest='label', type=str, default='sample',
                          help="(Optional) Sample labels. Default: sample")
fragmentsize.add_argument('--modelname', dest='modelname',
                          type=str, default='dirmulhmm', help='Model name. Default: dirmulhmm')
fragmentsize.add_argument('--maxfraglen', dest='maxfraglen', type=int,
                          default=1000, help="Maximum fragment length in bp.")
fragmentsize.add_argument('--output', dest='output', type=str,
                        default=None,
                        help='Alternative output directory. If not specified, '
                             'the results are stored in <storage>/<modelname>/summary')




def make_folders(output):
    """ Create folder """
    os.makedirs(output, exist_ok=True)

def save_score(scmodel, data, output):
    """ Save log-likelihood score to file."""
    score = scmodel.score(data)
    logging.debug('loglikelihood = {}'.format(score))
    with open(os.path.join(output, "summary", "score.txt"), 'w') as f:
        f.write('{}\n'.format(score))

def get_cell_grouping(table):
    """ Extract cell-group mapping"""
    if table.endswith('.csv'):
        group2cellmap = pd.read_csv(table, sep=',', usecols=[0,1])
    elif table.endswith('.tsv'):
        group2cellmap = pd.read_csv(table, sep='\t', usecols=[0,1])

    group2cellmap.columns = ['cells', 'groups']

    cell = group2cellmap.cells.values
    groups = group2cellmap.groups.values

    return cell, groups

def make_state_summary(model, output, labels):
    """ Make and save summary statistics."""
    make_folders(os.path.join(output, 'summary'))
    model.get_state_stats().to_csv(os.path.join(output, 'summary', 'statesummary.csv'))
    model.plot_state_abundance().savefig(os.path.join(output, 'summary', 'state_abundance.svg'))
    model.plot_readdepth().savefig(os.path.join(output, 'summary', 'state_readdepth.svg'))

def plot_normalized_emissions(model, output, labels):
    """ Save normalized emission probabilities"""
    make_folders(os.path.join(output, 'summary'))
    for i, dataname in enumerate(labels):
        model.plot_normalized_emissions(i).savefig(os.path.join(output, 'summary', 'emission_{}.png'.format(dataname)))

def plot_state_annotation_relationship_heatmap(model, storage, labels,
                                       title, threshold=0.0, groupby=None):

    make_folders(os.path.join(storage, 'annotation'))

    fig, ax = plt.subplots(figsize=(10, 20))

    segdf = model._segments[model._segments.Prob_max>=threshold].copy()

    segdf_ = segdf[labels].apply(zscore)
    segdf_['name'] = segdf.name

    segdf = segdf_.groupby("name").agg('mean')

    sns.heatmap(segdf, cmap="RdBu_r",
                robust=True, center=0.0, ax=ax)
    logging.debug('writing {}'.format(os.path.join(storage, 'annotation', '{}.png'.format(title))))
    fig.tight_layout()
    fig.savefig(os.path.join(storage, 'annotation', '{}.png'.format(title)))

def plot_state_annotation_relationship(model, storage, labels,
                                       title, plottype='boxplot',
                                       threshold=0.0, groupby=None):

    make_folders(os.path.join(storage, 'annotation'))

    fig, axes = plt.subplots(len(labels))

    if len(labels) == 1:
        axes = [axes]

    segdf = model._segments[model._segments.Prob_max>=threshold].copy()

    for ax, label in zip(axes, labels):
        if plottype=='countplot':
            sns.countplot(y='name', hue=label,
                          data=segdf,
                          ax=ax)
        elif plottype == 'boxplot':
            #segdf['log_'+label] = np.log10(segdf[label]+1)
            sns.boxplot(x='log_'+label, y='name',
                        data=segdf,
                        hue=groupby, orient='h', ax=ax)

    logging.debug('writing {}'.format(os.path.join(storage, 'annotation', '{}.png'.format(title))))
    fig.tight_layout()
    fig.savefig(os.path.join(storage, 'annotation', '{}.png'.format(title)))




def local_main(args):
    if hasattr(args, 'storage'):
        logfile = os.path.join(args.storage, 'log', 'logfile.log')
        make_folders(os.path.dirname(logfile))
        logging.basicConfig(filename = logfile, level=logging.DEBUG,
                            format='%(asctime)s;%(levelname)s;%(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        
    logging.debug(args)
    
    if args.program == 'bam_to_counts':

        logging.debug('Make countmatrix ...')
        cm = CountMatrix.create_from_bam(args.bamfile,
                                    args.regions, barcodetag=args.barcodetag,
                                    mode=args.mode)

        if args.cellgroup is not None:
            cells,  groups = get_cell_grouping(args.cellgroup)
            cm = cm.pseudobulk(cells, groups)

        cm.export_counts(args.counts)

    elif args.program == 'filter_counts':
        logging.debug('Filter counts ...')
        cm = CountMatrix.create_from_countmatrix(args.incounts, args.regions)
        cm.filter_count_matrix(args.mincounts, args.maxcounts,
                               args.minregioncounts, binarize=False, maxcount=args.trimcounts)
        cm.export_counts(args.outcounts)

    elif args.program == 'batchannot':
        logging.debug('Adding annotation ...')
        cannot = get_cell_annotation(args.counts)
        for batch in args.batches:
            name, value = batch.split(':')
            cannot[name] = value
        write_cannot_table(args.counts, cannot)

    elif args.program == 'groupcells':
        logging.debug('Group cells (pseudobulk)...')
        cm = CountMatrix.create_from_countmatrix(args.incounts, args.regions)

        cells,  groups = get_cell_grouping(args.cellgroup)
        pscm = cm.pseudobulk(cells, groups)
        pscm.export_counts(args.outcounts)

    elif args.program == 'subset':

        logging.debug('Subset cells ...')
        cm = CountMatrix.create_from_countmatrix(args.incounts, args.regions)

        cells,  _ = get_cell_grouping(args.subset)
        pscm = cm.subset(cells)
        pscm.export_counts(args.outcounts)

    elif args.program == 'merge':
        logging.debug('Merge count matrices ...')
        cms = []
        for incount in args.incounts:
            cm = CountMatrix.create_from_countmatrix(incount, args.regions)
            cms.append(cm)


        merged_cm = CountMatrix.merge(cms)
        merged_cm.export_counts(args.outcounts)

    elif args.program == 'fit_segment':
        assert len(args.labels) == len(args.counts)

        outputpath = os.path.join(args.storage, args.modelname)
        logging.debug('Segmentation ...')
        # fit on subset of the data
        data = load_count_matrices(args.counts, args.regions,
                                               args.mincounts, args.maxcounts,
                                               args.trimcounts, args.minregioncounts)

        scmodel = run_segmentation(data, args.nstates,
                                   args.niter, args.randomseed,
                                   args.n_jobs)

        # predict on the entire genome
        data = load_count_matrices(args.counts, args.regions,
                                   args.mincounts, args.maxcounts,
                                   args.trimcounts, None)

        logging.debug('segmentation data:')
        for d in data:
            logging.debug(d)

        scmodel.segment(data, args.regions)
        scmodel.save(outputpath)

        logging.debug('summarize results ...')
        make_state_summary(scmodel, outputpath, args.labels)
        plot_normalized_emissions(scmodel, outputpath, args.labels)
        save_score(scmodel, data, outputpath)

    elif args.program == 'plot_emission':

        outputpath = os.path.join(args.storage, args.modelname)
        scmodel = Scregseg.load(outputpath)
        scmodel.plot_normalized_emissions(selectedstates=args.states).savefig(args.output)

    elif args.program == 'segment':
        assert len(args.labels) == len(args.counts)

        outputpath = os.path.join(args.storage, args.modelname)
        data = load_count_matrices(args.counts,
                                               args.regions,
                                               args.mincounts,
                                               args.maxcounts, args.trimcounts,
                                               0)
        scmodel = Scregseg.load(outputpath)
        logging.debug('State calling ...')
        scmodel.segment(data, args.regions)
        scmodel.save(outputpath)
        make_state_summary(scmodel, outputpath, args.labels)
        plot_normalized_emissions(scmodel, outputpath, args.labels)
        save_score(scmodel, data, outputpath)


    elif args.program == 'seg_to_bed':
        outputpath = os.path.join(args.storage, args.modelname)

        scmodel = Scregseg.load(outputpath)

        sdf = scmodel._segments.copy()
        if args.method == "manuelselect":
            if args.statenames is None:
                raise ValueError("--method manuelselect also requires --statenames <list state names>")
            query_states = args.statenames
        elif args.method == "rarest":
            if args.nsmallest <= 0:
                raise ValueError("--method rarest also requires --nsmallest <int>")
            query_states = pd.Series(scmodel.model.get_stationary_distribution(), index=['state_{}'.format(i) for i in range(scmodel.n_components)])
            query_states = query_states.nsmallest(args.nsmallest).index.tolist()
        elif args.method == "abundancethreshold":
            query_states = ['state_{}'.format(i) for i, p in enumerate(scmodel.model.get_stationary_distribution()) \
                            if p<=args.max_state_abundance]
        elif args.method == "nucfree":
            if 'nf_prop' not in scmodel._segments.columns:
                raise ValueError("'scregseg fragmentsize' must be run before.")
            if args.nlargest <= 0:
                raise ValueError("--method nucfree also requires --nlargest <int>")
            sdf.nf_prop = sdf.nf_prop.fillna(0.0)
            nrdf = sdf[['name', 'nf_prop']].groupby('name').mean()
            query_states = nrdf.nlargest(args.nlargest, 'nf_prop').index.tolist()

            sdf.readdepth = sdf.readdepth*sdf.nf_prop
        logging.debug(query_states)

        if args.exclude_states is not None:
            query_states = list(set(query_states).difference(set(args.exclude_states)))

        # subset and merge the state calls
        #subset = scmodel.get_statecalls(scmodel._segments, query_states, collapse_neighbors=args.merge_neighbors,
        #                                state_prob_threshold=args.threshold)
        subset = get_statecalls(sdf, query_states, ntop=args.nregsperstate,
                                collapse_neighbors=not args.no_bookended_merging,
                                         state_prob_threshold=args.threshold)

        logging.debug("Exporting {} states with {} regions".format(len(query_states), subset.shape[0]))
        if args.output == '':
            output = outputpath = os.path.join(args.storage, args.modelname, 'summary', 'segments.bed')
        else:
            output = args.output

        # export the state calls as a bed file
        export_bed(subset, output,
                   individual_beds=args.individualbeds)

    elif args.program == 'annotate':
        outputpath = os.path.join(args.storage, args.modelname)
        scmodel = Scregseg.load(outputpath)

        assert len(args.labels) == len(args.files), "Number of files and labels mismatching"
        logging.debug('annotate states ...')
        files = {key: filename for key, filename in zip(args.labels, args.files)}
        scmodel.annotate(files)

        scmodel.save(outputpath)

    elif args.program == 'plot_annot':
        outputpath = os.path.join(args.storage, args.modelname)
        logging.debug('Plot annotation ...')
        scmodel = Scregseg.load(outputpath)

        if args.plottype == 'heatmap':
            plot_state_annotation_relationship_heatmap(scmodel, outputpath,
                                           args.labels, args.title,
                                           args.threshold, args.groupby)
        else:
            plot_state_annotation_relationship(scmodel, outputpath,
                                           args.labels, args.title, args.plottype,
                                           args.threshold, args.groupby)

    elif args.program == 'enrichment':
        outputpath = os.path.join(args.storage, args.modelname)

        logging.debug('enrichment analysis')
        scmodel = Scregseg.load(outputpath)

        if args.output is None:
            outputenr = os.path.join(outputpath, 'annotation')
        else:
            outputenr = args.output

        make_folders(outputenr)

        if os.path.isdir(args.features):
            featuresets = glob.glob(os.path.join(args.features, '*.bed'))
            featurenames = [os.path.basename(name)[:-4] for name in featuresets]
            obs, lens, _ = scmodel.geneset_observed_state_counts(featuresets, flanking=args.flanking)
        else:
            obs, lens, featurenames = scmodel.observed_state_counts(args.features,
            flanking=args.flanking, using_tss = not args.using_genebody)
            obs.to_csv(os.path.join(outputenr, 'state_counts_{}.tsv'.format(args.title)), sep='\t')

        enr = scmodel.broadregion_enrichment(obs, mode=args.method)

        if not args.noplot:
            if args.method == 'logfold':
                g = sns.clustermap(enr, cmap="RdBu_r", figsize=(20,30), robust=True, **{'center':0.0, 'vmin':-1.5, 'vmax':1.5})

            elif args.method == 'chisqstat':
                g = sns.clustermap(enr, cmap="Reds", figsize=(20,30), robust=True)

            elif args.method == 'pvalue':
                g = sns.clustermap(enr, cmap="Reds", figsize=(20,30), robust=True)
            g.savefig(os.path.join(outputenr, "state_enrichment_{}_{}.png".format(args.method, args.title)))

        enr.to_csv(os.path.join(outputenr, 'state_enrichment_{}_{}.tsv'.format(args.method, args.title)), sep='\t')
        ntop = args.ntop
        with open(os.path.join(outputenr, 'state_enrichment_top{}_{}_{}.tsv'.format(ntop, args.method, args.title)), 'w') as f:
            for state in enr.columns:
                x = enr.nlargest(ntop, state)
                for i, row in x.iterrows():
                    f.write('{}\t{}\t{}\n'.format(state, i, row[state]))

    elif args.program == 'extract_motifs':
        outputpath = os.path.join(args.storage, args.modelname)

        if args.output is None:
            motifoutput = os.path.join(outputpath, 'motifs')
        else:
            motifoutput = args.output
        make_folders(motifoutput)

        scmodel = Scregseg.load(outputpath)
        if args.method == "regression":
            motifextractor = MotifExtractor(scmodel, args.refgenome, ntop=args.ntop,
                                            nbottom=args.nbottom, ngap=args.ngap,
                                            nmotifs=args.nmotifs, flank=args.flank)
        elif args.method == "betweenstates":
            motifextractor = MotifExtractor2(scmodel, args.refgenome, ntop=args.ntop,
                                                    nmotifs=args.nmotifs, flank=args.flank)
        else:
            raise ValueError("--method {} unknown. regression or classification supported.".format(args.method))

        os.environ['JANGGU_OUTPUT'] = motifoutput
        motifextractor.extract_motifs()
        motifextractor.save_motifs(os.path.join(motifoutput, 'scregseg_motifs.meme'))

    elif args.program == 'fragmentsize':
        logging.debug('Extract fragment size distribution ...')
        outputpath = os.path.join(args.storage, args.modelname)

        if args.output is None:
            resultspath = os.path.join(outputpath, 'summary')
        else:
            resultspath = args.output
        make_folders(resultspath)

        scmodel = Scregseg.load(outputpath)

        bed = BedTool([Interval(row.chrom, row.start, row.end) \
                       for _, row in scmodel._segments.iterrows()])

        aggfmat = None
        for bamfile in args.bamfiles:
            fmat = CountMatrix.create_from_fragmentsize(bamfile,
                                                        bed.TEMPFILES[-1],
                                                        resolution=1,
                                                        maxlen=args.maxfraglen)

            if aggfmat is None:
                aggfmat = fmat
            else:
                aggfmat.cmat += fmat.cmat

        nfr = np.asarray(aggfmat.cmat[:, :150].sum(1)).flatten()
        total = np.asarray(aggfmat.cmat.sum(1)).flatten()

        scmodel._segments['nf_prop'] = nfr / total
        scmodel._segments['nf_prop'] = scmodel._segments['nf_prop'].fillna(0.0)

        scmodel.save(outputpath)

        aggfmat.export_counts(os.path.join(resultspath, 
                           'fragmentsize_per_state_{}.mtx'.format(args.label)))
        adf = fragmentlength_by_state(scmodel, aggfmat)
        adf.to_csv(os.path.join(resultspath, 
                   'fragmentsize_per_state_{}.csv'.format(args.label)))

        fig, ax =  plt.subplots(figsize=(10,10))
        sns.heatmap(adf, ax=ax)
        fig.savefig(os.path.join(resultspath,
                    'fragmentsize_per_state_{}.svg'.format(args.label)))

        fig, ax =  plt.subplots(figsize=(10,10))
        x = np.asarray(aggfmat.cmat.sum(0)).flatten()
        ax.plot(np.arange(aggfmat.shape[1]), x)
        fig.savefig(os.path.join(resultspath, 
                    'fragmentsize_{}.svg'.format(args.label)))



def main():
    args = parser.parse_args()
    if args.program is None:
        parser.print_help()
        sys.exit(0)
    local_main(args)

if __name__ == '__main__':

    main()
