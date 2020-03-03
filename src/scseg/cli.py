"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mscseg` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``scseg.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``scseg.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import os
import sys
import glob
from scseg.countmatrix import CountMatrix
#from scseg.countmatrix import get_cell_annotation_first_row_
from scseg.countmatrix import write_cannot_table
from scseg.countmatrix import get_cell_annotation
from scseg.countmatrix import make_counting_bins
from scseg.countmatrix import sparse_count_reads_in_regions
from scseg.hmm import MultinomialHMM
from scseg.hmm import DirMulHMM
from scseg import Scseg
from scseg.scseg import export_bed
from scipy.sparse import hstack
from scipy.stats import zscore
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#from umap import UMAP
import numpy as np
from pybedtools import BedTool
from pybedtools import Interval
import argparse

parser = argparse.ArgumentParser(description='Scseg - single-cell genome segmentation.')
#parser.add_help()
#subparsers = parser.add_subparsers(dest='segment', help='segmentation')

subparsers = parser.add_subparsers(dest='program')


# dataset preprocessing and rearrangements
counts = subparsers.add_parser('bam_to_counts', help='Make countmatrix')
counts.add_argument('--bamfile', dest='bamfile', type=str, help="Location of a bamfile", required=True)
counts.add_argument('--binsize', dest='binsize', type=int, help="Binsize", default=1000)
counts.add_argument('--barcodetag', dest='barcodetag', type=str, help="Barcode readtag", default='CB')
counts.add_argument('--counts', dest='counts', type=str, help="Location of count matrix or matrices", required=True)
counts.add_argument('--cellgroup', dest='cellgroup', type=str, help="Location of table defining cell to group mapping.")
counts.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format. If regions bed file exists, "
                                              "it will be used to construct the count matrix. In this case, binsize is ignored. "
                                              "Otherwise, a regions file is created using the specified binsize.", required=True)
counts.add_argument('--mode', dest='mode', type=str, default='eitherend', help='Indicates whether to count mid-points, both ends '
                                              'independently or once if either end is located in the interval. Options are midpoint, countboth and eitherend. '
                                              ' Default: mode=eitherend')

batchannot = subparsers.add_parser('batchannot', help='Add batch annotation')
batchannot.add_argument('--counts', dest='counts', type=str, help="Location of count matrix", required=True)
batchannot.add_argument('--batches', dest='batches', type=str, nargs='+', help="Batch name:value pairs to attach", required=True)

filtering = subparsers.add_parser('filter_counts', help='Filter countmatrix to remove poor quality cells')
filtering.add_argument('--incounts', dest='incounts', type=str, help="Location of input count matrix", required=True)
filtering.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
filtering.add_argument('--outcounts', dest='outcounts', type=str, help="Location of output count matrix", required=True)
filtering.add_argument('--mincount', dest='mincounts', type=int, default=0, help='Minimum number of counts per cell')
filtering.add_argument('--maxcount', dest='maxcounts', type=int, default=sys.maxsize, help='Maximum number of counts per cell')
filtering.add_argument('--trimcount', dest='trimcounts', type=int, default=sys.maxsize, help='Maximum number of counts per matrix element. For instance, trimcount 1 amounts to binarization.')

merge = subparsers.add_parser('merge', help='Merge matrices')
merge.add_argument('--incounts', dest='incounts', type=str, nargs='+', help="Location of count matrix or matrices", required=True)
merge.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
merge.add_argument('--outcounts', dest='outcounts', type=str, help="Location of count matrix or matrices", required=True)

groupcells = subparsers.add_parser('groupcells', help='Collapse cells within pre-defined groups.')
groupcells.add_argument('--incounts', dest='incounts', type=str, help="Location of count matrix", required=True)
groupcells.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
groupcells.add_argument('--outcounts', dest='outcounts', type=str, help="Location of count matrix", required=True)
groupcells.add_argument('--cellgroup', dest='cellgroup', type=str, help="Location of table defining cell to group mapping.", required=True)

subset = subparsers.add_parser('subset', help='Subset cells.')
subset.add_argument('--incounts', dest='incounts', type=str, help="Location of count matrix", required=True)
subset.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
subset.add_argument('--outcounts', dest='outcounts', type=str, help="Location of count matrix", required=True)
subset.add_argument('--subset', dest='subset', type=str, help="Location of table defining cell to group mapping.", required=True)

# score a model from scratch
fsegment = subparsers.add_parser('fit_segment', help='Fit HMM and segment genome')
fsegment.add_argument('--counts', dest='counts', nargs='+', type=str, help="Location of one or several count matrices")
fsegment.add_argument('--labels', dest='labels', nargs='*', type=str, help="Name of the countmatrix")
fsegment.add_argument('--mincount', dest='mincounts', type=int, default=0, help='Minimum number of counts per cell')
fsegment.add_argument('--maxcount', dest='maxcounts', type=int, default=sys.maxsize, help='Maximum number of counts per cell')
fsegment.add_argument('--trimcount', dest='trimcounts', type=int, default=sys.maxsize, help='Maximum number of counts per matrix element. For instance, trimcount 1 amounts to binarization.')
fsegment.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
fsegment.add_argument('--storage', dest='storage', type=str, help="Location for storing output")
fsegment.add_argument('--nstates', dest='nstates', type=int, default=20)
fsegment.add_argument('--randomseed', dest='randomseed', nargs='+', type=int, default=32, help='Random seed or list of seeds. If a list is specified, the best scoring model is kept.')
fsegment.add_argument('--niter', dest='niter', type=int, default=100, help='Number of EM iterations')
fsegment.add_argument('--n_jobs', dest='n_jobs', type=int, default=1, help='Number Jobs')
fsegment.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name')
fsegment.add_argument('--replicate', dest='replicate', type=str, default='sum', choices=['sum', 'geometric_mean', 'arithmetic_mean'], help='Model name')

# fitting a model from scratch
segment = subparsers.add_parser('segment', help='Segment genome with existing model.')
segment.add_argument('--counts', dest='counts', nargs='+', type=str, help="Location of one or several count matrices")
segment.add_argument('--labels', dest='labels', nargs='*', type=str, help="Name of the countmatrix")
segment.add_argument('--mincount', dest='mincounts', type=int, default=0, help='Minimum number of counts per cell')
segment.add_argument('--maxcount', dest='maxcounts', type=int, default=sys.maxsize, help='Maximum number of counts per cell')
segment.add_argument('--trimcount', dest='trimcounts', type=int, default=sys.maxsize, help='Maximum number of counts per matrix element. For instance, trimcount 1 amounts to binarization.')
segment.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
segment.add_argument('--storage', dest='storage', type=str, help="Location for storing output")
segment.add_argument('--nstates', dest='nstates', type=int, default=20)
segment.add_argument('--randomseed', dest='randomseed', nargs='+', type=int, default=32, help='Random seed or list of seeds. If a list is specified, the best scoring model is kept.')
segment.add_argument('--niter', dest='niter', type=int, default=100, help='Number of EM iterations')
segment.add_argument('--n_jobs', dest='n_jobs', type=int, default=1, help='Number Jobs')
segment.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name')

stats = subparsers.add_parser('make_stats', help='Obtain segmentation stats')
stats.add_argument('--counts', dest='counts', nargs='+', type=str, help="Location of one or several count matrices")
stats.add_argument('--mincount', dest='mincounts', type=int, default=0, help='Minimum number of counts per cell')
stats.add_argument('--maxcount', dest='maxcounts', type=int, default=sys.maxsize, help='Maximum number of counts per cell')
stats.add_argument('--trimcount', dest='trimcounts', type=int, default=sys.maxsize, help='Maximum number of counts per matrix element. For instance, trimcount 1 amounts to binarization.')
stats.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
stats.add_argument('--storage', dest='storage', type=str, help="Location for storing output")
stats.add_argument('--labels', dest='labels', nargs='*', type=str, help="Name of the countmatrix")
stats.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name')

#llscore = subparsers.add_parser('score', help='Print log-likelihood score')
#llscore.add_argument('--counts', dest='counts', nargs='+', type=str, help="Location of one or several count matrices")
#llscore.add_argument('--mincount', dest='mincounts', type=int, default=0, help='Minimum number of counts per cell')
#llscore.add_argument('--maxcount', dest='maxcounts', type=int, default=sys.maxsize, help='Maximum number of counts per cell')
#llscore.add_argument('--trimcount', dest='trimcounts', type=int, default=sys.maxsize, help='Maximum number of counts per matrix element. For instance, trimcount 1 amounts to binarization.')
#llscore.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
#llscore.add_argument('--storage', dest='storage', type=str, help="Location for storing output")
#llscore.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name')

seg2bed = subparsers.add_parser('seg_to_bed', help='Export segmentation to bed-file or files')
seg2bed.add_argument('--storage', dest='storage', type=str, help="Location for storing output")
seg2bed.add_argument('--individual', dest='individualbeds', action='store_true', default=False, help="Save segmentation in individual bed files.")
seg2bed.add_argument('--threshold', dest='threshold', type=float, default=0.0, help="Threshold on posterior decoding probability. "
                                                                                     "Only export state calls that exceed the posterior decoding threshold. "
                                                                                     "This allows to adjust the stringency of state calls for down-stream analysis steps.")
seg2bed.add_argument('--merge_neighbors', dest='merge_neighbors', action='store_true', default=False, help='Whether to merge neighboring bins representing the same state. Default=False.')
seg2bed.add_argument('--exclude_states', dest='exclude_states', nargs='*', type=str, help='List of state names which should be exclued.')
seg2bed.add_argument('--max_state_abundance', dest='max_state_abundance', type=float, default=1., help='Max. state abundance across the genome. '
         'This parameters allows to report only rarely occurring states. '
         'Abundant states are filtered out as they usually reflect the genomic background distribution. '
         'A good choice for this is a value that is slightly lower than 1./n_state. Default=1.')
seg2bed.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name')
seg2bed.add_argument('--output', dest='output', type=str, help='Output BED file containing the state calls.', default='')


annotate = subparsers.add_parser('annotate', help='Annotate states')
annotate.add_argument('--files', dest='files', nargs='+', type=str, help="Location of a BAM-, BIGWIG or BED-files to annotate the states with.")
annotate.add_argument('--labels', dest='labels', nargs='+', type=str, help="Annotation labels.")
#annotate.add_argument('--plot', dest='plot', help="Flag indicating whether to plot the features association.", action='store_true', default=False)
annotate.add_argument('--storage', dest='storage', type=str, help="Location for containing the pre-trained segmentation and for storing the annotated segmentation results")
annotate.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name')

#statecorrelation = subparsers.add_parser('inspect_state', help='Inspect state correlation structure')
#statecorrelation.add_argument('--stateid', dest='stateid', type=int, help="State ID to explore")
#statecorrelation.add_argument('--counts', dest='counts', nargs='+', type=str, help="Location of one or several count matrices")
#statecorrelation.add_argument('--mincount', dest='mincounts', type=int, default=0, help='Minimum number of counts per cell')
#statecorrelation.add_argument('--maxcount', dest='maxcounts', type=int, default=sys.maxsize, help='Maximum number of counts per cell')
#statecorrelation.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
#statecorrelation.add_argument('--n_cells', dest='n_cells', default=1000, type=int, help="Number of top variable cells to consider")
#statecorrelation.add_argument('--threshold', dest='threshold', type=float, default=0.9, help="Threshold on posterior decoding probability. "
#                                                                                     "Only export results that exceed the posterior decoding threshold. "
#                                                                                     "This allows to adjust the stringency of state calls for down-stream analysis steps.")
#statecorrelation.add_argument('--output', dest='output', help="Output figure path.", type=str, default='statecorrelationstructure.png')
#statecorrelation.add_argument('--storage', dest='storage', type=str, help="Location for containing the pre-trained segmentation and for storing the annotated segmentation results")
#statecorrelation.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name')

plotannotate = subparsers.add_parser('plot_annot', help='Plot annotation')
plotannotate.add_argument('--labels', dest='labels', nargs='+', type=str, help="Annotation labels.", required=True)
plotannotate.add_argument('--title', dest='title', type=str, help="Plot name.", required=True)
plotannotate.add_argument('--storage', dest='storage', type=str, help="Location for containing the pre-trained segmentation and for storing the annotated segmentation results", required=True)
plotannotate.add_argument('--threshold', dest='threshold', type=float, default=0.0, help="Threshold on posterior decoding probability. "
                                                                                     "Only export results that exceed the posterior decoding threshold. "
                                                                                     "This allows to adjust the stringency of state calls for down-stream analysis steps.")
plotannotate.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name')
plotannotate.add_argument('--plottype', dest='plottype', type=str, default='boxplot', choices=['boxplot', 'countplot', 'heatmap'], help='Plot type')
plotannotate.add_argument('--groupby', dest='groupby', type=str, help="Annotation label.", default=None)


featurecorr = subparsers.add_parser('feature_correspondence', help='Plot annotation')
featurecorr.add_argument('--inputdir', dest='inputdir', type=str, help="Input directory.")
featurecorr.add_argument('--title', dest='title', type=str, help="Title.", default=None)
featurecorr.add_argument('--storage', dest='storage', type=str, help="Location for containing the pre-trained segmentation and for storing the annotated segmentation results")
featurecorr.add_argument('--threshold', dest='threshold', type=float, default=0.0, help="Threshold on posterior decoding probability. "
                                                                                     "Only export results that exceed the posterior decoding threshold. "
                                                                                     "This allows to adjust the stringency of state calls for down-stream analysis steps.")
featurecorr.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name')

celltyping = subparsers.add_parser('celltype', help='Cell type characterization')
celltyping.add_argument('--storage', dest='storage', type=str, help="Location for containing the pre-trained segmentation and for storing the annotated segmentation results")
celltyping.add_argument('--counts', dest='counts', nargs='+', type=str, help="Location of count matrix or matrices")
celltyping.add_argument('--mincount', dest='mincounts', type=int, default=0, help='Minimum number of counts per cell')
celltyping.add_argument('--maxcount', dest='maxcounts', type=int, default=sys.maxsize, help='Maximum number of counts per cell')
celltyping.add_argument('--trimcount', dest='trimcounts', type=int, default=sys.maxsize, help='Maximum number of counts per matrix element. For instance, trimcount 1 amounts to binarization.')
celltyping.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
celltyping.add_argument('--cell_annotation', dest='cell_annotation', type=str, help='Location of a cell annotation table.')
celltyping.add_argument('--method', dest='method', type=str, default='probability', choices=['probability', 'zscore', 'logfold', 'chisqstat'])
celltyping.add_argument('--post', dest='post', help="Flag indicating whether to use posterior decoding.", action='store_true', default=False)
celltyping.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name')

enrichment = subparsers.add_parser('enrichment', help='State over-representation test')
enrichment.add_argument('--storage', dest='storage', type=str, help="Location for containing the pre-trained segmentation and for storing the annotated segmentation results")
enrichment.add_argument('--title', dest='title', type=str, help='Name of the state enrichment test. Default: "geneset"', default='geneset')
enrichment.add_argument('--features', dest='features', type=str, help='Path to a folder containing bed-files that define the feature sets.')
enrichment.add_argument('--flanking', dest='flanking', type=int, default=30000, help='Flanking window.')
enrichment.add_argument('--method', dest='method', type=str, default='chisqstat', choices=['pvalue', 'logfold', 'chisqstat'])
enrichment.add_argument('--modelname', dest='modelname', type=str, default='dirmulhmm', help='Model name')


def load_count_matrices(countfiles, bedfile, mincounts, maxcounts, trimcounts):
    data = []
    for cnt in countfiles:
        cm = CountMatrix.create_from_countmatrix(cnt, bedfile)
        cm.filter_count_matrix(mincounts, maxcounts, 0, binarize=False, trimcount=trimcounts)
    
        print(cm)
        data.append(cm)
        #cannot.append(cm.cannot)
        #data.append(cm.cmat)
    return data

def run_segmentation(data, bedfile, nstates, niter, random_states, n_jobs, mode):
    best_score = -np.inf
    scores = []
    print('Fitting {} models'.format(len(random_states)))
    for random_state in random_states:
        print("Starting {}".format(random_state))
        model = Scseg(DirMulHMM(n_components=nstates, n_iter=niter, random_state=random_state, verbose=True,
                          n_jobs=n_jobs, replicate=mode))
        model.fit(data)
        score = model.score(data)
        scores.append(score)
        if best_score < score:
            best_score = score
            best_model = model.model
            best_seed = random_state

    print('all models: seed={}, score={}'.format(random_states, scores))
    print('best model: seed={}, score={}'.format(best_seed, best_score))
    scmodel = Scseg(best_model)
    scmodel.segment(data, bedfile)
    return scmodel
    

def make_folders(output):
    os.makedirs(output, exist_ok=True)

def save_score(scmodel, data, output):
    score = scmodel.score(data)
    print('loglikelihood = {}'.format(score))
    with open(os.path.join(output, "summary", "score.txt"), 'w') as f:
        f.write('{}\n'.format(score))

def get_cell_grouping(table):
    group2cellmap = pd.read_csv(table, sep='\t')
    
    cell = group2cellmap.cells.values
    groups = group2cellmap.groups.values

    return cell, groups

def make_state_summary(model, output, labels):
    if len(labels) > 0:
        datanames = labels
    else:
        datanames = ['mat{}'.format(i) for i in range(len(model.model.emissionprobs_))]

    make_folders(os.path.join(output, 'summary'))
    model.get_state_stats().to_csv(os.path.join(output, 'summary', 'statesummary.csv'))
    model.plot_state_statistics().savefig(os.path.join(output, 'summary', 'statesummary.png'))
    model.plot_readdepth(datanames).savefig(os.path.join(output, 'summary', 'state_readdepth.png'))

def plot_normalized_emissions(model, output, labels):
    if len(labels) > 0:
        datanames = labels
    else:
        datanames = ['mat{}'.format(i) for i in range(len(model.model.emissionprobs_))]

    make_folders(os.path.join(output, 'summary'))
    for i, dataname in enumerate(datanames):
        model.plot_normalized_emissions(i).savefig(os.path.join(output, 'summary', 'emission_{}.png'.format(dataname)))

def plot_state_annotation_relationship_heatmap(model, storage, labels,
                                       title, threshold=0.0, groupby=None):

    make_folders(os.path.join(storage, 'annotation'))

    fig, ax = plt.subplots()

    segdf = model._segments[model._segments.Prob_max>=threshold].copy()

    segdf = segdf[labels].apply(zscore)

    segdf.groupby(name).agg('mean')

    sns.heatmap(segdf, cmap="RdBu_r",
                figsize=(10,20), robust=True, center=0.0, ax=ax)
    print('writing {}'.format(os.path.join(storage, 'annotation', '{}.png'.format(title))))
    fig.tight_layout()
    fig.savefig(os.path.join(storage, 'annotation', '{}.png'.format(title)))

def plot_state_annotation_relationship(model, storage, labels,
                                       title, plottype='boxplot', threshold=0.0, groupby=None):

    make_folders(os.path.join(storage, 'annotation'))

    fig, axes = plt.subplots(len(labels))

    segdf = model._segments[model._segments.Prob_max>=threshold].copy()

    for ax, label in zip(axes, labels):
        if plottype=='countplot':
            sns.countplot(y='name', hue=label,
                          data=segdf,
                          ax=ax)
        elif plottype == 'boxplot':
            segdf['log_'+label] = np.log10(segdf[label]+1)
            sns.boxplot(x='log_'+label, y='name',
                        data=segdf,
                        hue=groupby, orient='h', ax=ax)
            
        #gr = '' if groupby is None else '_'+groupby
    print('writing {}'.format(os.path.join(storage, 'annotation', '{}.png'.format(title))))
    fig.tight_layout()
    fig.savefig(os.path.join(storage, 'annotation', '{}.png'.format(title)))



def local_main(args):
    if args.program == 'bam_to_counts':

        print('Make countmatrix ...')
        cm = CountMatrix.create_from_bam(args.bamfile,
                                    args.regions, barcodetag=args.barcodetag,
                                    binsize=args.binsize, mode=args.mode)

        if args.cellgroup is not None:
            cells,  groups = get_cell_grouping(args.cellgroup)
            cm = cm.pseudobulk(cells, groups)

        cm.export_counts(args.counts)
                                      
    elif args.program == 'filter_counts':
        print('Filter counts ...')
        cm = CountMatrix.create_from_countmatrix(args.incounts, args.regions)
#        print('loaded', cm)
        cm.filter_count_matrix(args.mincounts, args.maxcounts,
                               0, binarize=False, maxcount=args.trimcounts)
#        print('exporting', cm)
        cm.export_counts(args.outcounts)

    elif args.program == 'batchannot':
        print('Adding annotation ...')
        cannot = get_cell_annotation(args.counts)
        for batch in args.batches:
            name, value = batch.split(':')
            cannot[name] = value
        write_cannot_table(args.counts, cannot)

    elif args.program == 'groupcells':
        print('Group cells (pseudobulk)...')
        cm = CountMatrix.create_from_countmatrix(args.incounts, args.regions)

        cells,  groups = get_cell_grouping(args.cellgroup)
        pscm = cm.pseudobulk(cells, groups)
        pscm.export_counts(args.outcounts)

    elif args.program == 'subset':

        print('Subset cells ...')
        cm = CountMatrix.create_from_countmatrix(args.incounts, args.regions)

        cells,  _ = get_cell_grouping(args.subset)
        pscm = cm.subset(cells)
        pscm.export_counts(args.outcounts)

    elif args.program == 'merge':
        print('Merge count matrices ...')
        cms = []
        for incount in args.incounts:
            cm = CountMatrix.create_from_countmatrix(incount, args.regions)
            cms.append(cm)

        
        merged_cm = CountMatrix.merge(cms)
        merged_cm.export_counts(args.outcounts)

    elif args.program == 'fit_segment':

        outputpath = os.path.join(args.storage, args.modelname)
        print('Segmentation ...')
        data = load_count_matrices(args.counts, args.regions,
                                               args.mincounts, args.maxcounts,
                                               args.trimcounts)

        print('fitting the hmm ...')
        scmodel = run_segmentation(data, args.regions, args.nstates,
                                   args.niter, args.randomseed,
                                   args.n_jobs, args.replicate)
        scmodel.save(outputpath)

        print('summarize results ...')
        make_state_summary(scmodel, outputpath, args.labels)
        plot_normalized_emissions(scmodel, outputpath, args.labels)
        save_score(scmodel, data, outputpath)

#    elif args.program == 'score':
#        outputpath = os.path.join(args.storage, args.modelname)
#        print('loading data ...')
#        data, cell_annot = load_count_matrices(args.counts, args.regions, args.mincounts, args.maxcounts, args.trimcounts)
#        datanames = [os.path.basename(c) for c in args.counts]
#        scmodel = Scseg.load(outputpath)
#        print('score={}'.format(scmodel.model.score(data)))

    elif args.program == 'segment':
        outputpath = os.path.join(args.storage, args.modelname)
        print('loading data ...')
        data = load_count_matrices(args.counts,
                                               args.regions,
                                               args.mincounts,
                                               args.maxcounts, args.trimcounts)
        scmodel = Scseg.load(outputpath)
        print('Run state calling ...')
        scmodel.segment(data, args.regions)
        make_state_summary(scmodel, outputpath, args.labels)
        plot_normalized_emissions(scmodel, outputpath, args.labels)
        save_score(scmodel, data, outputpath)

#    elif args.program == 'make_stats':
#        outputpath = os.path.join(args.storage, args.modelname)
##        print('loading data ...')
##        data = load_count_matrices(args.counts,
##                                               args.regions,
##                                               args.mincounts,
##                                               args.maxcounts, args.trimcounts)
#        #datanames = [os.path.basename(c) for c in args.counts]
#        scmodel = Scseg.load(outputpath)
#        print('summarize results ...')
#        make_state_summary(scmodel, outputpath, args.labels)

    elif args.program == 'seg_to_bed':
        outputpath = os.path.join(args.storage, args.modelname)

        scmodel = Scseg.load(outputpath)

        # select query states
        query_states = ['state_{}'.format(i) for i, p in enumerate(scmodel.model.get_stationary_distribution()) \
                        if p<=args.max_state_abundance]

        if args.exclude_states is not None:
            query_states = list(set(query_states).difference(set(args.exclude_states)))

        print("exporting {} states".format(len(query_states)))
        # subset and merge the state calls
        subset = scmodel.get_statecalls(query_states, collapse_neighbors=args.merge_neighbors,
                                        state_prob_threshold=args.threshold)

        if args.output == '':
            output = outputpath = os.path.join(args.storage, args.modelname, 'summary', 'segments.bed')
        else:
            output = args.output
        # export the state calls as a bed file
        export_bed(subset, output,
                   individual_beds=args.individualbeds)

    elif args.program == 'annotate':
        outputpath = os.path.join(args.storage, args.modelname)
        scmodel = Scseg.load(outputpath)

        assert len(args.labels) == len(args.files), "Number of files and labels mismatching"
        print('annotate states ...')
        files = {key: filename for key, filename in zip(args.labels, args.files)}
        scmodel.annotate(files)

        scmodel.save(outputpath)
        
    elif args.program == 'plot_annot':
        outputpath = os.path.join(args.storage, args.modelname)
        print('plot annotation ...')
        scmodel = Scseg.load(outputpath)

        if args.plottype == 'heatmap':
            plot_state_annotation_relationship_heatmap(scmodel, outputpath,
                                           args.labels, args.title,
                                           args.threshold, args.groupby)
        else:
            plot_state_annotation_relationship(scmodel, outputpath,
                                           args.labels, args.title, args.plottype,
                                           args.threshold, args.groupby)

#    elif args.program == 'celltype':
#        outputpath = os.path.join(args.storage, args.modelname)
#        
#        outputcelltyping = os.path.join(outputpath, 'celltyping')
#
#        print('celltyping ...')
#        scmodel = Scseg.load(outputpath)
#
#        data, celllabels = load_count_matrices(args.counts, args.regions, args.mincounts, args.maxcounts, args.trimcounts)
#        datanames = [os.path.basename(c) for c in args.counts]
#
#        assoc = scmodel.cell2state_enrichment(data, mode=args.method, post=args.post)
#        method = args.method
#
#        make_folders(outputcelltyping)
#        for i, folds in enumerate(assoc):
#            sns.clustermap(folds, cmap="Blues", robust=True).savefig(os.path.join(outputcelltyping,
#                       'cellstate_heatmap_{}_{}.png'.format(method, datanames[i])))
#            df = pd.DataFrame(folds, columns=[scmodel.to_statename(i) for i in range(scmodel.n_components)],
#                              index=celllabels[i].cell)
#            df.to_csv(os.path.join(outputcelltyping, 'cell2state_{}.csv'.format(method)))
#
#        tot_assoc = np.concatenate(assoc, axis=0)
#        embedding = UMAP().fit_transform(tot_assoc)
#
#        merged_celllabels = pd.concat(celllabels, axis=0, ignore_index=True)
#
#        df = pd.DataFrame(embedding, columns=["X","Y"])
#        df = pd.concat([df, merged_celllabels], axis=1)
#        
#        for label in merged_celllabels.columns:
#            if label == 'cell':
#                continue
#            fig, ax = plt.subplots()
#            sns.scatterplot(x='X', y='Y', ax=ax, hue=label, data=df, alpha=.1)
#            fig.savefig(os.path.join(outputcelltyping, 'cellstate_umap_{}_color{}.png'.format(method, label)))
#        
#        for label in merged_celllabels.columns:
#            if label == 'cell':
#                continue
#            g = sns.FacetGrid(df, col=label)
#            g = g.map(sns.scatterplot, "X", "Y",
#                      edgecolor='w',
#                      **{'alpha':.2}).add_legend().savefig(os.path.join(outputcelltyping,
#                                                                        'cellstate_umap_{}_facet{}.png'.format(method, label)))
#        for i in np.arange(scmodel.n_components):
#            fig, ax = plt.subplots()
#            sns.scatterplot(x='X', y='Y', ax=ax, data=df,
#                            hue=tot_assoc[:, i],
#                            alpha=.1, hue_norm=(0, tot_assoc.max()),
#                            cmap='Blues')
#            fig.savefig(os.path.join(outputcelltyping,
#                                     'cellstate_umap_{}_{}.png'.format(method, scmodel.to_statename(i))))
#
#        fig, ax = plt.subplots()
#        sns.scatterplot(x='X', y='Y', ax=ax, data=df, alpha=.1)
#        fig.savefig(os.path.join(outputcelltyping, 'cellstate_umap_{}.png'.format(method)))
#
#        df.to_csv(os.path.join(outputcelltyping, 'umap_{}.csv'.format(method)))
#
    elif args.program == 'enrichment':
        outputpath = os.path.join(args.storage, args.modelname)

        outputenr = os.path.join(outputpath, 'annotation')

        print('enrichment analysis')
        scmodel = Scseg.load(outputpath)
        make_folders(outputenr)
 
        featuresets = glob.glob(os.path.join(args.features, '*.bed'))
        featurenames = [os.path.basename(name)[:-4] for name in featuresets]
        obs, lens, _ = scmodel.geneset_observed_state_counts(featuresets, flanking=args.flanking)

        enr = scmodel.broadregion_enrichment(obs, lens, featurenames, mode=args.method)

        if args.method == 'logfold':
            g = sns.clustermap(enr, cmap="RdBu_r", figsize=(20,30), robust=True, **{'center':0.0, 'vmin':-1.5, 'vmax':1.5})
        elif args.method == 'chisqstat':
            g = sns.clustermap(enr, cmap="Reds", figsize=(20,30), robust=True)
        g.savefig(os.path.join(outputenr, "state_enrichment_{}_{}.png".format(args.method, args.title)))

#    elif args.program == 'feature_correspondence':
#        outputpath = os.path.join(args.storage, args.modelname)
#        print('correspondence analysis')
#        scmodel = Scseg.load(outputpath)
#        make_folders(os.path.join(outputpath, 'annotation'))
#        
#        beds = glob.glob(os.path.join(args.inputdir, '*.bed'))
#        bnames = [os.path.basename(bed) for bed in beds]
#        x = np.zeros((scmodel.n_components, len(beds)))
#
#        for i in range(scmodel.n_components):
#            state = scmodel.to_statename(i)
#            segments = scmodel._segments[(scmodel._segments.name == state) & (scmodel._segments.Prob_max >= args.threshold)]
#            a = BedTool([Interval(row.chrom, row.start, row.end) for _, row in segments.iterrows()]).sort().merge()
#            for j, bed in enumerate(beds):
#                b = BedTool(bed).sort()
#                x[i,j] = a.jaccard(b)['jaccard']
#
#        df = pd.DataFrame(x, index=['state_{}'.format(i) for i in range(scmodel.n_components)],
#                          columns=bnames)
#
#        fig, ax = plt.subplots(figsize=(10,10))
#        sns.heatmap(df, cmap="Blues") 
#        fig.tight_layout()
#        fig.savefig(os.path.join(outputpath,  'annotation', args.title + '_state_heatmap.png'))
#
def main():
    args = parser.parse_args()
    local_main(args)

if __name__ == '__main__':

    main()
