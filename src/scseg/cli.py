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
import glob
from scseg.countmatrix import CountMatrix
from scseg.countmatrix import get_cell_annotation_first_row_
from scseg.countmatrix import write_cannot_table
from scseg.countmatrix import make_counting_bins
from scseg.countmatrix import sparse_count_reads_in_regions
from scseg.hmm import MultiModalMultinomialHMM as MultinomialHMM
from scseg.hmm import MultiModalDirMulHMM as DirMultinomialHMM
from scseg.hmm import MultiModalMixHMM as MixMultinomialHMM
from scseg import Scseg
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from umap import UMAP
import numpy as np
from pybedtools import BedTool
from pybedtools import Interval
import argparse

parser = argparse.ArgumentParser(description='Scseg - single-cell genome segmentation.')
#parser.add_help()
#subparsers = parser.add_subparsers(dest='segment', help='segmentation')

subparsers = parser.add_subparsers(dest='program')

counts = subparsers.add_parser('bam_to_counts', help='Make countmatrix')
counts.add_argument('--bamfile', dest='bamfile', type=str, help="Location of a bamfile", required=True)
counts.add_argument('--binsize', dest='binsize', type=int, help="Binsize", default=1000)
counts.add_argument('--barcodetag', dest='barcodetag', type=str, help="Barcode readtag", default='CB')
counts.add_argument('--counts', dest='counts', type=str, help="Location of count matrix or matrices", required=True)
counts.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)

batchannot = subparsers.add_parser('batchannot', help='Add batch annotation')
batchannot.add_argument('--counts', dest='counts', type=str, help="Location of count matrix", required=True)
batchannot.add_argument('--batches', dest='batches', type=str, nargs='+', help="Batch name:value pairs to attach", required=True)

filtering = subparsers.add_parser('filter_counts', help='Filter countmatrix to remove poor quality cells')
filtering.add_argument('--incounts', dest='incounts', type=str, help="Location of input count matrix", required=True)
filtering.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
filtering.add_argument('--outcounts', dest='outcounts', type=str, help="Location of output count matrix", required=True)
filtering.add_argument('--mincounts', dest='mincounts', type=int, default=0, help='Minimum number of counts per cell')
filtering.add_argument('--maxcounts', dest='maxcounts', type=int, default=30000, help='Maximum number of counts per cell')

merge = subparsers.add_parser('merge', help='Filter countmatrix to remove poor quality cells')
merge.add_argument('--incounts', dest='incounts', type=str, nargs='+', help="Location of count matrix or matrices", required=True)
merge.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
merge.add_argument('--outcounts', dest='outcounts', type=str, help="Location of count matrix or matrices", required=True)

stats = subparsers.add_parser('make_stats', help='Obtain segmentation stats')
stats.add_argument('--counts', dest='counts', nargs='+', type=str, help="Location of one or several count matrices")
stats.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
stats.add_argument('--storage', dest='storage', type=str, help="Location for storing output")
stats.add_argument('--labels', dest='labels', nargs='*', type=str, help="Name of the countmatrix")

segment = subparsers.add_parser('segment', help='Segment genome')
segment.add_argument('--counts', dest='counts', nargs='+', type=str, help="Location of one or several count matrices")
segment.add_argument('--labels', dest='labels', nargs='*', type=str, help="Name of the countmatrix")
segment.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
segment.add_argument('--storage', dest='storage', type=str, help="Location for storing output")
segment.add_argument('--nstates', dest='nstates', type=int, default=20)
segment.add_argument('--randomseed', dest='randomseed', type=int, default=32, help='Random seed')
segment.add_argument('--niter', dest='niter', type=int, default=100, help='Number of EM iterations')
segment.add_argument('--n_jobs', dest='n_jobs', type=int, default=1, help='Number Jobs')
segment.add_argument('--meth', dest='meth', type=str, default='mul', choices=['mix','mul', 'dirmul'], help='multinomialhmm or mixhmm')

seg2bed = subparsers.add_parser('seg_to_bed', help='Export segmentation to bed-file or files')
seg2bed.add_argument('--storage', dest='storage', type=str, help="Location for storing output")
seg2bed.add_argument('--individual', dest='individualbeds', action='store_true', default=False, help="Save segmentation in individual bed files.")
seg2bed.add_argument('--threshold', dest='threshold', type=float, default=0.0, help="Threshold on posterior decoding probability. "
                                                                                     "Only export results that exceed the posterior decoding threshold. "
                                                                                     "This allows to adjust the stringency of state calls for down-stream analysis steps.")


annotate = subparsers.add_parser('annotate', help='Annotate states')
annotate.add_argument('--files', dest='files', nargs='+', type=str, help="Location of a BAM-, BIGWIG or BED-files to annotate the states with.")
annotate.add_argument('--labels', dest='labels', nargs='+', type=str, help="Annotation labels.")
annotate.add_argument('--plot', dest='plot', help="Flag indicating whether to plot the features association.", action='store_true', default=False)
annotate.add_argument('--storage', dest='storage', type=str, help="Location for containing the pre-trained segmentation and for storing the annotated segmentation results")

plotannotate = subparsers.add_parser('plot_annot', help='Plot annotation')
plotannotate.add_argument('--labels', dest='labels', nargs='+', type=str, help="Annotation labels.")
plotannotate.add_argument('--groupby', dest='groupby', type=str, help="Annotation label.", default=None)
plotannotate.add_argument('--storage', dest='storage', type=str, help="Location for containing the pre-trained segmentation and for storing the annotated segmentation results")
plotannotate.add_argument('--threshold', dest='threshold', type=float, default=0.0, help="Threshold on posterior decoding probability. "
                                                                                     "Only export results that exceed the posterior decoding threshold. "
                                                                                     "This allows to adjust the stringency of state calls for down-stream analysis steps.")

featurecorr = subparsers.add_parser('feature_correspondence', help='Plot annotation')
featurecorr.add_argument('--inputdir', dest='inputdir', type=str, help="Input directory.")
featurecorr.add_argument('--title', dest='title', type=str, help="Title.", default=None)
featurecorr.add_argument('--storage', dest='storage', type=str, help="Location for containing the pre-trained segmentation and for storing the annotated segmentation results")
featurecorr.add_argument('--threshold', dest='threshold', type=float, default=0.0, help="Threshold on posterior decoding probability. "
                                                                                     "Only export results that exceed the posterior decoding threshold. "
                                                                                     "This allows to adjust the stringency of state calls for down-stream analysis steps.")

celltyping = subparsers.add_parser('celltype', help='Cell type characterization')
celltyping.add_argument('--storage', dest='storage', type=str, help="Location for containing the pre-trained segmentation and for storing the annotated segmentation results")
celltyping.add_argument('--counts', dest='counts', nargs='+', type=str, help="Location of count matrix or matrices")
celltyping.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
celltyping.add_argument('--cell_annotation', dest='cell_annotation', type=str, help='Location of a cell annotation table.')
celltyping.add_argument('--method', dest='method', type=str, default='probability', choices=['probability', 'zscore', 'logfold', 'chisqstat'])
celltyping.add_argument('--post', dest='post', help="Flag indicating whether to use posterior decoding.", action='store_true', default=False)

enrichment = subparsers.add_parser('enrichment', help='State over-representation test')
enrichment.add_argument('--storage', dest='storage', type=str, help="Location for containing the pre-trained segmentation and for storing the annotated segmentation results")
enrichment.add_argument('--title', dest='title', type=str, help='Name of the state enrichment test. Default: "geneset"', default='geneset')
enrichment.add_argument('--features', dest='features', type=str, help='Path to a folder containing bed-files that define the feature sets.')
enrichment.add_argument('--flanking', dest='flanking', type=int, default=30000, help='Flanking window.')
enrichment.add_argument('--method', dest='method', type=str, default='chisqstat', choices=['pvalue', 'logfold', 'chisqstat'])


#parser.add_argument('program', type=str, help='Program name', choices=['make_countmatrix', 'segment', 'annotate', 'enrich'])
#parser_a = subparsers.add_parser('command_a', help = "command_a help")
## Setup options for parser_a
#parser.add_argument('--bamfile', dest='bamfile', type=str, help="Location of a bamfile")
#parser.add_argument('--binsize', dest='binsize', type=int, help="Binsize")
#parser.add_argument('--barcodetag', dest='barcodetag', type=str, help="Barcode readtag", default='CB')
##parser.add_argument('--counts', dest='counts', type=str, help="Location of a bamfile")
#
#parser.add_argument('--counts', dest='counts', nargs='+', type=str, help="Location of count matrix or matrices")
#parser.add_argument('--regions', dest='regions', type=str, help="Location of regions in bed format", required=True)
#
#parser.add_argument('--output', dest='output', type=str, help="Location for storing output")
#parser.add_argument('--nstates', dest='nstates', type=int, default=20)


args = parser.parse_args()
print(args)
    
#from scseg

def load_count_matrices(countfiles, bedfile):
    data = []
    cannot = []
    for cnt in countfiles:
        cm = CountMatrix.create_from_countmatrix(cnt, bedfile)
    
        print(cm)
        cannot.append(cm.cannot)
        data.append(cm.cmat)
    return data, cannot

def run_segmentation(data, bedfile, nstates, niter, random_state, n_jobs, meth):
    if meth == 'mix':
        model = MixMultinomialHMM(n_components=nstates, n_iter=niter, random_state=random_state, verbose=True,
                               n_jobs=n_jobs)
    elif meth == 'dirmul':
        model = DirMultinomialHMM(n_components=nstates, n_iter=niter, random_state=random_state, verbose=True,
                               n_jobs=n_jobs)
    else:
        model = MultinomialHMM(n_components=nstates, n_iter=niter, random_state=random_state, verbose=True,
                               n_jobs=n_jobs)
    model.fit(data)
    scmodel = Scseg(model)
    scmodel.segment(data, bedfile)
    return scmodel
    

def make_folders(output):
    os.makedirs(output, exist_ok=True)

def make_state_summary(model, output, labels):
    if len(labels) > 0:
        datanames = labels
    else:
        datanames = ['mat{}'.format(i) for i in range(len(model.model.emissionprobs_))]

    make_folders(os.path.join(output, 'summary'))
    model.plot_state_statistics().savefig(os.path.join(output, 'summary', 'statesummary.png'))
    model.plot_readdepth(datanames).savefig(os.path.join(output, 'summary', 'state_readdepth.png'))

def plot_state_annotation_relationship(model, storage, labels, threshold=0.0, groupby=None):
    make_folders(os.path.join(storage, 'annotation'))
    for label in labels:
        fig, ax = plt.subplots()
        if len(model._segments[label].unique()) == 2:
            sns.countplot(y='name', hue=label, data=model._segments[model._segments.Prob_max>=threshold], ax=ax)
        else:
            model._segments['log_'+label] = np.log10(model._segments[label]+1)
            sns.boxplot(x='log_'+label, y='name', data=model._segments[model._segments.Prob_max>=threshold], hue=groupby, orient='h', ax=ax)
        gr = '' if groupby is None else '_'+groupby
        print('writing {}'.format(os.path.join(storage, 'annotation', label + gr+'_relation.png')))
        fig.tight_layout()
        fig.savefig(os.path.join(storage, 'annotation', label + gr+'_relation.png'))

def main():
    if args.program == 'bam_to_counts':

        print('make countmatrix ...')
        make_counting_bins(args.bamfile, args.binsize, args.regions)
        sparse_count_reads_in_regions(args.bamfile, args.regions,
                                      args.counts, args.barcodetag)
                                      
    elif args.program == 'filter_counts':
        print('filter counts ...')
        cm = CountMatrix.create_from_countmatrix(args.incounts, args.regions)
        print('loaded', cm)
        cm.filter_count_matrix(args.mincounts, args.maxcounts, 0, binarize=False)
        print('exporting', cm)
        cm.export_counts(args.outcounts)

    elif args.program == 'batchannot':
        cannot = get_cell_annotation_first_row_(args.counts)
        for batch in args.batches:
            name, value = batch.split(':')
            cannot[name] = value
        write_cannot_table(args.counts, cannot)

    elif args.program == 'merge':
        print('merge count matrices ...')
        cms = []
        for incount in args.incounts:
            cm = CountMatrix.create_from_countmatrix(incount, args.regions)
            cms.append(cm)

        
        #merged_cannot = ['m{}_{}'.format(i, a) for i, cm in enumerate(cms) for a in cm.cannot]
        merged_cm = CountMatrix.merge(cms)
       # merged_cm = CountMatrix(hstack([cm.cmat for cm in cms]), cm.regions, merged_cannot)
        print(merged_cm)
        merged_cm.export_counts(args.outcounts)

    elif args.program == 'segment':

        print('segmentation ...')
        print('loading data ...')
        data, cell_annot = load_count_matrices(args.counts, args.regions)

        print('fitting the hmm ...')
        scmodel = run_segmentation(data, args.regions, args.nstates, args.niter, args.randomseed, args.n_jobs, args.meth)
        #scmodel = Scseg.load(args.storage)
        #scmodel.segment(data, args.regions)

        scmodel.save(args.storage)

        print('summarize results ...')
        make_state_summary(scmodel, args.storage, args.labels)

    elif args.program == 'make_stats':
        print('loading data ...')
        data, cell_annot = load_count_matrices(args.counts, args.regions)
        datanames = [os.path.basename(c) for c in args.counts]
        scmodel = Scseg.load(args.storage)
        print('summarize results ...')
        make_state_summary(scmodel, args.storage, args.labels)
        print('loglikelihood = {}'.format(scmodel.model.score(data)))

    elif args.program == 'seg_to_bed':

        print("export segmentation as bed")
        scmodel = Scseg.load(args.storage)
        scmodel.export_bed(os.path.join(args.storage, 'beds', 'segments{}').format('' if args.threshold <= 0.0 else '_{}'.format(args.threshold)),
                           individual_beds=args.individualbeds,
                           prob_max_threshold=args.threshold)

    elif args.program == 'annotate':
        scmodel = Scseg.load(args.storage)

        print('annotate states ...')
        files = {key: filename for key, filename in zip(args.labels, args.files)}
        scmodel.annotate(files)

        print('save annotated segmentation')
        scmodel.save(args.storage)
        if args.plot:
            plot_state_annotation_relationship(scmodel, args.storage, args.labels, 0.0, None)
        
    elif args.program == 'plot_annot':
        print('plot annotation ...')
        scmodel = Scseg.load(args.storage)

        plot_state_annotation_relationship(scmodel, args.storage, args.labels, args.threshold, args.groupby)
        
    elif args.program == 'celltype':
        
        print('celltyping ...')
        scmodel = Scseg.load(args.storage)

        data, celllabels = load_count_matrices(args.counts, args.regions)
        datanames = [os.path.basename(c) for c in args.counts]
        #for cell in celllabels:

        assoc = scmodel.cell2state_enrichment(data, mode=args.method, post=args.post)
        method = args.method

        make_folders(os.path.join(args.storage, 'celltyping'))
        for i, folds in enumerate(assoc):
            sns.clustermap(folds, cmap="Blues", robust=True).savefig(os.path.join(args.storage,
                       'celltyping', 'cellstate_heatmap_{}_{}.png'.format(method, datanames[i])))
            print(folds.shape, scmodel.n_components, celllabels[i].shape, celllabels[i].head())
            df = pd.DataFrame(folds, columns=[scmodel.to_statename(i) for i in range(scmodel.n_components)],
                              index=celllabels[i].cell)
            df.to_csv(os.path.join(args.storage, 'celltyping', 'cell2state_{}.csv'.format(method)))

        tot_assoc = np.concatenate(assoc, axis=0)
        embedding = UMAP().fit_transform(tot_assoc)

        merged_celllabels = pd.concat(celllabels, axis=0, ignore_index=True)

        df = pd.DataFrame(embedding, columns=["X","Y"])
        df = pd.concat([df, merged_celllabels], axis=1)
        
        for label in merged_celllabels.columns:
            if label == 'cell':
                continue
            fig, ax = plt.subplots()
            sns.scatterplot(x='X', y='Y', ax=ax, hue=label, data=df, alpha=.1)
            fig.savefig(os.path.join(args.storage, 'celltyping', 'cellstate_umap_{}_color{}.png'.format(method, label)))
        
        for label in merged_celllabels.columns:
            if label == 'cell':
                continue
            g = sns.FacetGrid(df, col=label)
            g = g.map(sns.scatterplot, "X", "Y",
                      edgecolor='w',
                      **{'alpha':.2}).add_legend().savefig(os.path.join(args.storage,
                                                                        'celltyping',
                                                                        'cellstate_umap_{}_facet{}.png'.format(method, label)))
        for i in np.arange(scmodel.n_components):
            fig, ax = plt.subplots()
            sns.scatterplot(x='X', y='Y', ax=ax, data=df,
                            hue=tot_assoc[:, i],
                            alpha=.1, hue_norm=(0, tot_assoc.max()),
                            cmap='Blues')
            fig.savefig(os.path.join(args.storage, 'celltyping',
                                     'cellstate_umap_{}_{}.png'.format(method, scmodel.to_statename(i))))

        fig, ax = plt.subplots()
        sns.scatterplot(x='X', y='Y', ax=ax, data=df, alpha=.1)
        fig.savefig(os.path.join(args.storage, 'celltyping', 'cellstate_umap_{}.png'.format(method)))

        df.to_csv(os.path.join(args.storage, 'celltyping', 'umap_{}.csv'.format(method)))

    elif args.program == 'enrichment':

        print('enrichment analysis')
        scmodel = Scseg.load(args.storage)
        make_folders(os.path.join(args.storage, 'enrichment'))
 
        featuresets = glob.glob(os.path.join(args.features, '*.bed'))
        featurenames = [os.path.basename(name)[:-4] for name in featuresets]
        obs, lens, _ = scmodel.geneset_observed_state_counts(featuresets, flanking=args.flanking)

        enr = scmodel.broadregion_enrichment(obs, lens, featurenames, mode=args.method)

        if args.method == 'logfold':
            g = sns.clustermap(enr, cmap="RdBu_r", figsize=(10,20), robust=True, **{'center':0.0, 'vmin':-1.5, 'vmax':1.5})
        elif args.method == 'chisqstat':
            g = sns.clustermap(enr, cmap="Reds", figsize=(10,20), robust=True)
        g.savefig(os.path.join(args.storage, "enrichment", "state_enrichment_{}_{}.png".format(args.method, args.title)))

    elif args.program == 'feature_correspondence':
        print('correspondence analysis')
        scmodel = Scseg.load(args.storage)
        make_folders(os.path.join(args.storage, 'correspondence'))
        
        beds = glob.glob(os.path.join(args.inputdir, '*.bed'))
        sorted(beds)
        bnames = [os.path.basename(bed) for bed in beds]
        x = np.zeros((scmodel.n_components, len(beds)))

        for i in range(scmodel.n_components):
            state = scmodel.to_statename(i)
            print('processing '+state)
            segments = scmodel._segments[(scmodel._segments.name == state) & (scmodel._segments.Prob_max >= args.threshold)]
            a = BedTool([Interval(row.chrom, row.start, row.end) for _, row in segments.iterrows()]).sort().merge()
            for j, bed in enumerate(beds):
                b = BedTool(bed).sort()
                x[i,j] = a.jaccard(b)['jaccard']

        df = pd.DataFrame(x, index=['state_{}'.format(i) for i in range(scmodel.n_components)],
                          columns=bnames)

        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(df, cmap="Blues") 
        fig.tight_layout()
        fig.savefig(os.path.join(args.storage,  'correspondence', args.title + '_state_heatmap.png'))

if __name__ == '__main__':

    main(args=args)
