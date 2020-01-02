import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import seaborn as sns
import tempfile
from pybedtools import BedTool
from pybedtools import Interval
from pybedtools.helpers import cleanup
from .hmm import MultiModalMultinomialHMM
from .hmm import MultiModalMixHMM
from .hmm import MultiModalDirMulHMM
from scipy.stats import binom
from scipy.stats import norm
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import hstack
from scipy.stats import zscore
from scipy.sparse import issparse as is_sparse
import pickle
import os
import gzip
from janggu.data import Cover
from janggu.data import LogTransform
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import hypergeom
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import check_random_state
from scipy.special import logsumexp
from hmmlearn.utils import normalize
from sklearn.cluster import AgglomerativeClustering

from numba import jit, njit
from numba import prange
from operator import mod, floordiv, pow

@njit(parallel=True)
def faster_fft(c1, c2, ncomp, maxlen):
    """
    computes P(c1 + c2, current_state| prev_state) = sum_{middle_state} P(c1, current_state | middle_state) * P(c2, middle_state | prev_state)
    in Fourier space
    """

    fft_cnt_dist = np.zeros((maxlen+1, ncomp, ncomp, ncomp), dtype=np.complex128)
    for i in prange(pow(ncomp, 3)):
        do_i = mod(i, ncomp)
        curr_i = mod(floordiv(i, ncomp), ncomp)
        prev_i = mod(floordiv(i, ncomp*ncomp), ncomp)

        for ic in range(maxlen+1):
            for middle_i in range(ncomp):
                fft_cnt_dist[ic, do_i, prev_i, curr_i] += c1[ic, do_i, prev_i, middle_i] * c2[ic, do_i, middle_i, curr_i]
    return fft_cnt_dist

class Scseg(object):

    _enr = None
    _cnt_storage = {}
    _cnt_conditional = {}

    def __init__(self, model):
        self._segments = None
        self._nameprefix = 'state_'
        self.model = model
        self._color = {name: el for name, el in \
                       zip(self.to_statenames(np.arange(self.n_components)), sns.color_palette('bright', self.n_components))}

    def save(self, path):
        """
        saves current model parameters
        """
        self.model.save(os.path.join(path, 'modelparams'))

        if hasattr(self, "_segments") and self._segments is not None:
            self.export_segmentation(os.path.join(path, 'segments',
                                     'segmentation.tsv'), 0.0)

    @classmethod
    def load(cls, path):
        """
        loads model parameters from path
        """
        if os.path.exists(os.path.join(path, 'modelparams', 'hmm.npz')):
            model = MultiModalMultinomialHMM.load(path)
        elif os.path.exists(os.path.join(path, 'modelparams', 'dirmulhmm.npz')):
            model = MultiModalDirMulHMM.load(path)
        elif os.path.exists(os.path.join(path, 'modelparams', 'mixhmm.npz')):
            model = MultiModalMixHMM.load(path)
        else:
            raise ValueError("Model not available")
        scmodel = cls(model)

        scmodel.load_segments(os.path.join(path, 'segments', 'segmentation.tsv'))

        return scmodel

    def load_segments(self, path):
        if os.path.exists(path):
            self._segments = pd.read_csv(path, sep='\t')

    def to_statenames(self, states):
        """
        converts list of state ids (integer) to list of state names (str)
        """
        return [self.to_statename(state) for state in states]

    def to_statename(self, state):
        """
        converts state id (integer) to state name (str)
        """
        return '{}{}'.format(self._nameprefix, state)

    def to_stateid(self, statename):
        """
        converts state name (str) to state id (integer)
        """

        return int(statename[len(self._nameprefix):])

    def cell2state_counts(self, data, prob_max_threshold=0.0, post=False):
        """ Determines whether a states is overrepresented among
        the accessible sites in a given cellself.

        The P-value is determined using the binomial test
        and the log-fold-change is determine by Obs[state proportion]/Expected[state proportion].
        """

        #stateprob = self.model.get_stationary_distribution()

        if post:
            # use posterior decoding
            print('use post')
            _, statescores = self.model.score_samples(data)
            Z = statescores.T
        else:
            states = self.model.predict(data)

            values = np.ones(len(states))
            values[self._segments.Prob_max < prob_max_threshold] = 0

            Z = csc_matrix((values,
                           (states.astype('int'), np.arange(len(states),dtype='int'))))

        enrs = []

        for d in data:
            d_ = d.copy()
            d_[d_>0] = 1

            # adding a pseudo-count of 1 to avoid division by zero
            #obs_seqfreq = Z.dot(d_).toarray().T
            if is_sparse(Z):
                obs_seqfreq = Z.dot(d_).toarray().T
            else:
                obs_seqfreq = d_.T.dot(Z.T)

            mat = obs_seqfreq

            enrs.append(mat)

        return enrs

    def cell2state(self, data, mode='logfold', prob_max_threshold=0.0, post=False):
        """ Determines whether a states is overrepresented among
        the accessible sites in a given cellself.

        The P-value is determined using the binomial test
        and the log-fold-change is determine by Obs[state proportion]/Expected[state proportion].
        """

        obs_seqfreqs = self.cell2state_counts(data, prob_max_threshold, post)

        enrs = []

        for obs_seqfreq in obs_seqfreqs:

            if mode == 'probability':
                mat = obs_seqfreq / obs_seqfreq.sum(1, keepdims=True)
            elif mode == 'raw':
                mat = obs_seqfreq
            elif mode == 'zscore':
                mat = zscore(obs_seqfreq, axis=1)
            elif mode in ['fold', 'logfold', 'chisqstat']:
                mat = self.broadregion_enrichment(obs_seqfreq, obs_seqfreq.sum(1), mode=mode)
            else:
                raise ValueError('{} not supported for cell2state association'.format(mode))

            enrs.append(mat)

        return enrs

    def cell2state_umap(self, cell2states):
        x = UMAP().fit_transform(cell2states)
        return pd.DataFrame(x, columns=['dim1', 'dim2'])

    def plot_umap(self, cellstates):
        return sns.scatterplot(x='dim1', y='dim2', data=cellstates)

    def get_state_stats(self):
        if self._segments is None:
            raise ValueError("No segmentation results available. Please run segment(data, regions) first.")

        state_counts = pd.Series(self._segments.name).value_counts()
        return state_counts

    def plot_state_statistics(self):
        """
        plot state statistics
        """

        fig, ax = plt.subplots(1, 2)
        state_counts = self.get_state_stats()

        sns.barplot(y=[l for l in state_counts.index],
                    x=state_counts,
                    ax=ax[0],
                    palette=[self.color[i] for i in state_counts.index])

        sns.heatmap(self.model.transmat_, ax=ax[1], cmap='Reds')

        return fig

    def plot_readdepth(self, labels):
        """
        plots read depths associated with states
        """
        fig, axes = plt.subplots(1,len(self.model.n_features))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        segs = self._segments.copy()
        for i, ax in enumerate(axes):
            segs['log_readdepth_{}'.format(labels[i])] = np.log10(segs['readdepth_'+str(i)] + 1)
        for i, ax in enumerate(axes):
            sns.boxplot(x="log_readdepth_{}".format(labels[i]), y='name', data=segs, orient='h', ax=ax)
        fig.tight_layout()
        return fig

    def plot_logfolds_dist(self, logfoldenr, query_state=None):
        """
        plots log fold distribution
        """
        fig, axes = plt.subplots(len(logfoldenr))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for ax, fold in zip(axes, logfoldenr):

            if query_state is not None:
                #istate = self.to_stateid(query_state)
                #fold = fold[istate]
                fold = fold[query_state]

            ff = fold.values.flatten()
            sns.distplot(ff, ax=ax)

            x = np.linspace(-1.5, 2.5)
            scale = np.sqrt(np.mean(np.square(ff[ff<0.0])))
            ax.plot(x, norm(0, scale).pdf(x), 'R')

        return fig

    @property
    def color(self):
        return self._color

    def segment(self, data, regions):
        """
        performs segmentation.

        Parameters
        ----------
        data : list(np.array) or list(scipy.sparse.csc_matrix)
            List of count matrices
        regions : pd.DataFrame
            Dataframe containing the genomic intervals (e.g. from a bed file).
        """

        bed = BedTool(regions)

        regions_ = pd.DataFrame([[iv.chrom, iv.start, iv.end] for iv in bed],
                                columns=['chrom', 'start', 'end'])

        statenames = self.to_statenames(self.model.predict(data))
        _, statescores = self.model.score_samples(data)

        regions_['name'] = statenames
        regions_['strand'] = '.'
        regions_['thickStart'] = regions_.start
        regions_['thickEnd'] = regions_.end
        regions_['itemRbg'] = ['{},{},{}'.format(int(self.color[state][0]*255),
                                                int(self.color[state][1]*255),
                                                int(self.color[state][2]*255)) \
                               for state in statenames]

        for istate, statename in enumerate(self.to_statenames(np.arange(self.n_components))):
            regions_['Prob_' + statename] = statescores[:, istate]

        regions_['Prob_max'] = statescores.max(1)
        regions_['score'] = 1000*regions_['Prob_max']
        regions_['score'] = regions_['score'].astype('int')
        for i in range(len(data)):
            regions_['readdepth_' + str(i)] = data[i].sum(1)
        self._segments = regions_
        cleanup()


    def export_segmentation(self, filename, prob_max_threshold=0.99):
        """
        Exports the segmentation results table.
        """

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if self._segments is None:
            raise ValueError("No segmentation results available. Please run segment(data, regions) first.")

        self._segments[(self._segments.Prob_max >= prob_max_threshold)].to_csv(filename, sep='\t', index=False)


    def export_bed(self, dirname, individual_beds=False, prob_max_threshold=0.99):
        """
        Exports the segmentation results in bed format.
        """

        os.makedirs(dirname, exist_ok=True)

        if self._segments is None:
            raise ValueError("No segmentation results available. Please run segment(data, regions) first.")

        if individual_beds:
            self._single_bed = []
            for comp in range(self.n_components):
                self._segments[(self._segments.name == self.to_statename(comp)) & (self._segments.Prob_max >=prob_max_threshold)].to_csv(
                    os.path.join(dirname, 'state_{}.bed'.format(comp)), sep='\t',
                    header=False, index=False,
                    columns=['chrom', 'start', 'end',
                             'name', 'score', 'strand',
                             'thickStart', 'thickEnd', 'itemRbg'])

        else:
            self._segments[(self._segments.Prob_max >=prob_max_threshold)].to_csv(os.path.join(dirname, 'segments.bed'), sep='\t',
                           header=False, index=False,
                           columns=['chrom', 'start', 'end',
                                    'name', 'score', 'strand',
                                    'thickStart', 'thickEnd', 'itemRbg'])


    def annotate(self, annotations):
        """Annotate the bins with BED, BAM or BIGWIG files."""

        tmpdir = tempfile.mkdtemp()
        filename = 'dummyexport'
        tmpfilename = os.path.join(tmpdir, filename)
        self._segments.to_csv(tmpfilename, sep='\t', header=False, index=False, columns=['chrom', 'start', 'end'])
        #self.export_bed(tmpfilename)

        if not isinstance(annotations, (list, dict)):
            annotations = [annotations]

        if isinstance(annotations, list):
            annotationdict = {os.path.basename(annot) for annot in annotations}
        else:
            annotationdict = annotations

        binsize = BedTool(tmpfilename)[0].length

        """Annotate the individual regions."""
        for key, file in annotationdict.items():
            if isinstance(file, list):
                self._segments[key] = file
                continue
            if file.endswith('.bed') or file.endswith('.bed.gz') or file.endswith('.narrowPeak') or \
               file.endswith('.bedgraph'):
                cov = Cover.create_from_bed(key, bedfiles=file, roi=tmpfilename,
                                            binsize=binsize, resolution=binsize, store_whole_genome=False,
                                            cache=True)
            elif file.endswith('.bam'):
                cov = Cover.create_from_bam(key, bamfiles=file, roi=tmpfilename,
                                            stranded=False, normalizer=[LogTransform()],
                                            store_whole_genome=False, binsize=binsize, resolution=binsize,
                                            cache=True)
            elif file.endswith('.bw') or file.endswith('.bigwig'):
                cov = Cover.create_from_bigwig(key, bigwigfiles=file, roi=tmpfilename,
                                               binsize=binsize, resolution=binsize, store_whole_genome=False,
                                               cache=True)
            self._segments[key] = cov.garray.handle['data'][:, 0, 0, 0]

        os.remove(tmpfilename)
        os.rmdir(tmpdir)

    @property
    def n_components(self):
        return self.model.n_components

    def geneset_observed_state_counts_old(self, genesets, genes, flanking=50000):
        """
        collect state counts around gene sets for enrichment analysis.

        """
        if isinstance(genes, str) and os.path.exists(genes):
            genes = BedTool(genes)

        genesetnames = [os.path.basename(x) for x in genesets]

        states = self._segments.name.unique()
        nstates = len(states)

        ngsets = len(genesets)

        print('Segmentation enrichment for {} states and {} sets.'.format(nstates, ngsets))
        tmpdir = tempfile.mkdtemp()
        filename = 'dummyexport'
        tmpfilename = os.path.join(tmpdir, filename)
        self._segments.to_csv(tmpfilename, sep='\t', header=False, index=False, columns=['chrom', 'start', 'end', 'name'])

        segment_bedtool = BedTool(tmpfilename)

        # obtain tss's for genes
        tss = BedTool([Interval(iv.chrom, iv.end if iv.strand=='-' else iv.start,
                                (iv.end if iv.strand=='-' else iv.start) + 1,
                                name=iv.name) for iv in genes]).sort()

        observed_segmentcounts = np.zeros((ngsets, self.n_components))
        geneset_length = np.zeros(ngsets)

        for igeneset, geneset in enumerate(genesets):

            if isinstance(geneset, str) and os.path.exists(geneset):
                if geneset.endswith('.bed'):
                    genenames =[iv.name for iv in BedTool(geneset)]
                else:
                    genenames = pd.read_csv(geneset).values.flatten()
            else:
                genenames = geneset

            #flank tss's by flanking window
            flanked_tss_ = BedTool([Interval(iv.chrom, max(0, iv.start-flanking), iv.end+flanking, name=iv.name) for iv in tss if iv.name in genenames])

            # collect segments in the surounding region
            roi_segments = segment_bedtool.intersect(flanked_tss_, wa=True, u=True)

            geneset_length[igeneset] = len(roi_segments)
            # obtain segment counts

            for iv in roi_segments:
                observed_segmentcounts[igeneset, self.to_stateid(iv.name)] += 1

        os.remove(tmpfilename)
        os.rmdir(tmpdir)
        return observed_segmentcounts, geneset_length, genesetnames

    def geneset_observed_state_counts(self, genesets, flanking=50000):
        """
        collect state counts around gene sets for enrichment analysis.

        """

        genesetnames = [os.path.basename(x) for x in genesets]

        states = self._segments.name.unique()
        nstates = len(states)

        ngsets = len(genesets)

        print('Segmentation enrichment for {} states and {} sets.'.format(nstates, ngsets))
        tmpdir = tempfile.mkdtemp()
        filename = 'dummyexport'
        tmpfilename = os.path.join(tmpdir, filename)
        self._segments.to_csv(tmpfilename, sep='\t', header=False, index=False, columns=['chrom', 'start', 'end', 'name'])

        segment_bedtool = BedTool(tmpfilename)

        observed_segmentcounts = np.zeros((ngsets, self.n_components))
        geneset_length = np.zeros(ngsets)

        for igeneset, geneset in enumerate(genesets):

            # obtain tss's for genes
            tss = BedTool([Interval(iv.chrom, iv.end if iv.strand=='-' else iv.start,
                                    (iv.end if iv.strand=='-' else iv.start) + 1,
                                    name=iv.name) for iv in BedTool(geneset)]).sort()

            #flank tss's by flanking window
            flanked_tss_ = BedTool([Interval(iv.chrom, max(0, iv.start-flanking), iv.end+flanking, name=iv.name) for iv in tss])

            # collect segments in the surounding region
            roi_segments = segment_bedtool.intersect(flanked_tss_, wa=True, u=True)

            geneset_length[igeneset] = len(roi_segments)
            # obtain segment counts

            for iv in roi_segments:
                observed_segmentcounts[igeneset, self.to_stateid(iv.name)] += 1

        os.remove(tmpfilename)
        os.rmdir(tmpdir)
        return observed_segmentcounts, geneset_length, genesetnames

    def observed_state_counts(self, regions, flanking=0):
        if isinstance(regions, str) and os.path.exists(regions):
            regions = BedTool(regions)

        _segments = self._segments
        regionnames = ['{}:{}-{}'.format(iv.chrom, iv.start, iv.end) for iv in regions]

        states = _segments.name.unique()
        nstates = len(states)

        nregions = len(regions)

        print('Segmentation enrichment for {} states and {} regions.'.format(nstates, nregions))

        observed_segmentcounts = np.zeros((nregions, self.n_components))
        region_length = np.zeros(nregions)

        for iregion, region in enumerate(regions):

            subregs = _segments[(_segments.chrom == region.chrom) & (_segments.start >= region.start) &  (_segments.end <= region.end)]
            region_length[iregion] = len(subregs)

            for istate in range(self.n_components):

                observed_segmentcounts[iregion, istate] = len(subregs[subregs.name == self.to_statename(istate)])
        return observed_segmentcounts, region_length, regionnames

    def geneset_enrichment(self, genesets, genes, prob_max_threshold=0.99):
        """Runs hypergeometric test for gene set enrichment."""

        if isinstance(genes, str) and os.path.exists(genes):
            genes = BedTool(genes)

        _segments = self._segments

        states = _segments.name.unique()
        nstates = len(states)

        ngsets = len(genesets)

        gen2loc = {}
        loc2idx = {}
        ngene = 0
        for iv in genes:
            tup = (iv.chrom, iv.end if iv.strand=='-' else iv.start, (iv.end if iv.strand=='-' else iv.start) + 1)
            if iv.name not in gen2loc:
                gen2loc[iv.name] = tup
            if tup not in loc2idx:
                loc2idx[tup] = ngene
                ngene += 1

        # obtain tss's for genes
        tss = BedTool([Interval(iv.chrom, iv.end if iv.strand=='-' else iv.start,
                                (iv.end if iv.strand=='-' else iv.start) + 1,
                                name=iv.name) for iv in genes]).sort() #.moveto('tmp_tss.bed')

        tss_ = tss.sort().merge()

        # build a gene to segment matrix
        seg2gene = np.zeros((nstates, ngene))
        for istate, state in enumerate(states):

            subset = _segments[(_segments.name == state) & (_segments.Prob_max >= .99)]

            # find closest tss per segment
            topic_to_closest_tss = BedTool([Interval(row.chrom, row.start, row.end) for _, row in subset.iterrows()]).sort().merge().closest(tss)
            for tiv in topic_to_closest_tss:
                seg2gene[istate, loc2idx[gen2loc[tiv.fields[6]]]] = 1

        gene2annot = np.zeros((ngene, ngsets))
        for igeneset, geneset in enumerate(genesets):

                if isinstance(geneset, str) and os.path.exists(geneset):
                    if geneset.endswith('.bed'):
                        genenames =[iv.name for iv in BedTool(geneset)]
                    else:
                        genenames = pd.read_csv(geneset).values.flatten()
                else:
                    genenames = geneset

                for genename in genenames:
                    if genename in gen2loc:
                        gene2annot[loc2idx[gen2loc[genename]], igeneset] = 1

        overlap = seg2gene.dot(gene2annot)
        enr = np.zeros_like(overlap)

        ns = seg2gene.sum(1)
        Ns = gene2annot.sum(0)

        for istate, _ in enumerate(states):
            for igeneset, _ in enumerate(genesets):

                enr[istate, igeneset] = -hypergeom.logsf(overlap[istate, igeneset]-1, ngene, ns[istate], Ns[igeneset])

        cleanup()
        enr = pd.DataFrame(enr, index=states,
                           columns=[os.path.basename(x) for x in genesets]).T

        return enr

#    def sample_enrichment(self, ref_anchors, samplefiles, samplenames, other_anchors=None, prob_max_threshold=0.99):
#        if not isinstance(ref_anchors, BedTool):
#            ref_anchors = BedTool(ref_anchors)
#        ref_anchors = BedTool([Interval(iv.chrom, iv.start, iv.end, name=str(i)) for i, iv in enumerate(ref_anchors)])
#
#        state2refanchor = np.zeros((self.n_components, len(ref_anchors)))
#        states = self._segments.name.unique()
#
#        for istate in range(self.n_components):
#            state = self.to_statename(istate)
#            segments = self._segments[(self._segments.name == state) & (self._segments.Prob_max >= prob_max_threshold)]
#            segbed = BedTool([Interval(row.chrom, row.start, row.end) for _, row in segments.iterrows()]).sort().merge()
#
#            overlapbed = ref_anchors.intersect(segbed, wa=True)
#            state2refanchor[istate, np.asarray([int(iv.name) for iv in overlapbed])] = 1
#
#        if hasattr(samplefiles, 'shape') and samplefiles.shape[0] == len(ref_anchors):
#            featuremat = samplefiles
#        else:
#            featuremat = np.zeros((len(ref_anchors), len(samplefiles)))
#            if other_anchors is not None and not isinstance(other_anchors, BedTool):
#                other_anchors = BedTool(other_anchors)
#
#            other_anchors = BedTool([Interval(iv.chrom, iv.start, iv.end, name=str(i)) for i, iv in enumerate(other_anchors)])
#            for isample, sample in enumerate(samplefiles):
#                if not isinstance(sample, BedTool):
#                    sample = BedTool(sample)
#                overlap = other_anchors.intersect(sample, wa=True)
#                for iv in overlap:
#                    featuremat[int(iv.name), isample] = 1
#
#        # start with the enrichment test
#
#        overlap = state2refanchor.dot(featuremat)
#
#        enr = np.zeros_like(overlap)
#
#        ns = state2refanchor.sum(1)
#        Ns = featuremat.sum(0)
#        ntotal = (state2refanchor.sum(0)>0).sum()
#
#        for istate in range(self.n_components):
#            for ifeature in range(featuremat.shape[1]):
#                enr[istate, ifeature] = -hypergeom.logsf(overlap[istate, ifeature]-1, ntotal, ns[istate], Ns[ifeature])
#
#        enr = pd.DataFrame(enr,
#                           index=self.to_statenames(np.arange(self.n_components)),
#                           columns=samplenames)
#        enr = enr.T
#
#        return enr


    def broadregion_enrichment(self, state_counts, regionlengths, regionnames=None, mode='logfold'):

        stateprob = self.model.get_stationary_distribution()

        enr = np.zeros_like(state_counts)

        e = np.outer(regionlengths, stateprob)

        if mode == 'logfold':
            enr = np.log10(np.where(state_counts==0, 1, state_counts)) - np.log10(np.where(state_counts==0, 1, e))
        if mode == 'fold':
            enr = np.where(state_counts==0, 1, state_counts)/np.where(state_counts==0, 1, e)
        elif mode == 'chisqstat':
            stat = np.where((state_counts - e) >= 0.0, (state_counts - e), 0.0)

            enr = stat**2 / e**2
        elif mode == 'pvalue':
            for ireg, scnt in enumerate(state_counts.sum(1)):

                self._make_broadregion_null_distribution(np.array([int(scnt)]))
                null_dist, _ = self._get_broadregion_null_distribution(int(scnt))

                for istate in range(self.n_components):
                    n_obs = int(state_counts[ireg, istate])
                    enr[ireg, istate] = -min(np.log10(max(0.0, null_dist[n_obs:, istate].sum())), 15)

        enrdf = pd.DataFrame(enr, columns=self.to_statenames(np.arange(self.n_components)),
                             index=regionnames)

        return enrdf


    def _init_broadregion_null_distribution(self, max_len):
        max_len = int(max_len)
        cnt_dist = np.zeros((max_len+1, self.n_components, self.n_components))

        stationary = self.model.get_stationary_distribution()

        # initialize array
        for dostate_i in range(self.n_components):
            for curr_i in range(self.n_components):
                shift = 1 if dostate_i == curr_i else 0

                cnt_dist[shift, dostate_i, curr_i] += stationary[curr_i]

        fft_cnt_dist = np.empty((max_len+1, self.n_components, self.n_components), dtype='complex128')
        for dostate_i in range(self.n_components):
            for curr_i in range(self.n_components):
               fft_cnt_dist[:, dostate_i, curr_i] = np.fft.fft(cnt_dist[:, dostate_i, curr_i])

        self._fft_init_cnt_dist = fft_cnt_dist
        self._cnt_storage = {}
        self._cnt_conditional = {}


    def _make_broadregion_conditional_null_distribution(self, length, max_length):
        max_length = int(max_length)
        length = int(length)

        if length < 1:
            return

        if length in self._cnt_conditional:
            return

        if length == 1:
            # initialization condition
            # adding one more bin
            # respresents P(Csi, sj | s0)
            cnt_dist = np.zeros((max_length + 1, self.n_components, self.n_components, self.n_components))
            for dostate_i in range(self.n_components):
                for prev_i in range(self.n_components):
                    for curr_i in range(self.n_components):
                        shift = 1 if dostate_i == curr_i else 0

                        cnt_dist[shift, dostate_i, prev_i, curr_i] += self.model.transmat_[prev_i, curr_i]

            fft_cnt_dist = np.zeros((max_length + 1, self.n_components, self.n_components, self.n_components), dtype='complex128')
            for dostate_i in range(self.n_components):
                for prev_i in range(self.n_components):
                    for curr_i in range(self.n_components):
                        fft_cnt_dist[:, dostate_i, prev_i, curr_i] = np.fft.fft(cnt_dist[:, dostate_i, prev_i, curr_i])

        else:
            self._make_broadregion_conditional_null_distribution(int(np.ceil(length/2)), max_length)
            self._make_broadregion_conditional_null_distribution(int(np.floor(length/2)), max_length)

            if True:
                fft_cnt_dist = faster_fft(self._cnt_conditional[int(np.ceil(length/2))], self._cnt_conditional[int(np.floor(length/2))], self.n_components, max_length)
            else:
                fft_cnt_dist = np.zeros((max_length+1, self.n_components, self.n_components, self.n_components), dtype='complex128')

                for prev_i in range(self.n_components):
                    for middle_i in range(self.n_components):
                        for curr_i in range(self.n_components):

                            cnt_1 = self._cnt_conditional[int(np.ceil(length/2))][:, :, prev_i, middle_i]
                            cnt_2 = self._cnt_conditional[int(np.floor(length/2))][:, :, middle_i, curr_i]

                            fft_cnt_dist[:, :, prev_i, curr_i] += cnt_1 * cnt_2

        self._cnt_conditional[length] = fft_cnt_dist

    def _finalize_broadregion_null_distribution(self, keep_lengths):

        max_length = int(keep_lengths.max())

        for length in keep_lengths:
            self._make_broadregion_conditional_null_distribution(length - 1, max_length)

        for length in keep_lengths:
            length = int(length)
            if length == 1:
                fft_tmp_cnt = self._fft_init_cnt_dist
            else:
                fft_tmp_cnt = np.zeros_like(self._fft_init_cnt_dist)

                for dostate_i in range(self.n_components):
                    for prev_i in range(self.n_components):
                        for curr_i in range(self.n_components):
                            fft_tmp_cnt[:,dostate_i, curr_i] += self._fft_init_cnt_dist[:, dostate_i, prev_i] * \
                                                                     self._cnt_conditional[length - 1][:, dostate_i, prev_i, curr_i]

            fft_tmp_cnt = fft_tmp_cnt.sum(-1)
            tmp_cnt = np.zeros((max_length+1, self.n_components))
            self._cnt_storage[length] = tmp_cnt
            for dostate_i in range(self.n_components):
                self._cnt_storage[length][:, dostate_i]  = np.fft.ifft(fft_tmp_cnt[:, dostate_i]).real


    def _make_broadregion_null_distribution(self, keep_lengths):

        self._init_broadregion_null_distribution(keep_lengths.max())

        self._finalize_broadregion_null_distribution(keep_lengths)

    def _make_broadregion_null_distribution_slow(self, keep_lengths):

        length = int(keep_lengths.max())
        cnt_dist = np.zeros((2, self.n_components, self.n_components))

        stationary = self.model.get_stationary_distribution()

        # initialize array
        for dostate_i in range(self.n_components):
            for curr_i in range(self.n_components):
                shift = 1 if dostate_i == curr_i else 0

                cnt_dist[shift, dostate_i, curr_i] += stationary[curr_i]
        if 1 in keep_lengths and 1 not in self._cnt_storage:
            self._cnt_storage[1] = cnt_dist.sum(-1)

        for icnt in range(length-1):
            if (icnt % 1000) == 0:
                print("iter {}/{}".format(icnt, length))
            joint_cnt_dist = np.zeros((icnt + 3, self.n_components, self.n_components))
            for dostate_i in range(self.n_components):
                for prev_i in range(self.n_components):
                    for curr_i in range(self.n_components):
                        shift = 1 if dostate_i == curr_i else 0
                        if dostate_i == curr_i:
                            joint_cnt_dist[1:(icnt+3), dostate_i, curr_i] += cnt_dist[:(icnt+2), dostate_i, prev_i] * self.model.transmat_[prev_i, curr_i]
                        else:
                            joint_cnt_dist[:(icnt+2), dostate_i, curr_i] += cnt_dist[:(icnt+2), dostate_i, prev_i] * self.model.transmat_[prev_i, curr_i]
            cnt_dist = joint_cnt_dist
            if (icnt+2) in keep_lengths and (icnt+2) not in self._cnt_storage:
                self._cnt_storage[icnt+2] = cnt_dist.sum(-1)

        for k in self._cnt_storage:
            np.testing.assert_allclose(self._cnt_storage[k].sum(), self.n_components)

    def _get_broadregion_null_distribution(self, length):
        return self._cnt_storage[length], \
               self._cnt_storage[length].T.dot(np.arange(self._cnt_storage[length].shape[0]))

    def make_seeds(self, data, nstates, 
                   decoding_prob=0.90, n_jobs=10,
                   maxclusters=10, regions=3000):

        post = self.model.predict_proba(data)
        X = hstack(data)
        seeds = np.empty((0, X.shape[1]))

        # first, within each state apply LDA
        print('precluster')
        for stateid in range(self.n_components):
            idx = post[:,stateid]>decoding_prob
            
            x = X[idx]
            nsubclusters = min(max(x.shape[0] // regions, 1), maxclusters)
            tmpseeds = np.empty((nsubclusters, X.shape[1]))
    
            lda = LatentDirichletAllocation(nsubclusters, max_iter=100,
                                            n_jobs=n_jobs)
            lda.fit(x.T)
    
            print('process {}: {} regions, {} clusters'.format(stateid, x.shape[0], nsubclusters))
            idxs = np.argsort(lda.components_, 1)[:,-1000:][:,::-1]
            for icomp in range(nsubclusters):
                tmpseeds[icomp] = np.asarray(x[idxs[icomp]].sum(0)).flatten()
            seeds = np.concatenate([seeds, tmpseeds], axis=0)

        print('consolidate')
        labels = AgglomerativeClustering(min(nstates, seeds.shape[0]),
                                         affinity='cosine',
                                         linkage='average').fit_predict(pd.DataFrame(seeds.T).corr())

        new_states = np.empty((nstates, x.shape[1]))
        for ilabel in range(nstates):
            new_states[ilabel] = seeds[labels==ilabel].sum(0)

        return np.split(new_states, np.cumsum([d.shape[1] for d in data])[:-1],  axis=1)
        #return new_states
        
#    def state_filtering(self, data, stateid, nsubclusters=4, 
#                                           n_cells=1000, decoding_prob=0.99, n_jobs=10):
#        print('explore', stateid)
#        post = self.model.predict_proba(data)
#        idx = post[:,stateid]>decoding_prob
#        
#        X = hstack(data)
#        X = X[idx]
#        #v=np.var(X.toarray(), axis=0)
#        #bestcells = np.argsort(v)[::-1][:n_cells]
#        #
#        #
#        #x = X[:,bestcells].toarray()
#        #x -= x.mean(axis=0, keepdims=True)
#        #print(x.shape)
#        #
#        ac = LatentDirichletAllocation(nsubclusters, max_iter=100, n_jobs=n_jobs)
#        ac.fit(x.T)
#
#        idxs = np.argsort(lda.components_, 1)[:,-1000:][:,::-1]
#        seeds = np.empty((nsubclusters, X.shape[1]))
#        for icomp in ac.components_.shape[0]:
#            
#            ac.components_[icomp)
#        
#        labels = subcluster(x, nsubclusters)
#    
#        return bestcells, labels, x
#
#    def explore_state_correlationstructure(self, data, stateid, nsubclusters=4, 
#                                           n_cells=1000, decoding_prob=0.99):
#        bestcells, labels, x = self.state_filtering(data, stateid, nsubclusters, 
#                                       n_cells, decoding_prob)
#        
#        print('rowclusters', labels.value_counts())
#    
#        lut = dict(zip(np.sort(labels.unique()), 
#                   sns.color_palette("muted", n_colors=len(labels.unique()))))
#        row_colors = labels.map(lut)
#        g=sns.clustermap(pd.DataFrame(x).corr(),
#                        vmax=.1, cmap="YlGnBu", 
#                        row_colors=row_colors,
#                        col_colors=row_colors,
#                         figsize=(20,20))
#        return g
#
#    def extract_subcluster_seeds(self, data, stateid, nsubclusters=4, 
#                                           n_cells=1000, decoding_prob=0.99):
#        bestcells, labels, x = self.state_filtering(data, stateid, nsubclusters, 
#                                       n_cells, decoding_prob)
#        
#        M = np.hstack(self.model.emission_suffstats_)
#            
#        
#        # reuse the sufficient statistics
#        cluster_seed_regions = np.zeros((nsubclusters, M.shape[1]))
#        cluster_seed_regions += M[stateid:(stateid+1)] / nsubclusters
#        cluster_seed_regions[:, bestcells] = 0
#            
#        # make suff stats uncorrelated according to the clustering
#        for icluster in range(nsubclusters):
#            cluster_seed_regions[icluster, bestcells[icluster==labels]] += \
#               np.asarray(M[stateid:(stateid+1), 
#                            bestcells[icluster==labels]].sum(0)).flatten()
#            
#        # merge with original suffstats
#        M = np.delete(M, stateid, 0)
#        cluster_seed_regions = np.concatenate([M, cluster_seed_regions], axis=0)
#
#        # split up into submatrices
#        lens = [d.shape[1] for d in data]
#        np.cumsum(lens)
#        new_seeds = []
#        for i, end in enumerate(np.cumsum(lens)):
#            start = 0 if i == 0 else np.cumsum(lens)[i-1]
#            new_seeds.append(cluster_seed_regions[:, start:end])
#                
#        return new_seeds
#
    def newModel(self, new_seeds, n_iter=100, n_jobs=1, verbose=True):
        model = MultiModalDirMulHMM(new_seeds[0].shape[0], 
                                    n_iter=n_iter,
                                    n_jobs=n_jobs, verbose=verbose)
        model.init_params = 'st'

        model.emission_suffstats_ = new_seeds
        model.emission_prior_ = self.model.emission_prior_
        model.n_features = self.model.n_features

        return model

    def get_subdata(self, data, query_states, collapse_neighbors=True, state_prob_threshold=0.99):

        if not isinstance(data, list):
            data = [data]
        if not isinstance(query_states, list):
            query_states = [query_states]

        subset = self._segments[(self._segments.name.isin(query_states))
                                & (self._segments.Prob_max >= state_prob_threshold)].copy()

        submats = []

        if not collapse_neighbors:
            for datum in data:

                sm = np.asarray(datum[subset.index].todense())

                submats.append(sm)

            return submats, subset


        # determine neighboring bins that can be merged together
        prevind = -2
        prevstate = ''
        nelem = 0
        mapelem = []
        for i, r in subset.iterrows():
            curstate = r['name']
            nelem += 1
            if i == (prevind + 1) and prevstate == curstate:
               nelem -= 1
            mapelem.append(nelem - 1)
            prevind = i
            prevstate = curstate


        subset['common'] = mapelem
        print(subset.shape)

        subset_merged = subset.groupby(['common', 'name']).aggregate({'chrom': 'first',
                                        'start': 'min',
                                        'end': 'max', 'name':'first'})

        print(subset_merged.shape)
        for datum in data:

            dat = datum.tolil()
            submat = lil_matrix((nelem, datum.shape[1]))
            #sm = np.asarray(datum[subset.index].todense())

            for i, i2m in enumerate(mapelem):
                #submat[i2m, :] += sm[i]
                submat[i2m, :] += datum[subset.index[i]]

            submats.append(submat.tocsc())

        return submats, subset_merged


class MixtureOfMultinomials:
    def __init__(self, n_clusters, n_iters=10, random_state=None, verbose=False):
        self.n_clusters = n_clusters
        self.pi_prior = 1.
        self.component_prior = 1.
        self.n_iters = n_iters
        self.random_state = random_state
        self.verbose = verbose


    def fit(self, data):
        if not isinstance(data, list):
            data = [data]

        # init
        self._init(data)
        self._check()
        if self.verbose: print('init', self.score(data))
        for iit in range(self.n_iters):
            p_z = self._e_step(data)
            np.testing.assert_allclose(p_z.sum(), data[0].shape[0])
            self._m_step(data, p_z)
            self._check()
            if self.verbose: print(iit, self.score(data))

    def _check(self):
        assert np.all(self.pi > 0.0)
        np.testing.assert_allclose(self.pi.sum(), 1.)
        for i in range(len(self.components)):
            assert np.all(self.components[i] > 0.0)
            np.testing.assert_allclose(self.components[i].sum(), self.n_clusters)

    def _init(self, data):

        self.random_state_ = check_random_state(self.random_state)

        self.pi = np.ones(self.n_clusters)

        normalize(self.pi)

        seed_regions = self.random_state_.permutation(data[0].shape[0])[:self.n_clusters]

        self.components = []
        for i in range(len(data)):

            ex = data[i][seed_regions] + self.component_prior
            normalize(ex, axis=1)
            self.components.append(ex)
            #normalize(self.components, axis=1)


    def predict(self, data):
        p_z = self._e_step(data)
        labels = np.argmax(p_z, axis=1)
        return labels

    def transform(self, data):
        return self._e_step(data)

    def _log_likeli(self, data):
        log_p_z = np.zeros((data[0].shape[0], self.n_clusters))
        for i, datum in enumerate(data):
            log_p_z += datum.dot(np.log(self.components[i].T))
        log_p_z += np.log(self.pi)[None,:]
        return log_p_z

    def _e_step(self, data):
        log_p_z = self._log_likeli(data)
        log_p_z -= logsumexp(log_p_z, axis=1, keepdims=True)
        return np.exp(log_p_z)

    def _m_step(self, data, p_z):
        self.pi = p_z.sum(0) + self.pi_prior
        normalize(self.pi)

        for i, datum in enumerate(data):
            stats = p_z.T.dot(datum) + self.component_prior
            self.components[i] = stats
            normalize(self.components[i], axis=1)


    def score(self, data):
        return logsumexp(self._log_likeli(data), axis=1).sum()

    def get_important_components(self, min_explanation=0.5):
        def get_major_indices(pi, mexp):
            ipi = np.argsort(pi)

            return ipi[np.cumsum(pi[ipi]) > mexp]
        #i = np.where(self.pi >= min_explanation)[0]
        i = get_major_indices(self.pi, min_explanation)
        print('{}: n components: {}, proportion explained: {}'.format(self.pi, len(i), self.pi[i].sum()))
        #print('explained proportion of data: {}'.format(self.pi[i].sum()))
        return [comp[i].copy() for comp in self.components]


def subset_bam(mainbam, barcodes_to_keep, filename):

    inbam = AlignmentFile(mainbam, 'rb')

    outbam = AlignmentFile(filename, 'wb', template=inbam)
    for aln in inbam.fetch(until_eof=True):
        if aln.get_tag('RG') in barcodes_to_keep:
            outbam.write(aln)
    outbam.close()
    inbam.close()

def bam2bigwig(inbamfile, outbigwigfile):
    inbam = AlignmentFile(inbamfile, 'rb')

    outbw = pyBigWig.open(outbigwigfile, 'w')
    header = [(chrom['SN'], clen['LN']) for chrom in inbam.header.to_dict()['SQ']]

    outbw.addHeader(header)

    for chrom, clen in header:
        cov=np.zeros(clen)
        for aln in inbam.fetch(chrom):
            cov[aln.pos]+=1
        outbw.addEntries(str(chrom), 0, values=list(cov), span=1, step=1)

    outbw.close()
    inbam.close()


def axt_to_bedtool(axtfile, ref_filters=None, other_filters=None, min_cons_len=1, trim_length=None):
    ref_iv_list = []
    other_iv_list = []
    i = 0
    fopen = gzip.open if axtfile.endswith('.gz') else open
    with fopen(axtfile, 'r') as f:
        while 1:
            line = f.readline()
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            if not line:
                break

            items = line.split(' ')
            if int(items[3]) - int(items[2]) >= min_cons_len:

                if trim_length is None:
                    trl = (int(items[3]) - int(items[2]))//2
                else:
                    trl = trim_length

                mid = (float(items[2]) + float(items[3])) / 2.
                ref_start = int(max(1, mid-trl))
                ref_end = int(mid+trl)

                riv = Interval(items[1], ref_start, ref_end, name=str(i))

                mid = (float(items[5]) + float(items[6])) / 2.
                oth_start = int(max(1, mid-trl))
                oth_end = int(mid+trl)

                oiv = Interval(items[4], oth_start, oth_end, name=str(i))

                ref_iv_list.append(riv)
                other_iv_list.append(oiv)
                i += 1

            f.readline() # reference sequence
            f.readline() # sequence of other species
            f.readline() # blank

    refbed = BedTool(ref_iv_list)
    rkeepids = np.arange(len(refbed))
    if ref_filters is not None:
        if not isinstance(ref_filters, list):
            ref_filters = [ref_filters]

        remaining = refbed
        for ref_filt in ref_filters:
            remaining = remaining.intersect(BedTool(ref_filt), v=True, f=0.5)
        rkeepids = np.asarray([int(iv.name) for iv in remaining])

    othbed = BedTool(other_iv_list)
    okeepids = np.arange(len(othbed))
    if other_filters is not None:
        if not isinstance(other_filters, list):
            other_filters = [other_filters]

        remaining = othbed
        for filt in other_filters:
            remaining = remaining.intersect(BedTool(filt), v=True, f=0.5)
        othbed = remaining
        okeepids = np.asarray([int(iv.name) for iv in remaining])

    keepids = np.intersect1d(rkeepids, okeepids)

    # filter and reindex the regions
    refbed = BedTool([Interval(iv.chrom, iv.start, iv.end, name=str(i)) for i, iv in enumerate(refbed) if int(iv.name) in keepids])
    othbed = BedTool([Interval(iv.chrom, iv.start, iv.end, name=str(i)) for i, iv in enumerate(othbed) if int(iv.name) in keepids])

    return refbed, othbed
