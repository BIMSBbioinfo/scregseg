import pandas as pd
import numpy as np
import seaborn as sns
import tempfile
from pybedtools import BedTool
from pybedtools import Interval
from pybedtools.helpers import cleanup
from scseg.hmm import DirMulHMM
from scseg.countmatrix import CountMatrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.stats import zscore
from scipy.sparse import issparse as is_sparse
import os
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state

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

def export_segmentation(segments, filename, prob_max_threshold=0.99):
    """
    Exports the segmentation results table.
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    segments[(segments.Prob_max >= prob_max_threshold)].to_csv(filename, sep='\t', index=False)


def export_bed(subset, filename, individual_beds=False):
    """
    Exports the segmentation results in bed format.
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if individual_beds:
        f = filename.split('.bed')[0]
        
        for state in subset.name.unique():
            subset[(subset.name == state)].to_csv(
                '{}_{}.bed'.format(f, state), sep='\t',
                header=False, index=False,
                columns=['chrom', 'start', 'end',
                         'name', 'score', 'strand',
                         'thickStart', 'thickEnd', 'itemRbg'])

    else:
        subset.to_csv(filename, sep='\t',
                       header=False, index=False,
                       columns=['chrom', 'start', 'end',
                                'name', 'score', 'strand',
                                'thickStart', 'thickEnd', 'itemRbg'])


def get_labeled_data(X):
    if not isinstance(X, list):
        X = [X]
    X_ = []
    labels_ = []
    for cm in X:
       if isinstance(cm, CountMatrix):
           X_.append(cm.cmat.tocsr())
           labels_.append(cm.cannot)
       else:
           X_.append(cm)
           labels_.append(None)
    return X_, labels_

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

    def score(self, X):
        X_, labels_ = get_labeled_data(X)
        if not hasattr(self, "labels_"):
            setattr(self, "labels_", labels_)
        return self.model.score(X_)
        
    def fit(self, X):
        X_, labels_ = get_labeled_data(X)
        if not hasattr(self, "labels_"):
            setattr(self, "labels_", labels_)
        self.model.fit(X_)

    def save(self, path):
        """
        saves current model parameters
        """
        self.model.save(os.path.join(path, 'modelparams'))

        if hasattr(self, "_segments") and self._segments is not None:
            export_segmentation(self._segments, os.path.join(path, 'summary',
                                     'segmentation.tsv'), 0.0)
            merged_segment = self.get_statecalls(self.all_statenames(),  state_prob_threshold=0.0)
            export_segmentation(merged_segment, os.path.join(path, 'summary',
                                     'merged_segmentation.tsv'), 0.0)

    @classmethod
    def load(cls, path):
        """
        loads model parameters from path
        """
        if os.path.exists(os.path.join(path, 'modelparams', 'dirmulhmm.npz')):
            model = DirMulHMM.load(path)
        else:
            raise ValueError("Model not available")
        scmodel = cls(model)

        scmodel.load_segments(os.path.join(path, 'summary', 'segmentation.tsv'))

        return scmodel

    def load_segments(self, path):
        if os.path.exists(path):
            self._segments = pd.read_csv(path, sep='\t')

    def all_statenames(self):
        """
        converts list of state ids (integer) to list of state names (str)
        """
        return [self.to_statename(state) for state in range(self.n_components)]

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

    def cell2state_counts(self, X, prob_max_threshold=0.0, post=False):
        """ Determines the state calls per cells.

        """

        X_, labels_ = get_labeled_data(X)
        if not hasattr(self, "labels_"):
            setattr(self, "labels_", labels_)

        if post:
            # use posterior decoding
            statescores = self.model.predict_proba(X_)
            Z = statescores.T
        else:
            states = self.model.predict(X_)

            values = np.ones(len(states))
            values[self._segments.Prob_max < prob_max_threshold] = 0

            Z = csc_matrix((values,
                           (states.astype('int'), np.arange(len(states),dtype='int'))))

        enrs = []

        for d in X_:
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

    def cell2state(self, X, mode='logfold', prob_max_threshold=0.0, post=False):
        """ Determines whether a states is overrepresented among
        the accessible sites in a given cellself.

        The P-value is determined using the binomial test
        and the log-fold-change is determine by Obs[state proportion]/Expected[state proportion].
        """

        X_, labels_ = get_labeled_data(X)
        if not hasattr(self, "labels_"):
            setattr(self, "labels_", labels_)

        obs_seqfreqs = self.cell2state_counts(X_, prob_max_threshold, post)

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

    def plot_normalized_emissions(self, idat):
       em = self.model.emission_suffstats_[idat] + self.model.emission_prior_[idat]

       nem = em / em.sum(1, keepdims=True)
       tem = em.sum(0, keepdims=True)/em.sum()
       lodds = np.log(nem) - np.log(tem)

       if hasattr(self, "labels_"):
           if 'cell' in self.labels_[idat].columns:
               l = self.labels_[idat].cell
           elif 'barcodes' in self.labels_[idat].columns:
               l = self.labels_[idat].barcodes
       else:
           l = [str(i) for i in range(lodds.shape[1])]

       g = sns.clustermap(pd.DataFrame(lodds, columns=l), center=0., robust=True, cmap='RdBu_r', figsize=(15,15))
       g.ax_heatmap.set_ylabel('States')
       g.ax_heatmap.set_xlabel('Features')

       return g

    @property
    def color(self):
        return self._color


    def segment(self, X, regions, algorithm=None):
        """
        performs segmentation.

        Parameters
        ----------
        X : list(np.array) or list(scipy.sparse.csc_matrix)
            List of count matrices
        regions : pd.DataFrame
            Dataframe containing the genomic intervals (e.g. from a bed file).
        """
        X_, labels_ = get_labeled_data(X)
        if not hasattr(self, "labels_"):
            setattr(self, "labels_", labels_)

        bed = BedTool(regions)

        regions_ = pd.DataFrame([[iv.chrom, iv.start, iv.end] for iv in bed],
                                columns=['chrom', 'start', 'end'])

        statenames = self.to_statenames(self.model.predict(X_, algorithm))
        statescores = self.model.predict_proba(X_, algorithm)

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

        for i in range(len(X_)):
            regions_['readdepth_' + str(i)] = X_[i].sum(1)

        self._segments = regions_
        cleanup()




    def annotate(self, annotations):
        """Annotate the bins with BED, BAM or BIGWIG files."""
        from janggu.data import Cover
        from janggu.data import LogTransform

        tmpdir = tempfile.mkdtemp()
        filename = 'dummyexport'
        tmpfilename = os.path.join(tmpdir, filename)
        self._segments.to_csv(tmpfilename, sep='\t', header=False, index=False, columns=['chrom', 'start', 'end'])

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

    def _get_broadregion_null_distribution(self, length):
        return self._cnt_storage[length], \
               self._cnt_storage[length].T.dot(np.arange(self._cnt_storage[length].shape[0]))

    def get_statecalls(self, query_states,
                       collapse_neighbors=True,
                       state_prob_threshold=0.99):

        if not isinstance(query_states, list):
            query_states = [query_states]

        subset = self._segments[(self._segments.name.isin(query_states))
                                & (self._segments.Prob_max >= state_prob_threshold)].copy()

        if not collapse_neighbors:
            return subset

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

        processing = {'chrom': 'first',
                      'start': 'min',
                      'end': 'max', 'name':'first',
                      'score': 'max', 'strand': 'first',
                      'thickStart': 'min', 'thickEnd': 'max',
                      'itemRbg': 'first',
                      'Prob_max': 'max',
                     }

        for state in query_states:
            processing['Prob_' + state] = 'max'
        
        for field in list(set(subset.columns) - set(processing.keys())):
            processing[field] = 'mean'
        
        subset_merged = subset.groupby(['common', 'name']).aggregate(processing)

        return subset_merged


    def get_subdata(self, X, query_states, collapse_neighbors=True, state_prob_threshold=0.99):
        """ function deprecated: use get_statecalls() """
        X_, labels_ = get_labeled_data(X)
        if not hasattr(self, "labels_"):
            setattr(self, "labels_", labels_)

        if not isinstance(query_states, list):
            query_states = [query_states]

        subset = self._segments[(self._segments.name.isin(query_states))
                                & (self._segments.Prob_max >= state_prob_threshold)].copy()

        submats = []

        if not collapse_neighbors:
            for datum in X_:

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

        subset_merged = subset.groupby(['common', 'name']).aggregate({'chrom': 'first',
                                        'start': 'min',
                                        'end': 'max', 'name':'first'})

        for datum in X_:

            dat = datum.tocsr()
            submat = lil_matrix((nelem, datum.shape[1]))

            for i, i2m in enumerate(mapelem):
                submat[i2m, :] += dat[subset.index[i]]

            submats.append(submat.tocsc())

        return submats, subset_merged


