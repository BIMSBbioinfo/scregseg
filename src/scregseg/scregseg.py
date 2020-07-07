""" Scregseg main module
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import seaborn as sns
import tempfile
from pybedtools import BedTool
from pybedtools import Interval
from pybedtools.helpers import cleanup
from scregseg.hmm import DirMulHMM
from scregseg.countmatrix import CountMatrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.stats import zscore
from scipy.sparse import issparse as is_sparse
import os
import matplotlib.pyplot as plt
from matplotlib import cm
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
    """ Returns countmatrices and cell labels

    If the input is a sparse matrix, labels will be None.
    Otherwise, the labels are extracted from the CountMatrix
    object.

    Parameters
    ----------
    X : list(CountMatrix) or list(sparse matrix)
        Input count matrix.

    Returns
    -------
    tuple(sparse count matrix, cell labels)
    """
    if not isinstance(X, list):
        X = [X]
    X_ = []
    labels_ = []
    for i, cm in enumerate(X):
       if isinstance(cm, CountMatrix):
           X_.append(cm.cmat.tocsr())
           labels_.append(pd.DataFrame({'label': cm.cannot.cell,
                                        'matrixid': i}))
       else:
           X_.append(cm)
           labels_.append(None)
    if isinstance(labels_[0], pd.DataFrame):
        labels_ = pd.concat(labels_, ignore_index=True)
    return X_, labels_

def run_segmentation(data, nstates, niter, random_states, n_jobs):
    best_score = -np.inf
    scores = []
    print('Fitting {} models'.format(len(random_states)))
    for random_state in random_states:
        print("Starting {}".format(random_state))
        model = Scregseg(DirMulHMM(n_components=nstates, n_iter=niter, random_state=random_state, verbose=True,
                                n_jobs=n_jobs))
        model.fit(data)
        score = model.score(data)
        scores.append(score)
        if best_score < score:
            best_score = score
            best_model = model
            best_seed = random_state

    print('all models: seed={}, score={}'.format(random_states, scores))
    print('best model: seed={}, score={}'.format(best_seed, best_score))
    scmodel = best_model
    return scmodel

def get_statecalls(segments, query_states,
                    ntop=5000,
                    state_prob_threshold=0.9,
                    collapse_neighbors=True,
                    minreads=0.):
    """ obtain state calls from segmentation.

    This function allows to filter state calls in various ways
    and optionally to collapse bookended bins of the same state.

    Parameters
    ----------
    segments : pd.DataFrame
        Segmentation results.
    query_states : list(str)
        List of query state names to report
    ntop : int
        Minimum posterior decoding probability to select high confidence state calls.
        Default: 0.99

    Returns
    -------
    pandas.DataFrame :
        Filtered segmentation results
    """

    if not isinstance(query_states, list):
        query_states = [query_states]

    subset = segments[(segments.name.isin(query_states))
                      & (segments.Prob_max >= state_prob_threshold)
                      & (segments.readdepth >= minreads)].copy()
    #subset = self._segments[(self._segments.name.isin(query_states))
    #                         & (self._segments.Prob_max >= state_prob_threshold)].copy()

    if collapse_neighbors:
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

        processing['readdepth'] = 'sum'

        subset = subset.groupby(['common', 'name']).aggregate(processing)

    dfs = []

    for process_state in query_states:
        subset['pscore'] = subset['Prob_{}'.format(process_state)] * subset['readdepth']
        #subset = subset[subset.pscore >= minreads]
        dfs.append(subset.nlargest(ntop, 'pscore').copy())

    subset_merged = pd.concat(dfs, axis=0)
    return subset_merged

def get_statecalls_posteriorprob(segments, query_states,
                   collapse_neighbors=True,
                   state_prob_threshold=0.99):
    """ obtain state calls from segmentation.

    This function allows to filter state calls in various ways
    and optionally to collapse bookended bins of the same state.

    Parameters
    ----------
    query_states : list(str)
        List of query state names to report
    collapse_neighbors : bool
        Whether to merged bookended bins if they represent the same state.
        Default: True
    state_prob_threshold : float
        Minimum posterior decoding probability to select high confidence state calls.
        Default: 0.99

    Returns
    -------
    pandas.DataFrame :
        Filtered segmentation results
    """

    if not isinstance(query_states, list):
        query_states = [query_states]

    subset = segments[(segments.name.isin(query_states))
                            & (segments.Prob_max >= state_prob_threshold)].copy()

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


class Scregseg(object):
    """Scregseg class: Single-cell segmentation

    Parameters
    ----------
    model : DirMulHMM
        HMM model to maintain and perform the analysis with.
    """

    _enr = None
    _cnt_storage = {}
    _cnt_conditional = {}

    def __init__(self, model):
        self._segments = None
        self._nameprefix = 'state_'
        self.model = model
        self._color = {name: el for name, el in \
                       zip(self.to_statenames(np.arange(self.n_components)), cm.get_cmap('gist_rainbow')(np.linspace(0.0, 1.0, self.n_components))[:, :3].tolist())}

    def score(self, X):
        """ Log-likelihood score for X given the HMM.

        Parameters
        ----------
        X : list(CountMatrix) or list(sparse matrix)
           Count matrix.

        Returns
        -------
        float :
            Log-likelihood score for X given the model.
        """
        X_, _ = get_labeled_data(X)
        return self.model.score(X_)

    def fit(self, X):
        """ Model fitting given X.

        Parameters
        ----------
        X : list(CountMatrix) or list(sparse matrix)
            Count matrix.

        """
        X_, labels_ = get_labeled_data(X)
        setattr(self, "labels_", labels_)
        self.model.fit(X_)

    def save(self, path):
        """ saves current model parameters

        Parameters
        ----------
        path : str
            Path to output root directory. In the subfolder 'modelparams', the model is stored.
            If segmentation was performed, the results are stored in the subfolder 'summary'.
        """
        self.model.save(os.path.join(path, 'modelparams'))
        self.labels_.to_csv(os.path.join(path, 'modelparams', 'labels.csv'), index=False)

        if hasattr(self, "_segments") and self._segments is not None:
            export_segmentation(self._segments, os.path.join(path, 'summary',
                                     'segmentation.tsv'), 0.0)

    @classmethod
    def load(cls, path):
        """ Load model parameters from file.

        Parameters
        ----------
        path : str
            path to result root folder. This folder in turn
            is expected to contain the model parameters in '{path}/modelparams/dirmulhmm.npz'.
            Moreover, the segmentation results are obtained from '{path}/summary/segmentation.tsv'
            if available.

        Returns
        -------
        Scregseg object
        """

        if os.path.exists(os.path.join(path, 'modelparams', 'dirmulhmm.npz')):
            model = DirMulHMM.load(path)
        else:
            raise ValueError("Model not available")

        scmodel = cls(model)

        if os.path.exists(os.path.join(path, 'modelparams',  'labels.csv')):
            scmodel.labels_ = pd.read_csv(os.path.join(path, 'modelparams', 'labels.csv'))

        scmodel.load_segments(os.path.join(path, 'summary', 'segmentation.tsv'))

        return scmodel

    def load_segments(self, path):
        """ Load segmentation from file if available.

        Parameters
        ----------
        path : str
            path to segmentation results.
        """
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

        X_, _ = get_labeled_data(X)

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
        the accessible sites in each cell.

        Similar to the enrichment test which is used for defined features sets,
        here the enrichment is considered across all accessible sites in the cell.

        Parameters
        ----------
        X : list of count matrices
            Input count matrices
        mode : str
            Type of test to perform (see broadregion_enrichment).
        prob_max_threshold : float
            Consider state calls with a minimum posterior probability.
        post : bool
            Whether to use posterior decoding probability (soft-decision) or categorical state calls (hard-decision).
            Default: False. categorical state calls are used.
        """

        X_, _ = get_labeled_data(X)

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

    def plot_state_abundance(self):
        """
        plot state statistics
        """

        fig, ax = plt.subplots()
        state_counts = self.get_state_stats()

        sns.barplot(y=[l for l in state_counts.index],
                    x=state_counts,
                    ax=ax,
                    color="lightblue")
                    #palette=[self.color[i] for i in state_counts.index])

        return fig

    def plot_fragmentsize(self, frag):
        df =  pd.DataFrame(frag.cmat.toarray(), columns=frag.cannot.cell)
        df['name'] = self._segments.name
        adf = df.groupby('name').aggregate('mean')
        fig, ax =  plt.subplots()
        sns.heatmap(sdf, ax=ax)


    def plot_readdepth(self):
        """
        plots read depths associated with states
        """
        fig, axes = plt.subplots()
        segs = self._segments.copy()
        segs['log_readdepth'] = np.log10(segs['readdepth'] + 1)
        sns.violinplot(x="log_readdepth", y='name', data=segs, orient='h', ax=axes, color="lightblue")
        fig.tight_layout()
        return fig

    def plot_normalized_emissions(self, idat=None, selectedstates=None):
       """ Plot background normalized emission probabilities.

       Parameters
       ----------
       idat : int
           index of the idat's countmatrix

       Returns
       -------
       fig :
           sns.clustermap figure object
       """
       if idat is None:
           em = np.concatenate([es + ep for es, ep in zip(self.model.emission_suffstats_, self.model.emission_prior_)], axis=1)
       else:
           em = self.model.emission_suffstats_[idat] + self.model.emission_prior_[idat]

       nem = em / em.sum(1, keepdims=True)
       tem = em.sum(0, keepdims=True)/em.sum()
       lodds = np.log(nem) - np.log(tem)

       if hasattr(self, "labels_"):
           if idat is None:
               l = self.labels_.label
           else:
               l = self.labels_[self.labels_.matrixid == idat].label
       else:
           l = [str(i) for i in range(lodds.shape[1])]

       df = pd.DataFrame(lodds, columns=l, index=self.to_statenames(np.arange(self.n_components))).T

       if selectedstates is not None:
           df = df[selectedstates]

       g = sns.clustermap(df, center=0., robust=True, cmap='RdBu_r', figsize=(15,15))
       g.ax_heatmap.set_ylabel('Features')
       g.ax_heatmap.set_xlabel('States')

       return g

    @property
    def color(self):
        return self._color


    def segment(self, X, regions=None):
        """
        Performs segmentation.

        The result of this method is the _segments results
        pd.DataFrame.

        Parameters
        ----------
        X : list(CountMatrix) or list(np.array) or list(scipy.sparse.csc_matrix)
            List of count matrices for which state calling is performed
        regions : pd.DataFrame or None
            Dataframe containing the genomic intervals (e.g. from a bed file).
            If None, the regions are extracted from the countmatrix object.

        """
        if not isinstance(X, list):
            X = [X]
        X_, _ = get_labeled_data(X)
        if isinstance(X[0], CountMatrix):
            regions_ = X[0].regions.copy()
        else:
            bed = BedTool(regions)

            regions_ = pd.DataFrame([[iv.chrom, iv.start, iv.end] for iv in bed],
                                    columns=['chrom', 'start', 'end'])

        statenames = self.to_statenames(self.model.predict(X_))
        statescores = self.model.predict_proba(X_)

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
            regions_['readdepth'] = X_[i].sum(1)

        self._segments = regions_
        cleanup()

    def annotate(self, annotations):
        """Annotate the bins with BED, BAM or BIGWIG files.

        For each feature file, the feature coverage scores are computed
        for the genome-wide bins.
        The resulting signals are added as new columns to the _segments
        DataFrame that is maintained by the object.

        Parameters
        ----------
        annotations : dict(featurename: filename)
            Dictionary of annotations name:file pairs.
        """
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
        # load bed or bedgraph
        fpairs = [(key, file) for key, file in annotationdict.items() if (file.endswith('.bed') or \
                                                              file.endswith('.bed.gz') or file.endswith('.narrowPeak') or \
                                                              file.endswith('.bedgraph'))]
        files = [f[1] for f in fpairs]
        labels = [f[0] for f in fpairs]

        if len(files) > 0:
            cov = Cover.create_from_bed('bedfiles', bedfiles=files, roi=tmpfilename, conditions=labels,
                                        binsize=binsize, resolution=binsize, store_whole_genome=False,
                                        cache=True)
            for i, label in enumerate(labels):
                self._segments[label] = cov.garray.handle['data'][:, 0, 0, i]

        # load bam file coverage
        fpairs = [(key, file) for key, file in annotationdict.items() if file.endswith('.bam')]
        files = [f[1] for f in fpairs]
        labels = [f[0] for f in fpairs]

        if len(files) > 0:
            cov = Cover.create_from_bam('bamfiles', bamfiles=files, roi=tmpfilename, conditions=labels,
                                        stranded=False, normalizer=[LogTransform()],
                                        store_whole_genome=False, binsize=binsize, resolution=binsize,
                                        cache=True)
            for i, label in enumerate(labels):
                self._segments[label] = cov.garray.handle['data'][:, 0, 0, i]

        # load bigwig file coverage
        fpairs = [(key, file) for key, file in annotationdict.items() if (file.endswith('.bw') or file.endswith('.bigwig'))]
        files = [f[1] for f in fpairs]
        labels = [f[0] for f in fpairs]

        if len(files) > 0:
            cov = Cover.create_from_bigwig('bigwigfiles',
                                           bigwigfiles=files,
                                           roi=tmpfilename, conditions=labels,
                                           binsize=binsize, resolution=binsize,
                                           store_whole_genome=False,
                                           cache=True)
            for i, label in enumerate(labels):
                self._segments[label] = cov.garray.handle['data'][:, 0, 0, i]

        os.remove(tmpfilename)
        os.rmdir(tmpdir)

    @property
    def n_components(self):
        return self.model.n_components

    def geneset_observed_state_counts(self, genesets, flanking=50000, using_tss=True):
        """ Collect observed states for a set of gene sets.

        Parameters
        ----------
        genesets : dict(str)
            genesets represents a directory containing a set of bed files.
            The keys represent the respective labels.
        flanking : int
            determines the flanking window by which each feature is extended up- and down-stream.
            Default: 50000
        using_tss : bool
            Determines whether to restrict attention to the surrounding of the TSS.
            In this case, the flanking extends the 5' ends of the genes/features.
            Alternatively, the entire feature/gene length is considered. Default: True.

        Returns
        -------
        tuple(pd.DataFrame[n_features, n_states], list(feature length for each feature), list(feature names))

        """
        if isinstance(genesets, dict):

            genesetnames = []
            genesetfile =[]
            for k in genesets:
                genesetnames.append(k)
                genesetfile.append(genesets[k])
            genesets = genesetfile
        else:
            genesetnames = [os.path.basename(x) for x in genesets]

        ngsets = len(genesets)

        print('Segmentation enrichment for {} states and {} sets.'.format(self.n_components, ngsets))
        tmpdir = tempfile.mkdtemp()
        filename = 'dummyexport'
        tmpfilename = os.path.join(tmpdir, filename)
        self._segments.to_csv(tmpfilename, sep='\t', header=False, index=False, columns=['chrom', 'start', 'end', 'name'])

        segment_bedtool = BedTool(tmpfilename)

        observed_segmentcounts = np.zeros((ngsets, self.n_components))
        geneset_length = np.zeros(ngsets)

        for igeneset, geneset in enumerate(genesets):

            if using_tss:
                # obtain tss's for genes
                tss = BedTool([Interval(iv.chrom, iv.end if iv.strand=='-' else iv.start,
                                        (iv.end if iv.strand=='-' else iv.start) + 1,
                                        name=iv.name) for iv in BedTool(geneset)]).sort()
            else:
                tss = BedTool(geneset).sort()

            #flank tss's by flanking window
            flanked_tss_ = BedTool([Interval(iv.chrom, max(0, iv.start-flanking), iv.end+flanking, name=iv.name) for iv in tss])

            # collect segments in the surounding region
            roi_segments = segment_bedtool.intersect(flanked_tss_, wa=True, u=True)

            geneset_length[igeneset] = len(roi_segments)
            # obtain segment counts

            for iv in roi_segments:
                observed_segmentcounts[igeneset, self.to_stateid(iv.name)] += 1

        obscntdf = pd.DataFrame(observed_segmentcounts, columns=self.to_statenames(np.arange(self.n_components)),
                             index=genesetnames)
        os.remove(tmpfilename)
        os.rmdir(tmpdir)
        return obscntdf, geneset_length, genesetnames

    def observed_state_counts(self, regions, flanking=50000, using_tss=True):
        """ Collect observed states for a set of regions individually.

        Parameters
        ----------
        regions : bed file
            regions represents a single bed files.
            An enrichment test is performed for each interval individually.
        flanking : int
            determines the flanking window by which each feature is extended up- and down-stream.
            Default: 50000
        using_tss : bool
            Determines whether to restrict attention to the surrounding of the TSS.
            In this case, the flanking extends the 5' ends of the genes/features.
            Alternatively, the entire feature/gene length is considered. Default: True.

        Returns
        -------
        tuple(pd.DataFrame[n_features, n_states], list(feature length for each feature), list(feature names))
        """

        if isinstance(regions, str) and os.path.exists(regions):
            regions = BedTool(regions)


        regionnames = ['{}:{}-{}:({})'.format(iv.chrom, iv.start, iv.end, iv.name) for iv in regions]
        #regionnames = [iv.name for iv in regions]
        reg2id = {iv.name: i for i, iv in enumerate(regions)}

        nregions = len(regionnames)

        print('Segmentation enrichment for {} states and {} regions.'.format(self.n_components, nregions))

        tmpdir = tempfile.mkdtemp()
        filename = 'dummyexport'
        tmpfilename = os.path.join(tmpdir, filename)
        self._segments.to_csv(tmpfilename, sep='\t', header=False, index=False, columns=['chrom', 'start', 'end', 'name'])
        segment_bedtool = BedTool(tmpfilename)

        observed_segmentcounts = np.zeros((nregions, self.n_components))
        region_length = np.zeros(nregions)

        if using_tss:
            tss = BedTool([Interval(iv.chrom, iv.end if iv.strand=='-' else iv.start,
                                    (iv.end if iv.strand=='-' else iv.start) + 1,
                                    name=iv.name) for iv in regions])
        else:
            tss = regions

        #flank tss's by flanking window
        flanked_tss_ = BedTool([Interval(iv.chrom, max(0, iv.start-flanking), iv.end+flanking, name=iv.name) for iv in tss])

        # collect segments in the surounding region
        roi_segments = flanked_tss_.intersect(segment_bedtool, wa=True, wb=True)

        for iv in roi_segments:
            observed_segmentcounts[reg2id[iv.name], self.to_stateid(iv.fields[-1])] += 1

        region_length = observed_segmentcounts.sum(-1)


        obscntdf = pd.DataFrame(observed_segmentcounts, columns=self.to_statenames(np.arange(self.n_components)),
                             index=regionnames)
        os.remove(tmpfilename)
        os.rmdir(tmpdir)
        return obscntdf, region_length, regionnames

    def broadregion_enrichment(self, state_counts, featurenames=None, mode='logfold'):
        """ Broad region enrichment test.

        The enrichment test  determines whether a given region exhibits an excesss of calls for a
        particular state.

        Parameters
        ----------
        state_counts : np.array[n_features, n_states] or pd.DataFrame
            Observed state counts in a region or a set of regions.
        regionslengths :
            Total number of bins representing the regions.
        featurenames : list(str) or None
            Feature or region names.
        mode : str
            Type of enrichment test: 'logfold', 'fold', 'chisqstat', 'log10pvalue'.
            logfold determines logarithm of the observed state counts over the expected state counts.
            fold determines the observed state counts over the expected state counts.
            chisqstat determines (observed state counts - expected state counts ) **2 / (expected state counts) **2
            pvalue or log10pvalue computes the negative log10 pvalue for for a given state. This achieved
            by computing the null distribution of state counts over a window of the same size
            using dynamic programming and by making use of the models transition probabilities.
            The pvalue method is well suited for testing enrichment in short stretches, because its runtime
            depends on the number of bins. On the other hand, the remaining options are suited for long
            sequence stretches.

        Returns
        -------
        pd.DataFrame[n_features, n_states] :
           Table of enrichment test results
        """
        stateprob = self.model.get_stationary_distribution()
        if isinstance(state_counts, pd.DataFrame):
            featurenames = state_counts.index
            state_counts = state_counts.values

        regionlengths = state_counts.sum(1)
        enr = np.zeros_like(state_counts)

        e = np.outer(regionlengths, stateprob)

        if mode == 'logfold':
            enr = np.log10(np.where(state_counts==0, 1, state_counts)) - np.log10(np.where(state_counts==0, 1, e))
        if mode == 'fold':
            enr = np.where(state_counts==0, 1, state_counts)/np.where(state_counts==0, 1, e)
        elif mode == 'chisqstat':
            stat = np.where((state_counts - e) >= 0.0, (state_counts - e), 0.0)

            enr = stat**2 / e**2
        elif mode in ['pvalue', 'log10pvalue']:
            self._reset_broadregion_null_distribution()
            self._make_broadregion_null_distribution(state_counts.sum(1))
            for ireg, scnt in enumerate(state_counts.sum(1)):
                if scnt < 1:
                    continue

                null_dist, _ = self._get_broadregion_null_distribution(int(scnt))

                for istate in range(self.n_components):
                    n_obs = int(state_counts[ireg, istate])
                    enr[ireg, istate] = -min(np.log10(max(0.0, null_dist[n_obs:, istate].sum())), 15)

        enrdf = pd.DataFrame(enr, columns=self.to_statenames(np.arange(self.n_components)),
                             index=featurenames)

        return enrdf


    def _reset_broadregion_null_distribution(self):
        self._cnt_storage = {}
        self._cnt_conditional = {}
        #self._fft_init_cnt_dist = {}

    def _init_broadregion_null_distribution(self, max_len):
        max_len = int(max_len)
        cnt_dist = np.zeros((max_len+1, self.n_components, self.n_components))

        stationary = self.model.get_stationary_distribution()

        # initialize array with stationary distribution
        for dostate_i in range(self.n_components):
            for curr_i in range(self.n_components):
                shift = 1 if dostate_i == curr_i else 0

                cnt_dist[shift, dostate_i, curr_i] += stationary[curr_i]

        # FFT of stationary distribution
        fft_cnt_dist = np.empty((max_len+1, self.n_components, self.n_components), dtype='complex128')
        for dostate_i in range(self.n_components):
            for curr_i in range(self.n_components):
               fft_cnt_dist[:, dostate_i, curr_i] = np.fft.fft(cnt_dist[:, dostate_i, curr_i])

        # P(cnt, associated_state, observed_state)
        self._fft_init_cnt_dist = fft_cnt_dist


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

        #make conditional distribution for the remained of the sequence length
        for length in keep_lengths:
            self._make_broadregion_conditional_null_distribution(length - 1, max_length)

        for length in keep_lengths:
            if length in self._cnt_storage:
                continue
            if length < 1:
                continue
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
#        if keep_lengths.max() in self._cnt_storage:
#            # already computed, use cache
#            return

        self._init_broadregion_null_distribution(keep_lengths.max())
        self._finalize_broadregion_null_distribution(keep_lengths)

    def _get_broadregion_null_distribution(self, length):
        return self._cnt_storage[length], \
               self._cnt_storage[length].T.dot(np.arange(self._cnt_storage[length].shape[0]))

    def get_subdata(self, X, query_states, collapse_neighbors=True, state_prob_threshold=0.99):
        """ function deprecated: use get_statecalls() """
        X_, _ = get_labeled_data(X)

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
