import pandas as pd
import numpy as np
import seaborn as sns
import tempfile
from pybedtools import BedTool
from pybedtools import Interval
from pybedtools.helpers import cleanup
from hmmlearn.hmm import MultiModalMultinomialHMM
from scipy.stats import binom
from scipy.stats import norm
from scipy.sparse import csc_matrix
import pickle
import os
import gzip
from janggu.data import Cover
from janggu.data import LogTransform
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import hypergeom
from sklearn.decomposition import LatentDirichletAllocation


class Scseg(object):

    _enr = None

    def __init__(self, model):
        self._segments = None
        self._nameprefix = 'cluster_'
        self.model = model
        self._color = {name: el for name, el in \
                       zip(self.to_statenames(np.arange(self.n_components)), sns.color_palette('bright', self.n_components))}

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        savelist = [self.model.transmat_, self.model.startprob_] + self.model.emissionprob_
        np.savez(path, *savelist)


    @classmethod
    def load(cls, path):
        npzfile = np.load(path)

        trans = npzfile['arr_0']
        start = npzfile['arr_1']
        emissions = [npzfile[file] for file in npzfile.files[2:]]

        model = MultiModalMultinomialHMM(len(start))
        model.transmat_ = trans
        model.startprob_ = start
        model.emissionprob_ = emissions
        model.n_features = [e.shape[1] for e in emissions]

        return cls(model)


    def to_statenames(self, states):
        return [self.to_statename(state) for state in states]

    def to_statename(self, state):
        return '{}{}'.format(self._nameprefix, state)

    def to_stateid(self, statename):
        return int(statename[len(self._nameprefix):])

    def cell2state_enrichment(self, data):
        """ Determines whether a states is overrepresented among
        the accessible sites in a given cellself.

        The P-value is determined using the binomial test
        and the log-fold-change is determine by Obs[state proportion]/Expected[state proportion].
        """

        stateprob = self.model.get_stationary_distribution()
        
        expected_segfreq = stateprob * data.shape[0]

        states = self.model.predict(data)

        Z = csc_matrix((np.ones(len(states)),
                       (states.astype('int'), np.arange(len(states),dtype='int'))))

        logfoldenr = []
        logpvalues = []

        for d in data:
            d_ = d.copy()
            d_[d_>0] = 1

            # adding a pseudo-count of 1 to avoid division by zero
            obs_seqfreq = Z.dot(d_).toarray() + 1.
            obs_seqfreq = obs_seqfreq / obs_seqfreq.sum(0, keepdims=True)

            # log fold-change
            fold = np.log10(obs_seqfreq) - np.log10(expected_segfreq)
            logfoldenr.append(fold)

            ff = fold.flatten()
            # standard deviation estimated from the left-half of the distribution
            scale = np.sqrt(np.mean(np.square(ff[ff<0])))


            rnorm = norm(0.0, scale)
            logpvalues.append(rnorm.logsf(fold))

        return logfoldenr, logpvalues

    def plot_state_statistics(self):
        if self._segments is None:
            raise ValueError("No segmentation results available. Please run segment(data, regions) first.")

        fig, ax = plt.subplots(1, 2)
        state_counts=pd.Series(self._segments.name).value_counts()

        sns.barplot(y=[l for l in state_counts.index], 
                    x=state_counts, ax=ax[0],
                    palette=[self.color[i] for i in state_counts.index])

        sns.heatmap(self.model.transmat_, ax=ax[1], cmap='Reds')

        return fig

    def plot_readdepth(self):
        fig, axes = plt.subplots(1,len(self.model.emissionprob_))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for i, ax in enumerate(axes):
            sns.boxplot(x="readdepth_"+str(i), y='name', data=self._segments, orient='h', ax=ax)
        return fig
                    
    def plot_logfolds_dist(self, logfoldenr, query_state=None):
        fig, axes = plt.subplots(len(logfoldenr))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for ax, fold in zip(axes, logfoldenr):

            if query_state is not None:
                istate = self.to_stateid(query_state)
                fold = fold[istate]

            ff = fold.flatten()
            sns.distplot(ff, ax=ax)

            x = np.linspace(-1.5, 2.5)
            scale = np.sqrt(np.mean(np.square(ff[ff<0.0])))
            ax.plot(x, norm(0, scale).pdf(x), 'R')

        return fig

    @property
    def color(self):
        return self._color

    def segment(self, data, regions):
        """ determine segments """

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


    def export_bed(self, filename, individual_beds=False, prob_max_threshold=0.99):

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if self._segments is None:
            raise ValueError("No segmentation results available. Please run segment(data, regions) first.")

        if individual_beds:
            self._single_bed = []
            for comp in range(self.n_components):
                self._segments[(self._segments.name == self.to_statename(comp)) & (self._segments.Prob_max >=prob_max_threshold)].to_csv(
                    filename + self.to_statename(comp) + '.bed', sep='\t',
                    header=False, index=False,
                    columns=['chrom', 'start', 'end',
                             'name', 'score', 'strand',
                             'thickStart', 'thickEnd', 'itemRbg'])

        else:
            self._segments[(self._segments.Prob_max >=prob_max_threshold)].to_csv(filename, sep='\t',
                           header=False, index=False,
                           columns=['chrom', 'start', 'end',
                                    'name', 'score', 'strand',
                                    'thickStart', 'thickEnd', 'itemRbg'])

#    def export_collapsed_data(self, data, filename, prob_max_threshold=0.99, logfold_threshold=0.5):
#        os.makedirs(os.path.dirname(filename), exist_ok=True)
#        if self._segments is None:
#            raise ValueError("No segmentation results available. Please run segment(data, regions) first.")
#        logfold, _ = self.cell2state_enrichment(data)
#
#        Cover.create_from_array('save', 
#            for comp in range(self.n_components):
#                self._segments[(self._segments.name == self.to_statename(comp)) & (self._segments.Prob_max >= prob_max_threshold)].to_csv(
#                    filename + self.to_statename(comp) + '.bed', sep='\t',
#                    header=False, index=False,
#                    columns=['chrom', 'start', 'end',
#                             'name', 'score', 'strand',
#                             'thickStart', 'thickEnd', 'itemRbg'])
#

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
            if file.endswith('.bed') or file.endswith('.bed.gz') or file.endswith('.narrowPeak'):
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

    def geneset_enrichment(self, segments, genesets, genes, prob_max_threshold=0.99):
        """Obtain the state counts along a feature annotation."""

        if isinstance(genes, str) and os.path.exists(genes):
            genes = BedTool(genes)

        _segments = segments

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

    def sample_enrichment(self, ref_anchors, samplefiles, samplenames, other_anchors=None, prob_max_threshold=0.99):
        if not isinstance(ref_anchors, BedTool):
            ref_anchors = BedTool(ref_anchors)
        ref_anchors = BedTool([Interval(iv.chrom, iv.start, iv.end, name=str(i)) for i, iv in enumerate(ref_anchors)])

        state2refanchor = np.zeros((self.n_components, len(ref_anchors)))
        states = self._segments.name.unique()
        
        for istate in range(self.n_components):
            state = self.to_statename(istate)
            segments = self._segments[(self._segments.name == state) & (self._segments.Prob_max >= prob_max_threshold)]
            segbed = BedTool([Interval(row.chrom, row.start, row.end) for _, row in segments.iterrows()]).sort().merge()

            overlapbed = ref_anchors.intersect(segbed, wa=True)
            state2refanchor[istate, np.asarray([int(iv.name) for iv in overlapbed])] = 1
        
        if hasattr(samplefiles, 'shape') and samplefiles.shape[0] == len(ref_anchors):
            featuremat = samplefiles
        else:
            featuremat = np.zeros((len(ref_anchors), len(samplefiles)))
            if other_anchors is not None and not isinstance(other_anchors, BedTool):
                other_anchors = BedTool(other_anchors)

            other_anchors = BedTool([Interval(iv.chrom, iv.start, iv.end, name=str(i)) for i, iv in enumerate(other_anchors)])
            for isample, sample in enumerate(samplefiles):
                if not isinstance(sample, BedTool):
                    sample = BedTool(sample)
                overlap = other_anchors.intersect(sample, wa=True)
                for iv in overlap:
                    featuremat[int(iv.name), isample] = 1

        # start with the enrichment test

        overlap = state2refanchor.dot(featuremat)

        enr = np.zeros_like(overlap)

        ns = state2refanchor.sum(1)
        Ns = featuremat.sum(0)
        ntotal = (state2refanchor.sum(0)>0).sum()

        for istate in range(self.n_components):
            for ifeature in range(featuremat.shape[1]):
                enr[istate, ifeature] = -hypergeom.logsf(overlap[istate, ifeature]-1, ntotal, ns[istate], Ns[ifeature])

        enr = pd.DataFrame(enr,
                           index=self.to_statenames(np.arange(self.n_components)),
                           columns=samplenames)
        enr = enr.T

        return enr


    def broadregion_enrichment(self, regions):
        if isinstance(regions, str) and os.path.exists(regions):
            regions = BedTool(regions)
        
        region2state_enr_logpvalue = np.zeros((len(regions), self.n_components))
        region2state_enr_logfold = np.zeros((len(regions), self.n_components))
        region2state_enr_fold = np.zeros((len(regions), self.n_components))
        region2state_cnts = np.zeros((len(regions), self.n_components))

        maxlen = 0
        for ireg, region in enumerate(regions):
            regsegs = self._segments[(self._segments.chrom==region.chrom) &
                                     (self._segments.start>=region.start) &
                                     (self._segments.end<=region.end)]
            if maxlen < regsegs.shape[0]:
                maxlen = regsegs.shape[0]

        self._make_broadregion_null_distribution(maxlen)

        for ireg, region in enumerate(regions):
            regsegs = self._segments[(self._segments.chrom==region.chrom) &
                                     (self._segments.start>=region.start) &
                                     (self._segments.end<=region.end)]
            reg_length = regsegs.shape[0]

            null_dist, expected_counts = self._get_broadregion_null_distribution(reg_length)

            for istate in range(self.n_components):
                n_obs = regsegs[regsegs.name == self.to_statename(istate)].shape[0]           
                region2state_enr_logpvalue[ireg, istate] = -np.log(null_dist[n_obs:, istate].sum())
                region2state_enr_logfold[ireg, istate] = np.log(n_obs/expected_counts[istate])
                region2state_enr_fold[ireg, istate] = (n_obs-expected_counts[istate])/expected_counts[istate]
                region2state_cnts[ireg, istate] = n_obs

        return region2state_enr_logpvalue, region2state_enr_logfold, region2state_enr_fold
        # obtain the observed number of states 


    def _make_broadregion_null_distribution(self, length, keep_lengths):
        cnt_storage = {}
        cnt_dist = np.zeros((length + 1, self.n_components, self.n_components))

        stationary = self.model.get_stationary_distribution()

        # initialize array
        for dostate_i in range(self.n_components):
            for curr_i in range(self.n_components):
                shift = 1 if dostate_i == curr_i else 0
                
                cnt_dist[shift, dostate_i, curr_i] += stationary[curr_i]
        if 1 in keep_lengths:
            cnt_storage[0] = cnt_dist

        for icnt in range(length-1):
            joint_cnt_dist = np.zeros_like(cnt_dist)
            for dostate_i in range(self.n_components):
                for prev_i in range(self.n_components):
                    for curr_i in range(self.n_components):
                        shift = 1 if dostate_i == curr_i else 0
                        if dostate_i == curr_i:
                            joint_cnt_dist[1:, dostate_i, curr_i] += cnt_dist[:-1, dostate_i, prev_i] *self.model.transmat_[prev_i, curr_i]
                        else:
                            joint_cnt_dist[:, dostate_i, curr_i] += cnt_dist[:, dostate_i, prev_i] *self.model.transmat_[prev_i, curr_i]
            cnt_dist = joint_cnt_dist
            if (icnt+1) in keep_lengths:
                cnt_storage[icnt+1] = cnt_dist

        
        self._cnt_dist = {k: cnt_storage[k].sum(-1) for k in cnt_storage}
        for k in self._cnt_dist:
            np.testing.assert_allclose(self._cnt_dist[k].sum(), self.n_components)

    def _get_broadregion_null_distribution(self, length):
        return self._cnt_dist[length-1], self._cnt_dist[length-1].T.dot(np.arange(self._cnt_dist.shape[1]))


    @property
    def state_annotation(self):
        if self._enr is None:
            raise ValueError("State annotation enrichment not yet performed. Use geneset_enrichment")
        return self._enrfeaturemat

    def subcluster(self, data, query_states, n_subclusters, 
                   logfold_threshold, state_prob_threshold=0.99):
                   
        sdata, subdf, mapelem = self.get_subdata(data, query_states, logfold_threshold, state_prob_threshold)

        print(sdata.shape)
        labels, label_probs = self.do_subclustering(sdata, n_subclusters)
        x = ['subcluster_{}'.format(labels[mapelem[i]]) for i in range(len(subdf.index))]

        subdf.name = x
        return subdf, sdata

    def get_subdata(self, data, query_states, logfold_threshold, state_prob_threshold=0.99):

        if not isinstance(query_states, list):
            query_states = [query_states]

        if not isinstance(logfold_threshold, list):
            logfold_threshold = [logfold_threshold]

        subset = self._segments[(self._segments.name.isin(query_states)) 
                                & (self._segments.Prob_max >= state_prob_threshold)].copy()
        
        # determine neighboring bins that can be merged together
        prevind = -2
        nelem = 0
        mapelem = []
        for i in subset.index:
            nelem += 1
            if i == (prevind + 1):
               nelem -= 1
            mapelem.append(nelem - 1)
            prevind = i

        logfolds, _ = self.cell2state_enrichment(data)
    
        submats = []
        mapelems = []
        for datum, logfold, th in zip(data, logfolds, logfold_threshold):
            logfold = logfold[[self.to_stateid(query_state) for query_state in query_states]]

            keepcells = np.where(logfold >= th)[0]

            submat = np.zeros((nelem, len(keepcells)))

            sm = np.asarray(datum[subset.index][:, keepcells].todense())

            for i, i2m in enumerate(mapelem):
                submat[i2m, :] += sm[i]
            submats.append(submat)
            mapelems.append(mapelem)

        return submats, subset, mapelems

    def do_subclustering(self, submat, n_subclusters, method='mixture'):
        if method=='mixture':

            clust = MixtureOfMultinomials(n_subclusters, random_state=32)
        else:
            clust = LDAWrapper(n_subclusters)

        clust.fit(submat)

        tr = clust.transform(submat)
        labels = clust.predict(submat)
        return labels, tr


from sklearn.utils import check_random_state
from scipy.misc import logsumexp
from hmmlearn.utils import normalize

class MixtureOfMultinomials:
    def __init__(self, n_clusters, n_iters=10, random_state=None, verbose=False):
        self.n_clusters = n_clusters
        self.pi_prior = 1.
        self.component_prior = 1.
        self.n_iters = n_iters
        self.random_state = random_state
        self.verbose = verbose
     

    def fit(self, data):
        # init 
        self._init(data)
        self._check()
        if self.verbose: print('init', self.score(data))
        for iit in range(self.n_iters):
            p_z = self._e_step(data)
            np.testing.assert_allclose(p_z.sum(), data.shape[0])
            self._m_step(data, p_z)
            self._check()
            if self.verbose: print(iit, self.score(data))
    
    def _check(self):
        assert np.all(self.components > 0.0)
        assert np.all(self.pi > 0.0)
        np.testing.assert_allclose(self.components.sum(), self.n_clusters)
        np.testing.assert_allclose(self.pi.sum(), 1.)

    def _init(self, data):
        self.random_state_ = check_random_state(self.random_state)

        self.components = data[self.random_state_.permutation(data.shape[1])[:self.n_clusters]]*.2 + \
                          self.random_state_.rand(self.n_clusters, data.shape[1]) * .1 
        normalize(self.components, axis=1)

        self.pi = np.ones(self.n_clusters)

        normalize(self.pi)

    def predict(self, data):
        p_z = self._e_step(data)
        labels = np.argmax(p_z, axis=1)
        return labels

    def transform(self, data):
        return self._e_step(data)

    def _log_likeli(self, data):
        log_p_z = data.dot(np.log(self.components.T)) + np.log(self.pi)[None,:]
        return log_p_z

    def _e_step(self, data):
        log_p_z = self._log_likeli(data)
        log_p_z -= logsumexp(log_p_z, axis=1, keepdims=True)
        return np.exp(log_p_z)

    def _m_step(self, data, p_z):
        stats = p_z.T.dot(data) + self.component_prior
        self.components = stats
        normalize(self.components, axis=1)

        self.pi = p_z.sum(0) + self.pi_prior
        normalize(self.pi)

    def score(self, data):
        return logsumexp(self._log_likeli(data), axis=1).sum()

class LDAWrapper:
    def __init__(self, lda):
        self.lda = lda

    def fit(self, data):
        return self.lda.fit(data)

    def transform(self, data):
        return self.lda.transform(data)

    def score(self, data):
        return self.lda.score(data)

    def predict(self, data):
        return np.argmax(self.transform(data), axis=1)


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
