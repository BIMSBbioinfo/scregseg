import pandas as pd
import numpy as np
import seaborn as sns
import tempfile
from pybedtools import BedTool
from hmmlearn.hmm import MultiModalMultinomialHMM
from scipy.stats import binom
from scipy.stats import norm
from scipy.sparse import csc_matrix
import pickle
import os
from janggu.data import Cover
from janggu.data import LogTransform


class Scseg(object):

    def __init__(self, model):
        self.model = model
        self._color = {'cluster_{}'.format(i): el for i, el in \
                       enumerate(sns.color_palette('bright', self.n_components))}
        self._segments = None

    def save(self, path):
        np.savez(path, [self.model.transmat_, self.model.startprob_] + self.model.emissionprobs)


    @classmethod
    def load(cls, path):
        npzfile = np.load(path)

        trans = npzfile['arr_0']
        start = npzfile['arr_1']
        emissions = [npzfile[file] for file in npzfile.files[2:]]

        model = MultiModalMultinomialHMM(len(start))
        model.transmat_ = trans
        model.startprob_ = start
        model.emissionprobs = emission

        return cls(model)


    def predict(self, data):
        return ['cluster_{}'.format(state) for state in self.model.predict(data)]

    def cell2state_enrichment(self, data):
        """ Determines whether a states is overrepresented among
        the accessible sites in a given cellself.

        The P-value is determined using the binomial test
        and the log-fold-change is determine by Obs[state proportion]/Expected[state proportion].
        """

        states = self.model.predict(data)

        Z = csc_matrix((np.ones(len(states)),
                       (states.astype('int'), np.arange(len(states),dtype='int'))))
        expected_segfreq = np.asarray(Z.sum(1)/Z.sum())

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

    @staticmethod
    def clustermap_fold():
        pass

    @staticmethod
    def distplot_folds(logfoldenr, state):
        fig, axes = plt.subplots(len(logfoldenr))
        for ax, fold in zip(axes, logfoldenr):
            sns.distplot(fold[state], ax=ax)
        return fit

    @property
    def color(self):
        return self._color

    def segment(self, data, regions):
        """ determine segments """

        bed = BedTool(regions)

        regions_ = pd.DataFrame([[iv.chrom, iv.start, iv.end] for iv in bed],
                                columns=['chrom', 'start', 'end'])

        statenames = self.predict(data)

        regions_['name'] = statenames
        regions_['score'] = 1
        regions_['strand'] = '.'
        regions_['thickStart'] = regions_.start
        regions_['thickEnd'] = regions_.end
        regions_['itemRbg'] = ['{},{},{}'.format(int(self.color[state][0]*255),
                                                int(self.color[state][1]*255),
                                                int(self.color[state][2]*255)) for state in statenames]

        self._segments = regions_

    def export_segments(self, filename, individual_beds=False):
        if self._segments is None:
            raise ValueError("No segmentation results available. Please run segment(data, regions) first.")

        if individual_beds:
            for comp in range(self.n_components):
                regions[regions.name == 'cluster_{}'.format(comp)].to_csv(filename[:-4] + '_cluster{}.bed'.format(comp), sep='\t',
                                     header=False, index=False,
                                     columns=['chrom', 'start', 'end', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRbg'])

        else:
            self._segments.to_csv(filename, sep='\t',
                           header=False, index=False,
                           columns=['chrom', 'start', 'end', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRbg'])

    def annotate(self, annotations):
        """Annotate the bins with BED, BAM or BIGWIG files."""
        tmpdir = tempfile.mkdtemp()
        filename = 'dummyexport'
        tmpfilename = os.path.join(tmpdir, filename)
        self.export_segments(tmpfilename)

        if not isinstance(annotations, (list, dict)):
            annotations = [annotations]

        if isinstance(annotations, list):
            annotationdict = {os.path.basename(annot) for annot in annotations}
        else:
            annotationdict = annotations

        binsize = BedTool(tmpfilename)[0].length
        print("Binsize={} detected".format(binsize))

        """Annotate the individual regions."""
#        seg = BedTool(tmpfilename)
        for key, file in annotationdict.items():
            if file.endswith('.bed') or file.endswith('.bed.gz') or file.endswith('.narrowPeak'):
                cov = Cover.create_from_bed(key, bedfiles=file, roi=tmpfilename,
                                            binsize=binsize, resolution=binsize, store_whole_genome=True,
                                            cache=True)
            elif file.endswith('.bam'):
                cov = Cover.create_from_bam(key, bamfiles=file, roi=tmpfilename,
                                            stranded=False, normalizer=[LogTransform()],
                                            store_whole_genome=True, binsize=binsize, resolution=binsize,
                                            cache=True)
            elif file.endswith('.bw') or file.endswith('.bigwig'):
                cov = Cover.create_from_bigwig(key, bigwigfiles=file, roi=tmpfilename,
                                               binsize=binsize, resolution=binsize, store_whole_genome=True,
                                               cache=True)
            self._segments[key] = pd.Series(cov[:][:, 0, 0, 0])

        os.remove(tmpfilename)
        os.rmdir(tmpdir)

    @property
    def n_components(self):
        return self.model.n_components

    def region2state_enrichment(self):
        # raise NotImplemented("")
        # total feature length
        f_len = 10000
        cur_distr = np.zeros((f_len, self.n_components, self.n_components))

        for ibin in range(f_len):
            for to_state in range( self.n_components):
                for from_state in range( self.n_components):
                    cur_distr[:, :]
