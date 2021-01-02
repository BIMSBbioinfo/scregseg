import os
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tempfile
import pandas as pd
try:
    from keras import layers
except:
    layers = None
    
try:
    import janggu
except:
    janggu = None
    
import numpy as np
from scipy.special import logsumexp

from sklearn.metrics import r2_score
from keras import Model
from keras.callbacks import EarlyStopping

from scregseg import Scregseg

class Meme:
    """Class maintains motifs and exports them to MEME format."""
    def __init__(self):
        self.ppms = []
        self.names = []

    def add(self, ppm, name=None):
        self.ppms.append(ppm)
        if name is None:
            self.names.append('motif_{}'.format(len(self.ppms)))
        else:
            self.names.append(name)

    def _header(self):
        return "MEME version 5\n\nALPHABET= ACGT\n\nstrand: + -\n\n"

    def _motif_header(self, name, length):
        return "MOTIF {name}\nletter-probability matrix: alength= 4 w= {length}\n".format(name=name, length=length)

    def _motif_body(self, ppm):
        s = ""
        for i in range(ppm.shape[0]):
            s+='\t'.join([str(e) for e in ppm[i]]) + '\n'
        s += '\n'
        return s

    def _allmotifs(self):
        s = ""
        for n, p in zip(self.names, self.ppms):
            s += self._motif_header(n, len(p))
            s += self._motif_body(p)
        return s

    def __str__(self):
        return self._header() + self._allmotifs()

    def __repr__(self):
        return str(self)

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(str(self))

if janggu is not None:
    @janggu.inputlayer
    @janggu.outputdense('linear')
    def cnn_model(inputs, inp, oup, params):
        with inputs.use('dna') as dna_in:
            layer = dna_in
        #layer = Dropout(.2)(layer)
        layer = janggu.DnaConv2D(layers.Conv2D(100, (21, 1), activation='sigmoid'))(layer)
        layer = layers.GlobalMaxPooling2D()(layer)
        layer = layers.Dropout(.5)(layer)
        return inputs, layer
    
    @janggu.inputlayer
    @janggu.outputdense('sigmoid')
    def cnn_model_binary(inputs, inp, oup, params):
        with inputs.use('dna') as dna_in:
            layer = dna_in
        layer = janggu.DnaConv2D(layers.Conv2D(100, (21, 1), activation='sigmoid'))(layer)
        layer = layers.GlobalMaxPooling2D()(layer)
        layer = layers.Dropout(.5)(layer)
        return inputs, layer


class MotifExtractor:
    """ Neural network motif extractor. """
    def __init__(self, scmodel, refgenome,
                 ntop=15000, nbottom=15000,
                 ngap=70000, flank=250, nmotifs=10,
                 cnn=None):
        if janggu is None:
            raise Exception("janggu is not available, but required for this functionality. "
                            " Please run: pip install janggu[tf2]")
        if keras is None:
            raise Exception("keras is not available, but required for this functionality. "
                            " Please run: pip install keras")
        self.ntop = ntop
        self.refgenome = refgenome
        self.nmotifs = nmotifs
        self.nbottom = nbottom
        self.ngap = ngap
        self.flank = flank
        self.scmodel = scmodel
        self.meme = Meme()

        if cnn is None:
            self.cnn = cnn_model
        else:
            self.cnn = cnn

    def extract_motifs(self):
        """ Perform motif extraction."""
        tmpdir = tempfile.mkdtemp()
        filename = 'labels.bedgraph'
        roi = os.path.join(tmpdir, filename)


        binsize = self.scmodel._segments.iloc[0].end - \
                  self.scmodel._segments.iloc[0].start

        for i in range(self.scmodel.n_components):

            process_state= 'state_{}'.format(i)
            logging.debug('processing: {}'.format(process_state))

            df = self.scmodel._segments

            df['pscore'] = df['Prob_{}'.format(process_state)] * df['readdepth']
            df.pscore.rolling(3, win_type='triang',
                              center=True).sum().fillna(0.0)

            N = self.ntop + self.nbottom + self.ngap
            sdf = pd.concat([df[['chrom',
                                 'start',
                                 'end',
                                 'pscore']].nlargest(self.ntop, 'pscore'),
                             df[['chrom',
                                 'start',
                                 'end',
                                 'pscore']].nlargest(N,
                                'pscore').iloc[-self.nbottom:]],
                            ignore_index=True)

            sdf[['chrom', 'start',
                 'end', 'pscore']].to_csv(roi, sep='\t',
                header=False, index=False)

            DNA = janggu.data.Bioseq.create_from_refgenome('dna',
                                               refgenome=self.refgenome,
                                               roi=roi,
                                               binsize=binsize,
                                               flank=self.flank,
                                               cache=False)

            LABELS = janggu.data.ReduceDim(janggu.data.Cover.create_from_bed('score',
                                                     bedfiles=roi,
                                                     roi=roi,
                                                     binsize=binsize,
                                                     cache=False,
                                                     resolution=binsize))

            # fit the model
            model = janggu.Janggu.create(self.cnn, (),
                                  inputs=DNA,
                                  outputs=LABELS,
                                  name='simple_cnn_{}'.format(process_state))

            model.compile(optimizer='adam', loss='mae')
            model.summary()

            hist = model.fit(DNA, LABELS,
                             epochs=300, batch_size=32,
                             validation_data=['chr1', 'chr5'],
                             callbacks=[EarlyStopping(patience=20, restore_best_weights=True)])

            # extract motifs
            W, b = model.kerasmodel.layers[1].get_weights()
            fmodel = Model(model.kerasmodel.inputs,
                           model.kerasmodel.layers[1].output)
            featureacts = fmodel.predict(DNA).max(axis=(1,2))

            fdf = pd.DataFrame(featureacts)
            fdf['score']= LABELS[:]

            ranking = fdf.corr(method='spearman')['score'].sort_values(ascending=False)
            for i, m in enumerate(ranking.index[1:(self.nmotifs+1)]):
                s = W[:, 0, :, m]
                s -= logsumexp(s, axis=1, keepdims=True)
                s = np.exp(s)
                self.meme.add(s, '{}_{}'.format(process_state, i))
        os.remove(roi)

    def save_motifs(self, output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
        self.meme.save(output)

class MotifExtractor2:
    """ Neural network motif extractor. """
    def __init__(self, scmodel, refgenome,
                 ntop=15000,
                 flank=250, nmotifs=10,
                 cnn=None):
        if janggu is None:
            raise Exception("janggu is not available, but required for this functionality. "
                            " Please run: pip install janggu[tf2]")
        if keras is None:
            raise Exception("keras is not available, but required for this functionality. "
                            " Please run: pip install keras")
        self.ntop = ntop
        self.refgenome = refgenome
        self.nmotifs = nmotifs
        self.flank = flank
        self.scmodel = scmodel
        self.meme = Meme()

        if cnn is None:
            self.cnn = cnn_model
        else:
            self.cnn = cnn

    def extract_motifs(self):
        """ Perform motif extraction."""
        logging.debug('extract motifs')
        tmpdir = tempfile.mkdtemp()
        roifilename = 'roi.bedgraph'
        roi = os.path.join(tmpdir, roifilename)

        binsize = self.scmodel._segments.iloc[0].end - \
                  self.scmodel._segments.iloc[0].start

        for i in range(self.scmodel.n_components):

            process_state= 'state_{}'.format(i)
            logging.debug('processing: {}'.format(process_state))

            df = self.scmodel._segments.copy()

            sdf = df.copy()
            sdf['pscore'] = sdf['Prob_{}'.format(process_state)] * sdf['readdepth']
            sdf['pscore'] = sdf.pscore.rolling(3, win_type='triang',
                              center=True).sum().fillna(0.0)

            spos = sdf[['chrom',
                        'start',
                        'end',
                        'pscore']].nlargest(self.ntop, 'pscore').copy()

            sneg = []
            for negstate in range(self.scmodel.n_components):
                sdf = df.copy()
                if negstate == i:
                    continue
                sdf['pscore'] = sdf['Prob_state_{}'.format(negstate)] * sdf['readdepth']
                sdf['pscore'] = sdf.pscore.rolling(3, win_type='triang',
                              center=True).sum().fillna(0.0)
                sdf = sdf.nlargest(2*self.ntop//self.scmodel.n_components, 'pscore')
                sdf.pscore = sdf.pscore * sdf['Prob_{}'.format(process_state)]
                sneg.append(sdf)
                
            sdf = pd.concat([spos] + sneg, ignore_index=True)

            sdf[['chrom', 'start', 'end', 'pscore']].to_csv(roi, sep='\t',
                header=False, index=False)

            DNA = janggu.data.Bioseq.create_from_refgenome('dna',
                                               refgenome=self.refgenome,
                                               roi=roi,
                                               binsize=binsize,
                                               flank=self.flank,
                                               cache=False)

            LABELS = janggu.data.ReduceDim(janggu.data.Cover.create_from_bed('score',
                                                     bedfiles=roi,
                                                     roi=roi,
                                                     binsize=binsize,
                                                     cache=False,
                                                     resolution=binsize))

            # fit the model
            model = janggu.Janggu.create(self.cnn, (),
                                  inputs=DNA,
                                  outputs=LABELS,
                                  name='simple_cnn2_{}'.format(process_state))

            model.compile(optimizer='adam', loss='mae')

            hist = model.fit(DNA, LABELS,
                             epochs=300, batch_size=32,
                             validation_data=['chr1', 'chr5'],
                             callbacks=[EarlyStopping(patience=20, restore_best_weights=True)])

            model.summary()
            # extract motifs
            W, b = model.kerasmodel.layers[1].get_weights()
            fmodel = Model(model.kerasmodel.inputs,
                           model.kerasmodel.layers[1].output)
            featureacts = fmodel.predict(DNA).max(axis=(1,2))

            fdf = pd.DataFrame(featureacts)
            fdf['score']= LABELS[:]

            ranking = fdf.corr(method='spearman')['score'].sort_values(ascending=False)
            for i, m in enumerate(ranking.index[1:(self.nmotifs+1)]):
                s = W[:, 0, :, m]
                s -= logsumexp(s, axis=1, keepdims=True)
                s = np.exp(s)
                self.meme.add(s, '{}_{}'.format(process_state, i))
        os.remove(roi)

    def save_motifs(self, output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
        self.meme.save(output)
