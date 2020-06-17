import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tempfile
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalMaxPooling2D, Dropout
from janggu.data import Bioseq
from janggu.data import Cover
from janggu.data import ReduceDim
from janggu import inputlayer, outputdense
from janggu import Janggu
from janggu import DnaConv2D
import numpy as np
from scipy.special import logsumexp
from keras.callbacks import EarlyStopping
from janggu import Scorer
from sklearn.metrics import r2_score
from keras import Model
#import subprocess
from scregseg import Scregseg

class Meme:
  def __init__(self):
      self.ppms = []
      self.names = []
      pass
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


class MotifExtractor:
    def __init__(self, scmodel, refgenome, ntop=15000, nbottom=15000,
                 ngap=70000, flank=250, nmotifs=10):
       self.ntop = ntop
       self.refgenome = refgenome
       self.nmotifs = nmotifs
       self.nbottom = nbottom
       self.ngap = ngap
       self.flank = flank
       self.scmodel = scmodel
       self.meme = Meme()

    def extract_motifs(self):
        tmpdir = tempfile.mkdtemp()
        filename = 'labels.bedgraph'
        roi = os.path.join(tmpdir, filename)


        binsize = self.scmodel._segments.iloc[0].end - \
                  self.scmodel._segments.iloc[0].start

        for i in range(self.scmodel.n_components):

            process_state= 'state_{}'.format(i)

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

            DNA = Bioseq.create_from_refgenome('dna',
                                               refgenome=self.refgenome,
                                               roi=roi,
                                               binsize=binsize,
                                               flank=self.flank,
                                               cache=False)

            LABELS = ReduceDim(Cover.create_from_bed('score',
                                                     bedfiles=roi,
                                                     roi=roi,
                                                     binsize=binsize,
                                                     cache=False,
                                                     resolution=binsize))

            @inputlayer
            @outputdense('linear')
            def cnn_model(inputs, inp, oup, params):
              with inputs.use('dna') as dna_in:
                  layer = dna_in
              layer = DnaConv2D(Conv2D(100, (13, 1), activation='sigmoid'))(layer)
              layer = GlobalMaxPooling2D()(layer)
              layer = Dropout(.5)(layer)
              return inputs, layer

            # fit the model
            model = Janggu.create(cnn_model, (),
                                  inputs=DNA,
                                  outputs=LABELS,
                                  name='simple_cnn_{}'.format(process_state))

            model.compile(optimizer='adam', loss='mae')

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
                   #            columns=['motif_{}'.format(i) for i in range(featureacts.shape[1])])
            fdf['score']= LABELS[:]

            ranking = fdf.corr(method='spearman')['score'].sort_values(ascending=False)
            print(ranking.head())
            for i, m in enumerate(ranking.index[1:(self.nmotifs+1)]):
                s = W[:, 0, :, m]
                s -= logsumexp(s, axis=1, keepdims=True)
                s = np.exp(s)
                self.meme.add(s, '{}_{}'.format(process_state, i))
        os.remove(roi)

    def save_motifs(self, output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
        self.meme.save(output)
