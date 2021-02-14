import numpy as np
import pandas as pd
import scanpy as sc
from coolbox.core.track.base import Track
from coolbox.core.track.hist.plot import PlotHist
from coolbox.core.track.hist.base import HistBase
from scregseg.bam_utils import cell_scaling_factors
from scregseg.bam_utils import profile_counts
from scregseg.countmatrix import normalize_counts
from scregseg.countmatrix import collapse_cells
from scregseg.countmatrix import merge_samples
import coolbox
from coolbox.api import *
from coolbox.utilities import split_genome_range

class SingleCellTrack(HistBase):

    @classmethod
    def create_pseudobulk(cls, adata, groupby, maxcells=None, palettes=sc.pl.palettes.vega_20_scanpy,
                          **kwargs):
        names = adata.obs[groupby].unique()
        print(names)
        
        colorname = groupby + '_colors'
        if colorname not in adata.uns:
            adata.uns[colorname] = palettes[:len(names)]
        frame = None
        for i, name in enumerate(names):
            sadata = adata[adata.obs[groupby]==name,:]
            if maxcells is not None:
                sadata = sadata[:min(maxcells, sadata.shape[0]),:]
            barcodes = sadata.obs.index

            f = cls(adata, barcodes, title=f'{groupby}_{name}',
                    color=adata.uns[colorname][i], **kwargs)
            if frame is None:
                frame = f
            else:
                frame += f
            frame += HLine()
        return frame


    def __init__(self, adata, barcodes, **kwargs):
        properties = HistBase.DEFAULT_PROPERTIES.copy()
        properties.update({
            'type': properties['style'],
            "file": 'dummy',
            **kwargs,
        })

        super().__init__(**properties)
        self.adata = adata[barcodes,:]

    def fetch_data(self, gr, **kwargs):
        chrom, start, end = split_genome_range(gr)
        adata = self.adata
        sdata = adata[:, (adata.var.chrom==chrom) & (adata.var.start>= start) &
                      (adata.var.end<=end)]
        if self.properties['style'] == 'heatmap':
            return np.asarray(sdata.X.todense())
        data = np.asarray(sdata.X.sum(1)).flatten()
        #data *= 1000/data.sum()
        return data

class SingleCellBAMPlotter:
    def __init__(self, bamfiles, labels, celltable, tag='RG'):
        if not isinstance(bamfiles, list):
            bamfiles = [bamfiles]
        if not isinstance(labels, list):
            labels = [labels]
        self.factors = [cell_scaling_factors(bamfile, tag='RG') for bamfile in bamfiles]
        self.bamfiles = bamfiles
        if not isinstance(celltable, pd.DataFrame):
            self.celltable = pd.read_csv(celltable)
            self.celltable.set_index('barcode', inplace=True)
        self.celltable = celltable
        self.tag = tag
        self.labels = labels

    def _split_iv(self, iv):
        chr_, res = gr.split(':')
        start,end = res.split('-')
        return chr_, int(start), int(end)

    def plot(self, grange, groupby, frames_before=None, 
             frames_after=None, normalize=True,
             **kwargs):
        adatas = [profile_counts(bamfile, grange, tag=self.tag) for bamfile in self.bamfiles]
        df = self.celltable
        for label, adata, factor in zip(self.labels, adatas, self.factors):
            adata.obs.loc[:,"sample"]=label
            adata.obs.loc[list(set(adata.obs.index).intersection(set(df.index))),df.columns] = df.loc[list(set(adata.obs.index).intersection(set(df.index))), :]
            
            if normalize:
                _ = normalize_counts(adata, factor)
        adata = merge_samples(adatas)
        adata = adata[adata.obs.index.isin(df.index)]
        print(adata.X.sum())
        frame = XAxis()
        if frames_before:
            frame = frames_before

        frame += SingleCellTrack.create_pseudobulk(adata, groupby, **kwargs)

        if frames_after:
            frame += frames_after
        return frame.plot(grange)
