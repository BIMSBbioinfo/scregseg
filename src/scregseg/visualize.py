import numpy as np
import pandas as pd
import scanpy as sc
from coolbox.core.track.base import Track
from coolbox.core.track.hist.base import HistBase
from scregseg.bam_utils import cell_scaling_factors
from scregseg.bam_utils import profile_counts
from scregseg.countmatrix import normalize_counts
from scregseg.countmatrix import collapse_cells
from scregseg.countmatrix import merge_samples
import coolbox
from coolbox.api import *
from coolbox.utilities import split_genome_range

class SingleTrack(HistBase):
    def __init__(self, data, **kwargs):
        properties = HistBase.DEFAULT_PROPERTIES.copy()
        properties.update({
            'type': properties['style'],
            "file": 'dummy',
            'style': 'fill',
            **kwargs,
        })

        super().__init__(**properties)
        self.data = data

    def fetch_data(self, gr, **kwargs):
        chrom, start, end = split_genome_range(gr)
        sdata = self.data

        if self.properties['style'] == 'heatmap':
            return sdata.todense()

        data = np.asarray(sdata.mean(0)).flatten()
        return data


class SingleCellTracks:
    def __init__(self, cellannot, files, size_factor='rdepth', tag='RG', mapq=10):
        if not isinstance(files, list):
            files = [files]
        self.size_factor = size_factor
        self.files = files
        self.cellannot = cellannot
        self.tag = tag
        self.mapq = mapq

    def _split_iv(self, iv):
        chr_, res = gr.split(':')
        start,end = res.split('-')
        return chr_, int(start), int(end)

    def plot(self, grange, groupby, frames_before=None, 
             frames_after=None, normalize=True,
             palettes=sc.pl.palettes.vega_20_scanpy,
             add_total=True, style='fill',
             **kwargs):

        adatas = [profile_counts(file,
                                 grange,
                                 selected_barcodes=self.cellannot.index.tolist(),
                                 tag=self.tag, mapq=self.mapq) for file in self.files]

        df = self.cellannot
        for adata in adatas:
            adata.obs.loc[list(set(adata.obs.index).intersection(set(df.index))),df.columns] = df.loc[list(set(adata.obs.index).intersection(set(df.index))), :]
            
            if normalize:
                _ = normalize_counts(adata, self.size_factor)
        adata = merge_samples(adatas)
        adata = adata[adata.obs.index.isin(df.index)]

        frame = XAxis()

        if frames_before:
            frame = frames_before

        if add_total:

            frame += SingleTrack(adata.X,
                                 #max_value=ymax if normalize else 'auto',
                                 title='total',
                                 color='black',
                                 style='fill',
                                 **kwargs)

        if groupby in adata.obs.columns:
            cats = adata.obs[groupby].cat.categories
        else:
            cats = {}
        
        datasets = {}
        ymax = 0.0
        colors = {}

        for i, cat in enumerate(cats):
            sdata = adata[adata.obs[groupby]==cat,:]
            datasets[cat]=sdata.X
            ymax = max(ymax, sdata.X.mean(0).max())
            colors[cat] = palettes[i%len(palettes)]

        for cat in cats:
            frame += SingleTrack(datasets[cat],
                                 max_value=ymax if normalize else 'auto',
                                 title=str(cat),
                                 color=colors[cat],
                                 style=style,
                                 **kwargs)

        if frames_after:
            
            print('len before adding genes', len(frame.tracks))
            frame += frames_after
        print(len(frame.tracks))
        return frame.plot(grange)

