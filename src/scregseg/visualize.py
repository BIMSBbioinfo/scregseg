import copy
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import scanpy as sc
from coolbox.core.track.base import Track
from coolbox.core.track.bed import BED
from coolbox.core.track.hist.base import HistBase
from scregseg.bam_utils import cell_scaling_factors
from scregseg.bam_utils import profile_counts
from scregseg.countmatrix import normalize_counts
from scregseg.countmatrix import collapse_cells
from scregseg.countmatrix import merge_samples
import coolbox
from coolbox.api import *
from coolbox.utilities import split_genome_range
from svgutils.compose import Figure, SVG

class ScregsegTrack(BED):
    def __init__(self, scregsegobj, **kwargs):
        statecalls = scregsegobj.tobedtool().TEMPFILES[-1]
        super().__init__(statecalls, display='collapsed',
                         labels=False, bed_type='bed9',
                         **kwargs)

    
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
        if size_factor not in cellannot.columns:
            warnings.warn('Size-factor not known. Assuming constant size factor',
                          category=UserWarning)
            self.cellannot.loc[:,"rdepth"] = 1.
        self.tag = tag
        self.mapq = mapq

    def _split_iv(self, iv):
        chr_, res = gr.split(':')
        start,end = res.split('-')
        return chr_, int(start), int(end)

    def plot(self, grange, groupby, frames_before=None, 
             frames_after=None, normalize=True,
             palettes=sc.pl.palettes.vega_20_scanpy,
             add_total=True, style='fill', binsize=50,
             binarize=False, add_labels=True,
             **kwargs):

        df = self.cellannot
        adatas = [profile_counts(file,
                                 grange,
                                 selected_barcodes=df.index.tolist(),
                                 binsize=binsize,
                                 tag=self.tag, mapq=self.mapq) for file in self.files]

        df = self.cellannot
        for adata in adatas:
            if binarize:
                adata.X[adata.X>1]=1
            overlapping = list(set(adata.obs.index).intersection(set(df.index)))
            adata.obs[df.columns] = df.loc[overlapping, :]
            
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
                                 title='total' if add_labels else '',
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
                                 title=str(cat) if add_labels else '',
                                 color=colors[cat],
                                 style=style,
                                 **kwargs)

        if frames_after:
            frame += frames_after
        return frame.plot(grange)


def plot_fragmentsize(adata, ax=None, **kwargs):
    """ Plots the fragment size distribution.

    Parameters
    ----------
    adata : AnnData
        AnnData with associated fragment lengths per region.
        The number of regions in adata must be the same as in self._segments.
        Fragment lengths can be obtained using the --with-fraglens option
        when using scregseg bam_to_counts or scregseg fragment_to_counts.
    basis : str
        Key at which the fragment size is stored. Default: "frag_lens"
    ax : matplotlib.axes.Axes or None
        matplotlib.axes.Axes object 

    Returns
    -------
    matplotlib.axes.Axes
    """
    basis='frag_lens'
    if basis not in adata.obsm:
        raise ValueError(f'{basis} not in adata')
    if ax is None:
        fig, ax =  plt.subplots()
    
    fragsizes = adata.obsm[basis]
    fragsizes = np.asarray(fragsizes.sum(0)).flatten()
    fragsizes /= fragsizes.sum()
    ax.plot(np.arange(fragsizes.shape[0]), fragsizes)
    ax.set_xlabel("Fragment length")
    ax.set_ylabel("Frequency")
    return ax

def _extend(gr, window):
    chrom, range_ = gr.split(':')
    start, end = range_.split('-')
    return f'{chrom}:{max(0,int(start)-window)}-{int(end)+window}'

def plot_locus(grange, groupby,
               cellannot, files, 
               size_factor='rdepth', tag='RG', mapq=10,
               frames_before=None,
               frames_after=None, normalize=True,
               palettes=sc.pl.palettes.vega_20_scanpy,
               add_total=True, style='fill', binsize=50,
               binarize=False, frame_width=40, add_labels=True,
               width_overlap=18.,
               extend_window=0,
               save=None,
               **kwargs):


    if isinstance(grange, dict):
        names = [k for k in grange]
        ranges = [grange[k] for k in grange]
    
    elif not isinstance(grange, list):
        ranges = [grange]
        names = ['']
   
    elif isinstance(grange, list):
        names = ['']*len(grange)
        ranges = granges
    if frames_before is None:
        frames_before = Frame()

    if extend_window > 0:
       ranges = [_extend(gr, extend_window) for gr in ranges]

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (name, gr) in enumerate(zip(names,ranges)):
            sct = SingleCellTracks(cellannot, files, size_factor=size_factor, tag=tag, mapq=mapq)

            frame = copy.deepcopy(frames_before)
            frame.properties['title'] = name

            fig = sct.plot(gr, groupby, frames_before=frame,
                           frames_after=frames_after, normalize=normalize,
                           palettes=palettes,
                           add_total=add_total, style=style, binsize=binsize, 
                           add_labels=add_labels if i==(len(names)-1) else False,
                           binarize=binarize, **kwargs)
            fig.savefig(os.path.join(tmpdir, f'{name}_{gr}.svg'))


        panel = SVG(os.path.join(tmpdir, f'{name}_{gr}.svg'))
        width, height= panel.width, panel.height

        composite_figure = Figure(f"{(width-width_overlap)*len(names)+width_overlap}pt",f"{height}pt",
                                  *[SVG(os.path.join(tmpdir, f'{name}_{gr}.svg')).move((width-width_overlap)*i,0) for i, (name,gr) in enumerate(zip(names,ranges))])
        if save is not None:
            composite_figure.save(save)
            os.makedirs(save.split('.')[0], exist_ok=True)
            for name, gr in zip(names,ranges):
                shutil.copy(os.path.join(tmpdir, f'{name}_{gr}.svg'), save.split('.')[0])

    return composite_figure

