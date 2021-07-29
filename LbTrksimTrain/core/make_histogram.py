import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from LbTrksimTrain.core import Dataset

light_colors = [
    'red',
    '#44aaaa',
    'magenta',
    'orange',
]

dark_colors = [
    'black',
    'blue',
]


def make_histogram(cfg, cfg_dataset, max_chunks=1, entrysteps=1000000, selections=None, weight_dict=None, errorbars=None):
    hists = {}

    if weight_dict is None:
        weight_dict = {cfg_dataset.title: None}
    if not isinstance(weight_dict, dict):
        weight_dict = {cfg_dataset.title: weight_dict}

    if errorbars is None:
        errorbars = []

    progress_bar = tqdm(enumerate(Dataset.iterate(cfg_dataset,
                                                  max_chunks=max_chunks,
                                                  entrysteps=entrysteps
                                                  )),
                        total=max_chunks,
                        ascii=True,
                        )

    if selections is None:
        selections = []

    if isinstance(selections, str):
        selections = [selections]

    for iChunk, chunk in progress_bar:
        if iChunk > max_chunks:
            break

        for selcode, sel in selections:
            if sel not in hists.keys():
              hists [sel] = dict() 

            db = chunk.query(selcode) if sel is not None else chunk

            for hName, hCfg in cfg.items():
                if hName not in hists[sel].keys():
                    hists[sel][hName] = dict()

                sdb = db.query(hCfg.selection) if 'selection' in hCfg.keys() else db

                var = sdb.eval(hCfg.variable) if 'variable' in hCfg.keys() else hName
                binning = hCfg.binning if 'binning' in hCfg.keys() else [100, var.min(), var.max()]
                for wName, weights in weight_dict.items():
                    w = weights(sdb) if weights is not None and not isinstance(
                        weights, str) else weights
                    if isinstance(w, str):
                        w = sdb.eval(w).astype(np.float64)

                    if wName not in hists[sel][hName].keys():
                        hists[sel][hName][wName] = [
                            np.zeros(binning[0]),
                            np.linspace(binning[1], binning[2], binning[0]+1),
                        ]
                    hists[sel][hName][wName][0] += np.histogram(var,
                                                           bins=hists[sel][hName][wName][1],
                                                           weights=w,
                                                           )[0]

    from itertools import cycle
    for _, sel in selections:
      print ("### New selection") 
      for hName in sorted(list(hists[sel].keys())):
        print ("### New variable", hName) 
        dcolor = iter(cycle(dark_colors))
        lcolor = iter(cycle(light_colors))
        for wName in hists[sel][hName].keys():
            print ("### New weight", wName) 
            options = cfg[hName].options if 'options' in cfg[hName].keys() else []
            if wName not in errorbars:
                plt.hist(
                    0.5 * (hists[sel][hName][wName][1][1:] +
                           hists[sel][hName][wName][1][:-1]),
                    weights=hists[sel][hName][wName][0],
                    bins=hists[sel][hName][wName][1],
                    label=wName,
                    histtype='step',
                    linewidth=3,
                    density='density' in options,
                    color=next(lcolor),
                )
            else:
                bwidth = hists[sel][hName][wName][1][1] - hists[sel][hName][wName][1][0]
                div = np.sum(hists[sel][hName][wName][0]) * \
                    bwidth if 'density' in options else 1.
                plt.errorbar(
                    0.5 * (hists[sel][hName][wName][1][1:] +
                           hists[sel][hName][wName][1][:-1]),
                    hists[sel][hName][wName][0]/div,
                    xerr=bwidth/2.,
                    yerr=np.sqrt(np.maximum(hists[sel][hName][wName][0], 1))/div,
                    label=wName,
                    fmt='o',
                    markersize=2.5,
                    color=next(dcolor),
                )

            if 'notitle' not in options:
                plt.title(sel)
            if 'nolegend' not in options:
                plt.legend()

            plt.xscale("log" if "logx" in options else "linear")
            plt.yscale("log" if "logy" in options else "linear")

            plt.xlabel(
                cfg[hName].xtitle if 'xtitle' in cfg[hName].keys() else hName)
            plt.ylabel(
                cfg[hName].ytitle if 'ytitle' in cfg[hName].keys() else "Entries")

        print ('yield gca')
        yield plt.gca()


if __name__ == '__main__':
    from LbTrksimTrain import Configuration, CachedDataset

    cfg = Configuration([
        "/pclhcb06/landerli/LamarrTrainingScripts/Tracking/options/tracking.yaml",
        "/pclhcb06/landerli/LamarrTrainingScripts/Tracking/options/2016-MagDown.yaml"
    ])
    #generated = CachedDataset(cfg.datasets['BrunelGenerated'],  max_chunks = 10, entrysteps = 100000,)
    for a in make_histogram(cfg.acceptanceBDT.validationHistograms, cfg.datasets['BrunelGenerated']):
        print(a)
