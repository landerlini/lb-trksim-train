from pprint import pprint
import argparse
from logging import getLogger as logger
from datetime import datetime
import os.path
from functools import partial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from scipy.optimize import curve_fit

from LbTrksimTrain.core import Configuration
from LbTrksimTrain.core import Dataset
from LbTrksimTrain.core import GanModel
from LbTrksimTrain.core import gan_utils 
from LbTrksimTrain.core import Report

from sklearn.preprocessing import QuantileTransformer , MinMaxScaler, StandardScaler
from LbTrksimTrain.core.GanModel import DecorrTransformer

def train(cfg, report, modeldir):
    tf.config.threading.set_inter_op_parallelism_threads(cfg.threads)
    tf.config.threading.set_intra_op_parallelism_threads(cfg.threads)

    train_one_type(cfg, report, modeldir, 5)
    train_one_type(cfg, report, modeldir, 3)
    train_one_type(cfg, report, modeldir, 4)


def train_one_type(cfg, report, modeldir, tracktype=3):
    tracktypename = {3: 'Long', 4: 'Upstream', 5: 'Downstream'}[tracktype]
    report.add_markdown("# %s tracks" % tracktypename)

    config = cfg.covarianceGAN[tracktypename.lower()]
    if not config.do_training:
      report.add_markdown("DISABLED")
      return 


    ## Resolve hopaas
    config.resolve_hopaas()

    ################################################################################
    # Access to data
    recoed = Dataset.iterate(
        cfg.datasets['BrunelRecoed'],
        max_chunks=1000000000,
        entrysteps=config.chunksize
    )




    ################################################################################
    # Train Loop
    from itertools import repeat, chain
    dlr = float(config.discriminator_learning_rate)
    glr = float(config.generator_learning_rate)
    #if tracktype == 3:

    myGan = GanModel.GanModel(
          n_iterations=config.nIterations,
          batchsize=config.batchsize,
          n_noise_inputs=config.nRandomNodes,
          n_generator_layers=config.n_generator_layers,
          n_generator_nodes=config.n_generator_nodes,
          n_discriminator_layers=config.n_discriminator_layers,
          n_discriminator_nodes=config.n_discriminator_nodes,
          generator_optimizer=tf.keras.optimizers.RMSprop(glr if tracktype==3 else glr/5),
          discriminator_optimizer=tf.keras.optimizers.RMSprop(dlr if tracktype==3 else dlr/5),
          generator_learning_rate=glr,
          discriminator_learning_rate=dlr,
          wreferee=config.wreferee, 
          generator_lazyness=config.generator_lazyness, 
          lambda_=config.generator_lambda,
          gamma_=config.generator_gamma,
          preprocessing = [
            ('minmax', MinMaxScaler ),
            ('quantile1', partial (QuantileTransformer, output_distribution = 'normal') ),
            ('decorrelate', DecorrTransformer ),
            #('standardscaler', StandardScaler ),
            ('quantile2', partial (QuantileTransformer, output_distribution = 'normal') ),
            ],
      )
    #else:
    #if tracktype != 3:
      #myGan = GanModel.GanModel.load(os.path.join(modeldir, "long"))
#      myGan.generator_optimizer_ =
#      myGan.discriminator_optimizer_ = 
#      myGan.batchsize_=config.batchsize
#      myGan.n_noise_inputs_=config.nRandomNodes
#      myGan.n_generator_layers_=config.n_generator_layers
#      myGan.n_generator_nodes_=config.n_generator_nodes
#      myGan.n_discriminator_layers_=config.n_discriminator_layers
#      myGan.n_discriminator_nodes_=config.n_discriminator_nodes
#      del myGan.transformerX_
#      del myGan.transformerY_

    largedset = Dataset.get(cfg.datasets['BrunelRecoed'], "type==%d" % tracktype,
                            #max_files=1
                            max_files={3: 5, 4: 50, 5: 1000}[tracktype]
                            )

    X, Y = gan_utils.getData(config, largedset, tracktype)
    myGan._apply_preprocessing(X, Y)
    del largedset

    ################################################################################
    # Plotting
    pruned = False
    for losses, bdtl in gan_utils.train(config, myGan, recoed, tracktype):
        scaled_loss = ((losses - losses.min()) / (losses.max() - losses.min()) *
                       (bdtl[:, 1:].max() - bdtl[:, 1:].min()) + bdtl[:, 1:].min())

        nIterations = config.nIterations
        plt.plot(np.arange(len(losses))/nIterations,
                 scaled_loss, 'k-', label="GAN loss")
        plt.plot(bdtl[:, 0]/nIterations, bdtl[:, 1],
                 'go--', label="BDT monitor (train)")
        plt.plot(bdtl[:, 0]/nIterations, bdtl[:, 2],
                 'ro-', label="BDT monitor (test)")
        plt.xlabel("Data chunk")
        plt.ylabel("Loss")
        plt.legend()
        report_dir, report_name = os.path.split(report.filename)
        exported_file_name = os.path.join(report_dir, "data", report_name.replace(".html", "-%s-%s.npz"%(tracktypename, config.hopaas_trial)))
        np.savez(exported_file_name, loss=losses, bdt_iter=bdtl[:,0], bdt_train=bdtl[:,1], bdt_test=bdtl[:,2])
        with Report(report.title, report.filename.replace(".html", "-online-%s.html" % tracktypename)) as report_:
            report_ . add_figure()
            report_ . add_markdown(f"[Download dataset](data/{os.path.basename(exported_file_name)})")
            plt.clf()

        if config.should_prune(loss=bdtl[-1,2], step=bdtl[-1,0]/nIterations):
          pruned = True
          break 

    if not pruned:
      config.finalize_trial(loss=bdtl[-10:,2].mean())

    myGan.save(os.path.join (modeldir, tracktypename.lower(), config.hopaas_trial))
    report.add_markdown("Plots")
    recoed = Dataset.iterate(
        cfg.datasets['BrunelRecoed'], max_chunks=1, entrysteps=1000)
    db = next(iter(recoed))
    X, Y = gan_utils.getData(config, db, tracktype)
    Yhat = myGan.predict(X)

    for x, varName in zip(X.T, config.discrVars):
        plt.hist(x, bins=100, label="Training", density=True)
        plt.legend()
        plt.yscale('log')
        plt.xlabel(varName)
        plt.ylabel('Normalized entries')
        report.add_figure(options='width = 19%')
        plt.clf()

    for y, yhat, varName in zip(Y.T, Yhat.T, config.targetVars):
        plt.hist(y, bins=100, label="Training", density=True)
        plt.hist(yhat, bins=100, histtype='step', linewidth=3,
                 label="Generated", density=True, color='red')
        plt.legend()
        plt.yscale('log')
        plt.xlabel(varName)
        plt.ylabel('Normalized entries')
        report.add_figure(options='width = 19%')
        plt.clf()

    XY = np.concatenate([X, Y], axis=1)[:1000]
    XYhat = np.concatenate([X, Yhat], axis=1)[:1000]
    fXY = np.concatenate([myGan.transformerX_.transform(
        X), myGan.transformerY_.transform(Y)], axis=1)[:1000]
    fXYhat = np.concatenate([myGan.transformerX_.transform(
        X), myGan.transformerY_.transform(Yhat)], axis=1)[:1000]

    for iVar, xName in enumerate(tqdm(config.discrVars + config.targetVars, ascii=True, desc="Plotting")):
        for jVar, yName in enumerate(config.discrVars + config.targetVars):
            if jVar <= iVar:
                continue
            plt.plot(XY[:, iVar], XY[:, jVar], 'b.')
            if iVar >= X.shape[1] or jVar >= X.shape[1]:
                plt.plot(XYhat[:, iVar], XYhat[:, jVar], 'rx', alpha=0.3)
            plt.xlabel(xName)
            plt.ylabel(yName)
            report.add_figure(options='width = 24%')
            plt.clf()

            plt.plot(fXY[:, iVar], fXY[:, jVar], 'b.')
            if iVar >= X.shape[1] or jVar >= X.shape[1]:
                plt.plot(fXYhat[:, iVar], fXYhat[:, jVar], 'rx', alpha=0.3)

            plt.xlabel("Preprocessed x_{%d}" % iVar)
            plt.ylabel("Preprocessed x_{%d}" % jVar)
            report.add_figure(options='width = 24%')
            plt.clf()



################################################################################
# Validation
def validate(cfg, report, modeldir):
    tf.config.threading.set_inter_op_parallelism_threads(cfg.threads)
    tf.config.threading.set_intra_op_parallelism_threads(cfg.threads)

    for tracktype, tracktypename in [(3, 'Long'), (4, 'Upstream'), (5, 'Downstream')]:
        report.add_markdown("# Validation for %s tracks" % tracktypename)
        if not cfg.covarianceGAN[tracktypename.lower()].do_validation:
          report.add_markdown("DISABLED")
          continue 

        gan = GanModel.GanModel.load(os.path.join(modeldir, tracktypename.lower()))
        recoed = Dataset.get(cfg.datasets['BrunelRecoed'], "type==%d" % tracktype,
                             #max_files=1,
                             max_files = {3: 5, 4:50, 5:1000}[tracktype]
                             )

        print(tracktypename, "# entries:", len(recoed))

        pz_bins = np.array([0.1, 1, 2, 3, 5, 10, 15, 20, 30, 50, 70, 150])
        eta_bins = np.array([1.8,2.2,3.0,3.3,3.7,4.0,4.3,5.0])

        ksDistance = np.zeros ((len(pz_bins)-1, len(eta_bins)-1)) 

        for iVar in range(9): 
          for iPbin, pbin in enumerate(zip(pz_bins[:-1], pz_bins[1:])):
            for iEtaBin, etabin in enumerate(zip(eta_bins[:-1], eta_bins[1:])):
                db = recoed.query("(pz/1e3 > %f) and (pz/1e3 < %f)" % pbin).copy()
                db.query("(eta_ClosestToBeam > %f) and (eta_ClosestToBeam < %f)" % etabin, inplace=True) 
                try:
                    X, Y = gan_utils.getData(cfg.covarianceGAN[tracktypename.lower()], db, tracktype)
                except IndexError:
                    report.add_markdown(
                        "Empty bin for pz in (%.0f, %.0f) MeV/c" % pbin)
                    continue

                if len(X) < 100: continue 

                Ygen = gan.predict(X)
                bnds = None  # np.linspace(-0.5,0.5,101)
                fig = plt.figure(figsize=[5, 3])
                rCumulative = None
                for label, yVar in [("Reconstructed", Y[:, iVar]), ("Generated", Ygen[:, iVar])]:
                    #yVar = yVar [ (yVar > np.quantile(yVar,0.005)) & (yVar < np.quantile(yVar, 0.99)) ]
                    hist_fine, bnds = np.histogram(yVar, bins=bnds if bnds is not None else 800)
                    hist = np.sum ( np.reshape (hist_fine, (-1, 10)), axis = 1 )

                    binw = bnds[10] - bnds[0]
                    herr = np.sqrt(hist)
    #                hmax = np.max(hist)
    #                hist = hist/hmax
    #                herr = herr/hmax

                    xAxis = 0.5*(bnds[1:]+bnds[:-1])

                    if label == 'Reconstructed':
                      rCumulative = np.cumsum (hist_fine)
                      rCumulative = rCumulative/rCumulative[-1]
                    elif label == 'Generated':
                      gCumulative = np.cumsum (hist_fine)
                      gCumulative = gCumulative/gCumulative[-1]
                      ksDistance[iPbin,iEtaBin] = np.max ( np.abs ( rCumulative-gCumulative) ) 
                    else: 
                      raise KeyError (label) 

                    plt.gca().set_ylim([.2, 2. * np.max(hist)])
                    if label == 'Reconstructed': 
                      plt.errorbar(0.5*(bnds[1::10]+bnds[:-1:10]), hist, xerr=0.5*(
                        bnds[1::10]-bnds[:-1:10]), yerr=herr, fmt='o', markersize=0, label=label, linewidth=5, color = "#00ffff")
                    elif label == 'Generated':
                      plt.errorbar(0.5*(bnds[1::10]+bnds[:-1:10]), hist, xerr=0.5*(
                        bnds[1::10]-bnds[:-1:10]), yerr=herr, fmt='o', markersize=4, label=label, color = '#cc00cc')
                    else:
                      raise KeyError (label) 
                plt.title("$p_z$ in (%.1f, %.1f) GeV ; $\eta$ in (%.1f, %.1f)" % tuple(pbin + etabin))
                plt.yscale('log')
                plt.xlabel(cfg.covarianceGAN[tracktypename.lower()].targetVars[iVar])
                plt.legend(title = "KS dist.: %.1f %%" % ((ksDistance[iPbin,iEtaBin])*100.) )
                try: 
                  plt.tight_layout()
                  report.add_figure(options="width=24%")
                except ValueError: 
                  plt.yscale('linear')
                  plt.tight_layout()
                  report.add_figure(options="width=24%")
                plt.close()


          report.add_markdown("### Kolmogorov distance")
          for iBin, kd in enumerate(ksDistance.T): 
            plt.errorbar (0.5*(pz_bins[1:]+pz_bins[:-1]), kd*100., xerr = 0.5*(pz_bins[1:]-pz_bins[:-1]), fmt = 'o', label = "$\eta$ in (%.1f, %.1f)" % (eta_bins[iBin], eta_bins[iBin+1]), markersize = 3) 

          plt.legend(title = "%s tracks" % tracktypename ) 
          plt.xlabel ( "Longitudinal momentum [MeV/c]")
          plt.xscale ( 'log' ) 
          plt.ylabel ( "KS distance [%]" ) 
          plt.title ( "%s projection" % cfg.covarianceGAN[tracktypename.lower()].targetVars[iVar] ) 

          report.add_figure(options="width=60%")
          plt.close()




