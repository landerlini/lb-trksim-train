from logging import getLogger as logger
import os.path
import pickle 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from LbTrksimTrain.core import Dataset 
from LbTrksimTrain.core import CachedDataset 
from LbTrksimTrain.core.Report import Jscript  

def plot ( cfg, report ): 
  numerator = CachedDataset(cfg.datasets['BrunelRecoed'],      max_chunks = 100, entrysteps = 1000000, max_files = 1, files_key='files_train')
  denominator = CachedDataset(cfg.datasets['BrunelGenerated'], max_chunks = 100, entrysteps = 1000000, max_files = 1, files_key='files_train') 

  for state, stcfg in cfg.states.items():
    nh = np.zeros ( [stcfg.binning[0]]*2, dtype = np.float64 ) 
    dh = np.zeros ( [stcfg.binning[0]]*2, dtype = np.float64 ) 
    binning = [np.linspace(stcfg.binning[1], stcfg.binning[2], stcfg.binning[0]+1)]*2 
    for db in numerator.iterate():
      nh += np.histogram2d ( db['x_%s'%state ], db['y_%s'%state], bins = binning)[0]
    for db in denominator.iterate():
      dh += np.histogram2d ( db['x_%s'%state ], db['y_%s'%state], bins = binning)[0]

    ratio = np.where (dh > 0, nh/dh, np.min(nh/(dh+1)))
    plt.imshow (ratio.T+1e-2, extent = [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]], origin = 'lower') 
    plt.title ( "%s" % state ) 
    plt.xlabel ( "x coordinate [mm]" ) 
    plt.ylabel ( "y coordinate [mm]" ) 
    report.add_figure(options = 'width = 49%') ; plt.clf() 

  ####

    binning = [np.linspace(0,150, 151), np.linspace ( 1.5, 6, 151 )] 
    nh = np.zeros ( [150,150], dtype = np.float64 ) 
    dh = np.zeros ( [150,150], dtype = np.float64 ) 
    for db in numerator.iterate():
      nh += np.histogram2d ( 1e-3*db['p_%s'%state ], db['eta_%s'%state], bins = binning)[0]
    for db in denominator.iterate():
      dh += np.histogram2d ( 1e-3*db['p_%s'%state ], db['eta_%s'%state], bins = binning)[0]

    ratio = np.where (dh > 0, nh/dh, np.min(nh/(dh+1)))
    plt.imshow (ratio.T+1e-2, extent = [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]], origin = 'lower', aspect = 'auto') 
    plt.title ( "%s" % state ) 
    plt.xlabel ( "Momentum [GeV/c]" ) 
    plt.ylabel ( "Pseudorapidity" ) 
    report.add_figure(options = 'width = 49%') ; plt.clf() 





def train ( cfg, report, modelfile):
  from sklearn.ensemble import GradientBoostingClassifier
  from sklearn.utils import shuffle 
  from tqdm import trange
  plt.inferno()

  report.add_markdown ( "### Training efficiency BDT")
#  recoed = CachedDataset(cfg.datasets['BrunelRecoed'],      max_chunks = 10, entrysteps = 1000000, max_files = 1)
#  generated = CachedDataset(cfg.datasets['BrunelGenerated'], max_chunks = 10, entrysteps = 1000000,max_files = 1) 

  effBdt = cfg.efficiencyBDT
  recoed    = Dataset.get(cfg.datasets['BrunelRecoed'], "(type==3) or (type==4) or (type==5)",  max_files = effBdt.numFiles, files_key='files_train') 
  generated = Dataset.get(cfg.datasets['BrunelGenerated'], 'acceptance==1', max_files = effBdt.numFiles, files_key='files_train') 

  

#  dbs = [] 
#  for reconstructed, dataset in [ (1, recoed) , (0, generated ) ]: 
#    for db in dataset.iterate():
#      _type = db['type'] 
#      if not reconstructed: db.query ( "acceptance == 1 and reconstructed == 0" , inplace = True ) 
#      db.drop(columns = [c for c in db.columns if c not in effBdt.discrVars], inplace=True)
#      db['label'] =  _type if reconstructed else 0 
#      dbs.append (db) 
#
#  full_db = shuffle ( pd.concat ( dbs, ignore_index = True ).reset_index(drop=True))

  recoed['label']    = recoed['type']
  generated['label'] = 0
  unreco_cut = " or " .join ([
    "(reconstructed == 0)",
    "( (type!=3) and (type!=4) and (type!=5) )",
    ])


  print (effBdt.discrVars)
  print ( generated.query(unreco_cut) [ effBdt.discrVars ].columns )
  print ( recoed                      [ effBdt.discrVars ].columns ) 
  print ( recoed                                          .columns ) 

  full_db = shuffle ( 
      pd.concat ([
        generated.query(unreco_cut) [ effBdt.discrVars + ['label'] ],
        recoed                      [ effBdt.discrVars + ['label'] ]
        ], 
        ignore_index = True 
      )
  )
  full_db.reset_index (inplace=True, drop=True) 
  training, test = full_db[:len(full_db)//2], full_db[len(full_db)//2:] 

  logger("effTrain").info ( "Training sample: %d entries" % len (training))
  #np.savez ( "efficiencyBdtDataset.npz", training = training[effBdt.discrVars], test = test[effBdt.discrVars] ) 
    
  classifier = GradientBoostingClassifier(n_estimators = 1, warm_start = True, 
        subsample = cfg.efficiencyBDT.subsample, 
        learning_rate = cfg.efficiencyBDT.learning_rate,
        max_depth = cfg.efficiencyBDT.max_depth, 
      )
  testscore = []
  trainscore = [] 
  logger('effTrain').info ("Training started") 
  for i in trange (cfg.efficiencyBDT.nEpochs, ncols = 80, ascii = True):
    classifier.n_estimators += 1
    classifier.fit ( training[effBdt.discrVars].values, training['label'].values) #, np.where(training['label'].values > 0, 0.01, 1) ) 
    #trainscore.append( classifier.train_score_[-1] )
    strain = training.sample(1000) 
    stest  = test.sample(1000) 
    trainscore .append( classifier.score ( strain[effBdt.discrVars], strain['label']) )
    testscore  .append( classifier.score ( stest[effBdt.discrVars], stest['label']) )

  with open  (modelfile, 'wb') as fout:
    pickle.dump ( classifier, fout )

  print (full_db.describe()) 
  plt.plot ( trainscore, label = "Train" ) 
  plt.plot ( testscore, label="Validation" ) 
  plt.xlabel ( "Epochs" ) 
  plt.ylabel ( "Score" ) 
  plt.legend() 
  report.add_figure(); plt.clf() 

#  recoed = CachedDataset(cfg.datasets['BrunelRecoed'],      max_chunks = 10, entrysteps = 1000000, max_files = 1)
#  generated = CachedDataset(cfg.datasets['BrunelGenerated'], max_chunks = 10, entrysteps = 1000000,max_files = 1) 

  var = cfg.variables.logpz
  binning = np.linspace ( var.binning[1], var.binning[2], var.binning[0]+1 )
  nh = np.zeros ( var.binning[0] )   
  dh = np.zeros ( var.binning[0] )   
  wdh = np.zeros ( var.binning[0] )   
  #for db in recoed.iterate():
  nh += np.histogram ( recoed.query('type == 3').eval(var.formula), bins = binning )[0]

#  for db in generated.iterate():
  db = generated.query ( 'acceptance == 1')
  w = classifier.predict_proba ( db[cfg.efficiencyBDT.discrVars].values )[:,classifier.classes_.tolist().index(3)]
  dh += np.histogram ( db.eval(var.formula), bins = binning, weights = np.ones_like(w) )[0]
  wdh += np.histogram ( db.eval(var.formula), bins = binning, weights = w )[0]

  plt.hist ( 0.5*(binning[1:]+binning[:-1]), weights = nh, bins = binning, label = 'Selected', density = True )
  plt.hist ( 0.5*(binning[1:]+binning[:-1]), weights = dh, bins = binning, label = 'Unweighted', 
      histtype = 'step', linewidth = 2, linestyle = '--', density = True )
  plt.hist ( 0.5*(binning[1:]+binning[:-1]), weights = wdh, bins = binning, label = 'Weighted', 
      histtype = 'step', linewidth = 2 , density = True, color = 'red')
  plt.legend()
  plt.xlabel ( var.title ) 
  plt.ylabel ( "(Weighted) entries" ) 
  report.add_figure(); plt.clf() 


def validate (cfg, report, modelfile): 
  effBdt = cfg.efficiencyBDT 
  logger("effValidate").info ("effValidate is running") 

  ## Load the trained model 
  with open  (modelfile, 'rb') as fin:
    classifier = pickle.load ( fin )

  report.add_markdown ( "### Training efficiency BDT")
  recoed = CachedDataset(cfg.datasets['BrunelRecoed'],      max_files = 10, files_key='files_validate')
  generated = CachedDataset(cfg.datasets['BrunelGenerated'], max_files = 10, files_key='files_validate') 

  for state, stcfg in cfg.states.items():
    logger("effValidate").info ( "Processing state %s" % state ) 
    ########################## RECONSTRUCTED xy ###########################################
    logger("effValidate").info ("Plotting reconstructed xy") 
    nh = np.zeros ( [stcfg.binning[0]]*2, dtype = np.float64 ) 
    dh = np.zeros ( [stcfg.binning[0]]*2, dtype = np.float64 ) 
    binning = [np.linspace(stcfg.binning[1], stcfg.binning[2], stcfg.binning[0]+1)]*2 
    for db in generated.iterate():
      logger("effValidate").info ( "Loaded %d entries from generated" % len(db))
      db.query ( "acceptance == 1", inplace = True)
      dh += np.histogram2d ( db['x_%s'%state ], db['y_%s'%state], bins = binning)[0]
    for db in recoed.iterate():
      logger("effValidate").info ( "Loaded %d entries from recoed" % len(db))
      sdb = db.query ( "type == 3")
      nh += np.histogram2d ( sdb['x_%s'%state ], sdb['y_%s'%state], bins = binning)[0]

    logger("effValidate").info ("Loop stopped") 

    ratio = np.where (dh > 0, nh/dh, np.min(nh/(dh+1)))
    plt.imshow (ratio.T, extent = [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]], origin = 'lower') 
    plt.title ( "%s (Brunel reconstruction as Long track)" % state ) 
    plt.xlabel ( "x coordinate [mm]" ) 
    plt.ylabel ( "y coordinate [mm]" ) 
    js = Jscript().hist2d(f"{state}_xy_brunel", ratio.T, [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]])
    report.add_figure(options=str(js.width("49%"))) ; plt.clf() ; plt.close() 


    ########################## WEIGHTED xy ###########################################
    logger("effValidate").info ("Plotting weighted xy") 
    nh = np.zeros ( [stcfg.binning[0]]*2, dtype = np.float64 ) 
    dh = np.zeros ( [stcfg.binning[0]]*2, dtype = np.float64 ) 

    binning = [np.linspace(stcfg.binning[1], stcfg.binning[2], stcfg.binning[0]+1)]*2 
    generated.clear_cache() 
    for db in generated.iterate():
      db.query ( "acceptance == 1", inplace = True)
      dh += np.histogram2d ( db['x_%s'%state ], db['y_%s'%state], bins = binning)[0]
      w = classifier.predict_proba( db[effBdt.discrVars].values )[:,list(classifier.classes_).index(3)]
      nh += np.histogram2d ( db['x_%s'%state ], db['y_%s'%state], bins = binning, weights=w)[0]

    ratio = np.where (dh > 0, nh/dh, np.min(nh/(dh+1)))
    #ratio = np.clip(ratio, 0, 1)
    plt.imshow (ratio.T, extent = [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]], origin = 'lower') 
    plt.title ( "%s (GBDT Model)" % state ) 
    plt.xlabel ( "x coordinate [mm]" ) 
    plt.ylabel ( "y coordinate [mm]" ) 
    js = Jscript().hist2d(f"{state}_xy_model", ratio.T, [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]])
    report.add_figure(options=str(js.width("49%"))) ; plt.clf() ; plt.close() 
  

    ########################## RECONSTRUCTED peta ###########################################
    logger("effValidate").info ("Plotting reconstructed peta") 
    #binning = [np.linspace(0,150,150), np.linspace(0.5,6.5,120)] 
    binning = [np.linspace(0,150,50), np.linspace(0.5,6.5,40)] 
    nh = np.zeros ( [binning[0].size-1, binning[1].size-1], dtype = np.float64 ) 
    dh = np.zeros ( [binning[0].size-1, binning[1].size-1], dtype = np.float64 ) 
    for db in generated.iterate():
      db.query ( "acceptance == 1", inplace = True)
      dh += np.histogram2d ( db['p_%s'%state ]*1e-3, db['eta_%s'%state], bins = binning)[0]
    for db in recoed.iterate():
      sdb = db.query ( "type == 3")
      nh += np.histogram2d ( sdb['p_%s'%state ]*1e-3, sdb['eta_%s'%state], bins = binning)[0]

    ratio = np.where (dh > 0, nh/dh, np.min(nh/(dh+1)))
    plt.imshow (ratio.T, extent = [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]], origin = 'lower', aspect='auto') 
    plt.title ( "%s (Brunel reconstruction as Long track)" % state ) 
    plt.xlabel ( "Momentum [GeV/c]" ) 
    plt.ylabel ( "Pseudorapidity" ) 
    js = Jscript().hist2d(f"{state}_peta_brunel", ratio.T, [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]])
    report.add_figure(options=str(js.width("49%"))) ; plt.clf() ; plt.close() 


    ########################## WEIGHTED peta ###########################################
    logger("effValidate").info ("Plotting weighted peta") 
    #binning = [np.linspace(0,150,150), np.linspace(0.5,6.5,120)] 
    binning = [np.linspace(0,150,50), np.linspace(0.5,6.5,40)] 
    nh = np.zeros ( [binning[0].size-1, binning[1].size-1], dtype = np.float64 ) 
    dh = np.zeros ( [binning[0].size-1, binning[1].size-1], dtype = np.float64 ) 

    for db in generated.iterate():
      db.query ( "acceptance == 1", inplace = True)
      dh += np.histogram2d ( db['p_%s'%state ]*1e-3, db['eta_%s'%state], bins = binning)[0]
      w = classifier.predict_proba( db[effBdt.discrVars].values )[:,list(classifier.classes_).index(3)]
      nh += np.histogram2d ( db['p_%s'%state ]*1e-3, db['eta_%s'%state], bins = binning, weights=w)[0]

    ratio = np.where (dh > 0, nh/dh, np.min(nh/(dh+1)))
    plt.imshow (ratio.T, extent = [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]], origin = 'lower', aspect='auto') 
    plt.title ( "%s (GBDT Model)" % state ) 
    plt.xlabel ( "Momentum [GeV/c]" ) 
    plt.ylabel ( "Pseudorapidity" ) 
    js = Jscript().hist2d(f"{state}_peta_model", ratio.T, [binning[0][0],binning[0][-1],binning[1][0],binning[1][-1]])
    report.add_figure(options=str(js.width("49%"))) ; plt.clf() ; plt.close() 

  ## Export data for further analyses
  predictions = pd.DataFrame(np.concatenate([
    np.c_[
      db[effBdt.discrVars + ['type', 'acceptance']].values,
      classifier.predict_proba( db[effBdt.discrVars].values)
      ]
    for db in generated.iterate()
    ]), 
    columns=effBdt.discrVars + ['type', 'acceptance'] +
      [['NotRecoed', '', '', 'Long', 'Upstream', 'Downstream'][v] for v in classifier.classes_]
  )

  report_dir, report_name = os.path.split(report.filename)
  exported_file_name = os.path.join(report_dir, "data", report_name.replace(".html", ".npz"))
  predictions.to_pickle(exported_file_name)
  report.add_markdown(f"[Download dataset](data/{os.path.basename(exported_file_name)})")


  ## Summary plot
  p_boundaries = 1, 2, 5, 15, 30, 50, 150
  eta_boundaries = np.linspace(1, 8, 36)

  hists = [
      ("Long", "Long", "red", 3),
      ("Upstream", "Upstream", "green", 4),
      ("Downstream", "Downstream", "blue", 5),
  ]


  plt.figure(figsize=(14,9))
  for iBin, (pmin, pmax) in enumerate(zip(p_boundaries[:-1], p_boundaries[1:]), 1):
      df_bin = predictions.query(f"acceptance == 1 and p_ClosestToBeam >= {pmin*1e3} and p_ClosestToBeam < {pmax*1e3}")
      plt.subplot(2, 3, iBin)
      
      plt.hist(df_bin.eta_ClosestToBeam, bins=eta_boundaries, alpha=0.1, label="In acceptance")
      plt.hist(df_bin.eta_ClosestToBeam, bins=eta_boundaries, histtype='step', linewidth=0.6, color='black')
      for w, title, c, t in hists:
          plt.hist(
              df_bin.eta_ClosestToBeam, 
              bins=eta_boundaries, 
              weights=df_bin[w], 
              color=c, 
              histtype='step', 
              linewidth=2, 
              alpha=0.5
          )

          x = (eta_boundaries[1:] + eta_boundaries[:-1])/2
          xerr = (eta_boundaries[1] - eta_boundaries[0])/2
          y, _ = np.histogram(df_bin.query(f'type=={t}').eta_ClosestToBeam, bins=eta_boundaries)
          yerr = np.where(y > 0, np.sqrt(y), 0)
          plt.errorbar(
              x, y,
              yerr, xerr,
              fmt='o',
              color=c, 
              label=title,
              markersize=3
          )

          
      plt.xlabel("Pseudorapidity $\eta$")
      plt.ylabel(f"Number of tracks / {(eta_boundaries[-1]-eta_boundaries[0])/(len(eta_boundaries)-1):.2f}")
      plt.title (f"Momentum bin: $p \in [{pmin}, {pmax}]$ GeV/$c$")
      plt.yscale('log')
      plt.ylim(1e-1, plt.ylim()[1])
      
      if iBin == 1:
          plt.legend()
      
  plt.tight_layout()
  report.add_figure(options='width=100%')
  plt.clf()
  plt.close()



################################################################################
  report.add_markdown ("### One-dimensional distributions") 
  ################################################################################
  ## Velo, long and downstream tracks  
  ################################################################################
  pz_bins = [1e2, 1e3, 2e3, 5e3, 10e3, 50e3]
  eta_bins = [1.8, 2.7, 3.5, 4.2, 5.5] 
  report.add_markdown ( "## Velo-, long- and downstream-tracks") 

  recoed    = Dataset.get(cfg.datasets['BrunelRecoed'], max_files = 1, files_key='files_validate') 
  generated = Dataset.get(cfg.datasets['BrunelGenerated'], "acceptance == 1", max_files = 1, files_key='files_validate') 

  plt.hist (generated.eval("log(pz)/log(10)"), bins = 100)
  plt.hist (recoed.eval("log(pz)/log(10)"), bins = 100)
  report.add_figure(options = 'width = 24%') ; plt.clf() ; plt.close() 
  plt.xlabel ( "$\log_{10} (pz/1MeV)$" )

  plt.hist (generated.eval("eta_ClosestToBeam"), bins = 100)
  plt.hist (recoed.eval("eta_ClosestToBeam"), bins = 100)
  plt.xlabel ( "$\eta" )
  report.add_figure(options = 'width = 24%') ; plt.clf() ; plt.close() 



  logger("effValidate").info ( "Loaded %d generated and %d reconstructed events" % (len(generated), len(recoed)) ) 

  tracktypes = [
        #(1, 'Velo'), 
        (3, 'Long'), 
        (4, 'Upstream' ), 
        (5, 'Downstream'), 
      ]

  for tracktype, tracktypename in tracktypes: 
    report.add_markdown ( "### %s tracks" % tracktypename) 
    for hName, hist in cfg.efficiencyBDT.validationHistogramsPBins.items(): 
      for pBin, (pz_min, pz_max) in enumerate(zip ( pz_bins[:-1], pz_bins[1:] )): 
        js = Jscript()
        plt.clf()
        bins = np.linspace ( hist.binning[1], hist.binning[2], hist.binning[0]+1 )
        x = generated.query("(pz > %f) and (pz < %f)" % (pz_min, pz_max)).eval(hist.variable) 
        js.hist(f"{hName}_p{pBin}_g", 
          *plt.hist ( x, bins = bins, label = "Generated", color = 'red', histtype='step', linewidth=2)
        )
        ## 
        x = recoed.query("(pz > %f) and (pz < %f) and (type==%d)" % (pz_min, pz_max, tracktype)).eval(hist.variable) 
        js.hist(f"{hName}_p{pBin}_r", 
          *plt.hist ( x, bins = bins, label = "Reconstructed", color = '#44aaaa', histtype='step', linewidth=2)
          )
        ##
        db = generated.query("(pz > %f) and (pz < %f)" % (pz_min, pz_max))
        x = db.eval(hist.variable) 
        w = classifier.predict_proba( db[effBdt.discrVars].values )[:,list(classifier.classes_).index(tracktype)]
        h, _= np.histogram (x, bins = bins, weights = w)
        plt.errorbar ( 0.5*(bins[:-1]+bins[1:]), h, xerr=0.5*(bins[1]-bins[0]), yerr=np.sqrt(h), label='Weighted', color='black', fmt='o', markersize=2 ) 
        js.errorbar(f"{hName}_p{pBin}_w",0.5*(bins[:-1]+bins[1:]), h, xerr=0.5*(bins[1]-bins[0]), yerr=np.sqrt(h))

        plt.xlabel(
            hist.xtitle if 'xtitle' in hist.keys() else hName)
        plt.ylabel(
            hist.ytitle if 'ytitle' in hist.keys() else "Entries")
        plt.title ( "$p_z$ in (%.1f, %.1f) GeV/$c$" % (pz_min*1e-3, pz_max*1e-3)) 
        plt.legend(title="%s tracks"%tracktypename) 

        js.width("24%")
        report.add_figure(options = str(js)) ; plt.clf() ; plt.close() 

    for hName, hist in cfg.efficiencyBDT.validationHistogramsEtaBins.items(): 
      for etaBin, (eta_min, eta_max) in enumerate(zip ( eta_bins[:-1], eta_bins[1:] )): 
        plt.clf()
        js = Jscript()
        bins = np.linspace ( hist.binning[1], hist.binning[2], hist.binning[0]+1 )
        x = generated.query("(eta_ClosestToBeam > %f) and (eta_ClosestToBeam < %f)" % (eta_min, eta_max)).eval(hist.variable) 
        if len(x) == 0: continue 
        js.hist(f"{hName}_eta{etaBin}_g",
          *plt.hist ( x, bins = bins, label = "Generated", color = 'red', histtype='step', linewidth=2)
          )
        ## 
        x = recoed.query("(eta_ClosestToBeam > %f) and (eta_ClosestToBeam < %f) and (type==%d)" % (eta_min, eta_max, tracktype)).eval(hist.variable) 
        if len(x) == 0: continue 
        js.hist(f"{hName}_eta{etaBin}_r",
          *plt.hist ( x, bins = bins, label = "Reconstructed", color = '#44aaaa', histtype='step', linewidth=2)
          )
        ##
        db = generated.query("(eta_ClosestToBeam > %f) and (eta_ClosestToBeam < %f)" % (eta_min, eta_max))
        if len(x) == 0: continue 
        x = db.eval(hist.variable) 
        w = classifier.predict_proba( db[effBdt.discrVars].values )[:,list(classifier.classes_).index(tracktype)]
        h, _= np.histogram (x, bins = bins, weights = w)
        plt.errorbar ( 0.5*(bins[:-1]+bins[1:]), h, xerr=0.5*(bins[1]-bins[0]), yerr=np.sqrt(h), label='Weighted', color='black', fmt='o', markersize=2 ) 
        js.errorbar(f"{hName}_eta{etaBin}_w",0.5*(bins[:-1]+bins[1:]), h, xerr=0.5*(bins[1]-bins[0]), yerr=np.sqrt(h))

        plt.xlabel(
            hist.xtitle if 'xtitle' in hist.keys() else hName)
        plt.ylabel(
            hist.ytitle if 'ytitle' in hist.keys() else "Entries")
        plt.title ( "$\eta$ in (%.1f, %.1f)" % (eta_min, eta_max)) 
        plt.legend(title="%s tracks"%tracktypename) 

        js.width("24%")
        report.add_figure(options = str(js)) ; plt.clf() ; plt.close() 



