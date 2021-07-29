"""
Definition of the CLI for configuring the training steps.
"""

import sys 
import argparse 
from datetime import datetime 
import logging 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from LbTrksimTrain.core import Configuration 
from LbTrksimTrain.core import Dataset 
from LbTrksimTrain.core import CachedDataset 
from LbTrksimTrain.core import Report 

FORMAT = '%(asctime)-15s %(module)-15s %(levelname)-8s %(message)s'
logging.basicConfig(format=FORMAT, stream=sys.stdout, level = logging.INFO)

parser = argparse.ArgumentParser () 
parser.add_argument ( 'commands', nargs = "*", default = 'help', 
    choices = [
      'help',
      'acceptance', 
      'genlevelcheck', 
      'efficiency', 
      'effTrain', 
      'effValidate', 
      'accTrain', 
      'accValidate', 
      'resTrain',
      'resValidate',
      'covTrain',
      'covValidate'
      ]) 

parser.add_argument ( '--report', '-r', type=str, required=True, help="Path to html report")
parser.add_argument ( '--modelpath', '-m', type=str, required=True, help="Path to the model file or dir")
parser.add_argument ( '--config', '-F', nargs="*", type=str, default = ["config.yaml"] )
parser.add_argument ( '--verbose', '-v', action='store_true' )
parser.add_argument ( '--threads', '-j', default=18, type=int, help = "Maximum allowed number of threads")

args = parser.parse_args() 
cfg = Configuration ( args.config, args.threads ) 

if 'help' in args.commands:
  print ("""
    Toolset for the tracking parametrization in LHCb 
  """)


if args.verbose: 
  logging.getLogger().setLevel(logging.DEBUG) 

################################################################################
###  A C C E P T A N C E  
################################################################################
if 'acceptance' in args.commands:
  from LbTrksimTrain.steps import acceptance 
  with Report ("Acceptance", args.report) as report: 
    plt.inferno() 
    acceptance.plot ( cfg, report ) 


################################################################################
###  G E N L E V E L C H E C K
################################################################################
if 'genlevelcheck' in args.commands: 
  from LbTrksimTrain.steps import genlevelcheck
  with Report ("Generator Level Check", args.report) as report:
    plt.inferno() 
    genlevelcheck.plot ( cfg, report )



################################################################################
###  E F F I C I E N C Y 
################################################################################
if 'efficiency' in args.commands:
  from LbTrksimTrain.steps import efficiency
  with Report ("Efficiency", args.report) as report:
    plt.inferno()
    efficiency.plot ( cfg, report ) 



################################################################################
###  A C C E P T A N C E   T R A I N I N G
################################################################################
if 'accTrain' in args.commands:
  from LbTrksimTrain.steps import acceptance 
  with Report ("Training Acceptance", args.report) as report: 
    plt.inferno()
    acceptance.train (cfg, report, modelfile=args.modelpath) 

if 'accValidate' in args.commands:
  from LbTrksimTrain.steps import acceptance 
  with Report ("Validation Acceptance", args.report) as report: 
    plt.inferno()
    acceptance.validate (cfg, report, modelfile=args.modelpath) 



################################################################################
###  E F F I C I E N C Y   T R A I N I N G
################################################################################
if 'effTrain' in args.commands:
  from LbTrksimTrain.steps import efficiency 
  with Report ("Training Efficiency", args.report) as report:
    plt.inferno() 
    efficiency.train (cfg, report, modelfile=args.modelpath) 

if 'effValidate' in args.commands:
  from LbTrksimTrain.steps import efficiency 
  with Report ("Validation Efficiency", args.report) as report:
    plt.inferno() 
    efficiency.validate (cfg, report, modelfile=args.modelpath) 



################################################################################
###  R E S O L U T I O N   T R A I N I N G 
################################################################################
if 'resTrain' in args.commands:
  from LbTrksimTrain.steps import resolution 
  with Report ("Training Resolution", args.report) as report: 
    resolution.train ( cfg, report, modeldir=args.modelpath) 

if 'resValidate' in args.commands:
  from LbTrksimTrain.steps import resolution 
  with Report ("Validation Resolution", args.report) as report: 
    resolution.validate ( cfg, report, modeldir=args.modelpath) 


################################################################################
###  C O V A R I A N C E   T R A I N I N G 
################################################################################
if 'covTrain' in args.commands:
  from LbTrksimTrain.steps import covariance 
  with Report ("Training Covariance", args.report) as report: 
    covariance.train ( cfg, report, modeldir=args.modelpath) 

if 'covValidate' in args.commands:
  from LbTrksimTrain.steps import covariance 
  with Report ("Validation Covariance", args.report) as report: 
    covariance.validate ( cfg, report, modeldir=args.modelpath) 




