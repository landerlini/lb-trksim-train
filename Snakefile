import LbTrksimTrain
from pathlib import Path 

BASEDIR = Path(LbTrksimTrain.__file__).parents[0] 
LOGDIR = BASEDIR / 'log'
REPORTDIR = BASEDIR / 'reports'
CONFIGDIR = BASEDIR / 'config' 
OUTDIR = BASEDIR / 'results'
MODELDIR = OUTDIR / 'models'
GENCDIR = OUTDIR / 'generatedC' 
EXPORTDIR = OUTDIR / 'compiledC' 
CPIPELINES = BASEDIR / 'exporters' / 'cpipelines' 

COMMONCFG = [
      CONFIGDIR / "tracking.yaml", 
  ]

slots = ["2016-MagDown", "2016-MagUp"] 
types = ["long", "upstream", "downstream"]


################################################################################
##  A C C E P T A N C E 
################################################################################
rule accTrain:
  input:
    config = COMMONCFG + [
      CONFIGDIR / "{slot}.yaml"
    ]

  output: MODELDIR/"acceptance_{slot}.pkl"
    
  params:
    report = lambda w: REPORTDIR/f"accTrain_{w['slot']}.html"

  log: LOGDIR/"accTrain_{slot}.log" 

  shell:
    "python3 -m LbTrksimTrain accTrain"
    " -F {input.config}"
    " -r {params.report}"
    " -m {output}"
    " > {log}"


rule accValidate:
  input:
    config = COMMONCFG + [
      CONFIGDIR / "validation.yaml", 
      CONFIGDIR / "{slot}.yaml"
    ],
    model = MODELDIR / "acceptance_{slot}.pkl"

  output: REPORTDIR / "accValidate_{slot}.html" 
    
  log: LOGDIR/"accValidate_{slot}.log" 

  shell:
    "python3 -m LbTrksimTrain accValidate"
    " -F {input.config}"
    " -r {output}"
    " -m {input.model}"
    " > {log}"


rule accConvert:
  input: MODELDIR/"acceptance_{slot}.pkl"

  output: GENCDIR/"acceptance_{slot}.C"

  shell:
    "scikinC acceptance={input} > {output}"

rule accCompile:
  input: GENCDIR/"acceptance_{slot}.C"

  output: EXPORTDIR/"acceptance_{slot}.so"

  shell:
    "gcc -o {output} {input} -shared -fPIC -Ofast"



################################################################################
##  E F F I C I E N C Y
################################################################################
rule effTrain:
  input:
    config = COMMONCFG + [
      CONFIGDIR/"{slot}.yaml"
    ]

  output: MODELDIR/"trkEfficiency_{slot}.pkl"
    
  threads: 18

  log: LOGDIR/"effTrain_{slot}.log" 

  params:
    report = lambda w: REPORTDIR/f"effTrain_{w['slot']}.html"

  shell:
    "python3 -m LbTrksimTrain effTrain"
    " -F {input.config}"
    " -r {params.report}"
    " -j {threads}"
    " -m {output}"
    "  > {log}"


rule effValidate:
  input:
    config = COMMONCFG + [
      CONFIGDIR/"validation.yaml", 
      CONFIGDIR/"{slot}.yaml"
    ],
    model = MODELDIR/"trkEfficiency_{slot}.pkl"

  output: REPORTDIR/"effValidate_{slot}.html" 
    
  threads: 18

  log: LOGDIR/"effValidate_{slot}.log" 

  shell:
    "python3 -m LbTrksimTrain effValidate"
    " -F {input.config}"
    " -r {output}"
    " -j {threads}"
    " -m {input.model}"
    " > {log}"


rule effConvert:
  input: MODELDIR/"trkEfficiency_{slot}.pkl"

  output: GENCDIR/"trkEfficiencyTrav_{slot}.C"

  shell:
    "scikinC trkEfficiency={input} > {output}"


rule effCompile:
  input: GENCDIR/"trkEfficiencyTrav_{slot}.C"

  output: EXPORTDIR/"trkEfficiencyTrav_{slot}.so"

  shell:
    "gcc -o {output} {input} -shared -fPIC -Ofast"



################################################################################
##  R E S O L U T I O N 
################################################################################

rule resTrain:
  input:
    config = COMMONCFG + [
      CONFIGDIR/"{slot}.yaml"
    ]


  log: LOGDIR/"resTrain_{slot}.log" 

  output: 
    directory(MODELDIR/'resolution_{slot}')

  params:
    report = lambda w: REPORTDIR/f"resolution_{w['slot']}.html"

  threads: 18

  shell:
    "python3 -m LbTrksimTrain resTrain"
    " -F {input.config}"
    " -r {params.report}"
    " -m {output}"
    " -j {threads}"
    " > {log}"


rule resValidation:
  input: 
    model = MODELDIR/'resolution_{slot}', 
    config = COMMONCFG + [
      CONFIGDIR/"validation.yaml",
      CONFIGDIR/"{slot}.yaml",
    ]

  output: REPORTDIR/"resValidate_{slot}.html" 

  threads: 18

  log: LOGDIR/"resValidate_{slot}.log" 
    

  shell:
    "python3 -m LbTrksimTrain resValidate"
    " -F {input.config}"
    " -r {output}"
    " -m {input.model}"
    " -j {threads}"
    " > {log}"


rule resConvert:
  input: MODELDIR/'resolution_{slot}'
  
  output: GENCDIR/'resolution_{slot}_{type}.C'
  
  shell: 
    "scikinC res{wildcards.type}={input}/{wildcards.type} > {output};"


rule resConvertPreprocessing:
  input: MODELDIR/'resolution_{slot}'
  
  output: GENCDIR/'resolution-t{XY}_{slot}_{type}.C'

  params:
    pythonpath = BASEDIR.parents[0]
  
  shell: 
    "PYTHONPATH={params.pythonpath} "
    "scikinC res{wildcards.type}_t{wildcards.XY}={input}/{wildcards.type}/transformer{wildcards.XY}_ > {output};"


rule resCompile:
  input: 
    CPIPELINES/"resolution.C",
    expand(GENCDIR/"resolution_{slot}_{type}.C", type=types, allow_missing=True),
    expand(GENCDIR/"resolution-tX_{slot}_{type}.C", type=types, allow_missing=True),
    expand(GENCDIR/"resolution-tY_{slot}_{type}.C", type=types, allow_missing=True),

  output: 
    EXPORTDIR/"resolution_{slot}.so"

  shell:
    "gcc -o {output} {input} -shared -fPIC -Ofast"



################################################################################
##  C O V A R I A N C E
################################################################################

rule covTrain:
  input:
    config = COMMONCFG + [
      CONFIGDIR/"{slot}.yaml"
    ]


  log: LOGDIR/"covTrain_{slot}.log" 

  output: 
    directory(MODELDIR/'covariance_{slot}')

  params:
    report = lambda w: REPORTDIR/f"resolution_{w['slot']}.html"

  shell:
    "python3 -m LbTrksimTrain covTrain"
    " -F {input.config}"
    " -r {params.report}"
    " -m {output}"
    " > {log}"


rule covValidation:
  input: 
    config = COMMONCFG + [
      CONFIGDIR/"validation.yaml",
      CONFIGDIR/"{slot}.yaml",
      ],
    model = MODELDIR/'covariance_{slot}'

  output: REPORTDIR/"covValidate_{slot}.html" 

  log: LOGDIR/"covValidate_{slot}.log" 
    

  shell:
    "python3 -m LbTrksimTrain covValidate"
    " -F {input.config}"
    " -r {output}"
    " -m {input.model}"
    " > {log}"


rule covConvert:
  input: MODELDIR/'covariance_{slot}'
  
  output: GENCDIR/'covariance_{slot}_{type}.C'
  
  shell: 
    "scikinC cov{wildcards.type}={input}/{wildcards.type} > {output}"


rule covConvertPreprocessing:
  input: MODELDIR/'covariance_{slot}'
  
  output: GENCDIR/'covariance-t{XY}_{slot}_{type}.C'

  params:
    pythonpath = BASEDIR.parents[0]
  
  shell: 
    "PYTHONPATH={params.pythonpath} "
    "scikinC cov{wildcards.type}_t{wildcards.XY}={input}/{wildcards.type}/transformer{wildcards.XY}_ > {output};"


rule covCompile:
  input: 
    CPIPELINES/"covariance.C",
    expand(GENCDIR/"covariance_{slot}_{type}.C", type=types, allow_missing=True),
    expand(GENCDIR/"covariance-tX_{slot}_{type}.C", type=types, allow_missing=True),
    expand(GENCDIR/"covariance-tY_{slot}_{type}.C", type=types, allow_missing=True),

  output: 
    EXPORTDIR/"covariance_{slot}.so"

  shell:
    "gcc -o {output} {input} -shared -fPIC -Ofast"


################################################################################
##  G R O U P E D   R U L E S
################################################################################


rule train_slot:
  input: 
    acceptance = MODELDIR/"acceptance_{slot}.pkl", 
    efficiency = MODELDIR/"trkEfficiencyTrav_{slot}.pkl", 
    resolution = expand(MODELDIR/'resolution-{type}_{slot}', type=types, allow_missing=True),
    covariance = expand(MODELDIR/'covariance-{type}_{slot}', type=types, allow_missing=True),

  output:
    "train_{slot}"

  shell: "touch {output}"
    
rule validate_slot:
  input:
    acceptance = REPORTDIR/"accValidate_{slot}.html",
    efficiency = REPORTDIR/"effValidate_{slot}.html",
    resolution = REPORTDIR/"resValidate_{slot}.html",
    covariance = REPORTDIR/"covValidate_{slot}.html",

  output:
    "validate_{slot}"

  shell: "touch {output}"


rule trainall:
  input:
    expand("train_{slot}", slot=slots) 

rule validateall:
  input:
    expand("validate_{slot}", slot=slots) 


rule compile_all:
  input: 
    EXPORTDIR/"acceptance_{slot}.so",
    EXPORTDIR/"trkEfficiencyTrav_{slot}.so",
    expand(EXPORTDIR/"resolution_{slot}.so", type=types, allow_missing=True),
    expand(EXPORTDIR/"covariance_{slot}.so", type=types, allow_missing=True),
  
  output:
    "compile_{slot}"

  shell:
    "touch {output}"


