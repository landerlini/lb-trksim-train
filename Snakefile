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


slots = ["2016-MagDown", "2016-MagUp"] 
types = ["long", "upstream", "downstream"]


################################################################################
##  A C C E P T A N C E 
################################################################################
rule accTrain:
  input:
    config = [
      CONFIGDIR / "tracking-DEBUG.yaml", 
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
    "> {log}"


rule accValidate:
  input:
    config = [
      CONFIGDIR / "tracking-DEBUG.yaml", 
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
    "> {log}"


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
    config = [
      CONFIGDIR/"tracking-DEBUG.yaml", 
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
    "> {log}"


rule effValidate:
  input:
    config = [
      CONFIGDIR/"tracking-DEBUG.yaml", 
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
    "> {log}"


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
    config = [
      "LbTrksimTrain/config/tracking.yaml", 
      "LbTrksimTrain/config/{slot}.yaml"
    ]


  log: "LbTrksimTrain/log/resTrain_{slot}.log" 

  output: 
    directory('resolution_{slot}')

  threads: 18

  shell:
    "python3 -m LbTrksimTrain resTrain"
    " -F {input.config}"
    " -R LbTrksimTrain/reports"
    " -m {output}"
    " -j {threads}"
    "> {log}"


rule resValidation:
  input: 
    model = 'resolution_{slot}', 
    config = [
      "LbTrksimTrain/config/tracking.yaml", 
      "LbTrksimTrain/config/validation.yaml",
      "LbTrksimTrain/config/{slot}.yaml",
    ]

  output: "LbTrksimTrain/reports/resValidate_{slot}.html" 

  threads: 18

  log: "LbTrksimTrain/log/resValidate_{slot}.log" 
    

  shell:
    "python3 -m LbTrksimTrain resValidate"
    " -F {input.config}"
    " -R LbTrksimTrain/reports"
    " -o {input.model}"
    " -j {threads}"
    "> {log}"


rule resConvert:
  input: 'resolution-{type}_{slot}'
  
  output: 'generatedC/resolution-{type}_{slot}.C'
  
  shell: 
    "scikinC res{wildcards.type}={input} > {output};"


rule resConvertPreprocessing:
  input: 'resolution-{type}_{slot}/transformer{XY}_'
  
  output: 'generatedC/resolution-{type}_{slot}-t{XY}.C'
  
  shell: 
    "scikinC res{wildcards.type}_t{wildcards.XY}={input} > {output};"



rule resCompile:
  input: 
    "cpipelines/resolution.C",
    expand("generatedC/resolution-{type}_{slot}.C", type=types, allow_missing=True),
    expand("generatedC/resolution-{type}_{slot}-tX.C", type=types, allow_missing=True),
    expand("generatedC/resolution-{type}_{slot}-tY.C", type=types, allow_missing=True),

  output: 
    "compiledC/resolution_{slot}.so"

  shell:
    "gcc -o {output} {input} -shared -fPIC -Ofast"



rule covTrain:
  input:
    config = [
      "LbTrksimTrain/config/tracking.yaml", 
      "LbTrksimTrain/config/{slot}.yaml"
    ]


  log: "LbTrksimTrain/log/covTrain_{slot}.log" 

  output: 
    directory('covariance_{slot}')

  shell:
    "python3 -m LbTrksimTrain covTrain"
    " -F {input.config}"
    " -R LbTrksimTrain/reports"
    " -m {output}"
    "> {log}"


rule covValidation:
  input: 
    config = [
      "LbTrksimTrain/config/tracking.yaml", 
      "LbTrksimTrain/config/validation.yaml",
      "LbTrksimTrain/config/{slot}.yaml",
      ],
    model = 'covariance_{slot}'

  output: "LbTrksimTrain/reports/{slot}-covValidate.html" 

  log: "LbTrksimTrain/log/covValidate_{slot}.log" 
    

  shell:
    "python3 -m LbTrksimTrain covValidate"
    " -F {input.config}"
    " -R LbTrksimTrain/reports"
    " -m {input.model}"
    "> {log}"


rule covConvert:
  input: 'covariance-{type}_{slot}'
  
  output: 'generatedC/covariance-{type}_{slot}.C'
  
  shell: 
    "scikinC cov{wildcards.type}={input} > {output}"


rule covConvertPreprocessing:
  input: 'covariance-{type}_{slot}/transformer{XY}_'
  
  output: 'generatedC/covariance-{type}_{slot}-t{XY}.C'
  
  shell: 
    "scikinC cov{wildcards.type}_t{wildcards.XY}={input} > {output};"


rule covCompile:
  input: 
    "cpipelines/covariance.C",
    expand("generatedC/covariance-{type}_{slot}.C", type=types, allow_missing=True),
    expand("generatedC/covariance-{type}_{slot}-tX.C", type=types, allow_missing=True),
    expand("generatedC/covariance-{type}_{slot}-tY.C", type=types, allow_missing=True),

  output: 
    "compiledC/covariance_{slot}.so"

  shell:
    "gcc -o {output} {input} -shared -fPIC -Ofast"


rule compile_all:
  input: 
    "compiledC/acceptance_{slot}.so",
    "compiledC/trkEfficiencyTrav_{slot}.so",
    expand("compiledC/resolution_{slot}.so", type=types, allow_missing=True),
    expand("compiledC/covariance_{slot}.so", type=types, allow_missing=True),
  
  output:
    "compile_{slot}"

  shell:
    "touch {output}"




rule train_slot:
  input: 
    acceptance = "acceptance_{slot}.pkl", 
    efficiency = "trkEfficiencyTrav_{slot}.pkl", 
    resolution = expand('resolution-{type}_{slot}', type=types, allow_missing=True),
    covariance = expand('covariance-{type}_{slot}', type=types, allow_missing=True),

  output:
    "train_{slot}"

  shell: "touch {output}"
    
rule validate_slot:
  input:
    acceptance = "LbTrksimTrain/reports/{slot}-accValidate.html",
    efficiency = "LbTrksimTrain/reports/{slot}-effValidate.html",
    resolution = "LbTrksimTrain/reports/{slot}-resValidate.html",
    covariance = "LbTrksimTrain/reports/{slot}-covValidate.html",

  output:
    "validate_{slot}"

  shell: "touch {output}"


rule trainall:
  input:
    expand("train_{slot}", slot=slots) 

rule validateall:
  input:
    expand("validate_{slot}", slot=slots) 
