import json
import yaml
import requests 

from logging import getLogger as logger
try:
  from yaml import CLoader as yamlLoader
except ImportError :
  from yaml import Loader as yamlLoader 

class Configuration:
  def __init__ (self, configuration, threads=1):
    self.threads = threads 
    self.hopaas_server = None
    self.hopaas_trial = None

    if isinstance ( configuration, str ): 
      configuration = [configuration]


    _cfg = dict()
    _cfg_root = _cfg; 
    if isinstance ( configuration, (tuple,list) ): 
      for config in configuration:
        logger("Configuration").info ("Loaded %s" % config) 
        with open ( config ) as f:
          self.update_config ( _cfg, yaml.load ( f, Loader = yamlLoader ) )

      self._cfg = _cfg 

    elif isinstance ( configuration, dict ): 
        self._cfg = configuration.copy () 


  def resolve_hopaas(self):
    if self.hopaas_trial is None:
      self._cfg = self.consider_hopaas(self._cfg)

  def consider_hopaas(self, cfg):
    ret = cfg.copy()
    for key, value in cfg.items():
      if key == 'optuna_config':
        return self.submit_to_optuna(cfg)
      elif isinstance(value, dict):
        ret[key] = self.consider_hopaas(value)

    return ret


  def submit_to_optuna(self, cfg):
    self.hopaas_server = cfg['optuna_config']['server']
    res = requests.post(f"http://{self.hopaas_server}/suggest", data=json.dumps(cfg))
    ret = json.loads(res.text)
    self.hopaas_trial = ret['hopaas_trial']
    return ret 

  def should_prune(self, loss, step):
    if self.hopaas_server is None:
      return False 

    res = requests.post(f"http://{self.hopaas_server}/should_prune", data=json.dumps({
        'hopaas_trial': self.hopaas_trial,
        'loss': loss,
        'step': step
      }))

    return json.loads(res.text)

  def finalize_trial(self, loss):
    if self.hopaas_server is not None:
      res = requests.post(f"http://{self.hopaas_server}/tell", data=json.dumps({
          'hopaas_trial': self.hopaas_trial,
          'loss': loss,
        }))



  @staticmethod
  def update_config (_cfg, new_cfg):
    assert isinstance (new_cfg, dict)
    for k,v in new_cfg.items():
      if not isinstance(v, dict) or k not in _cfg.keys():
        _cfg.update ({k:v})
      else:
        Configuration.update_config (_cfg[k], v) 

    
  def __call__ ( self, slot, default = None ): 
    keys = slot.split ( "/" ) 
    ret = self._cfg 
    for key in keys:
      try: 
        ret = ret [ key ] 
      except KeyError:
        return default 

    if isinstance (ret, dict): 
      return Configuration ( ret, self.threads )
    else:
      return ret 

  def keys(self):
    return self._cfg.keys()

  def values(self):
    return self._cfg.values()

  def items(self):
    for dname, obj in self._cfg.items(): 
      if isinstance (obj, dict):
        yield dname, Configuration(obj, self.threads) 
      else:
        yield dname, obj
      

  def __getitem__ (self, key):
    ret = self._cfg [key] 
    if isinstance (ret, dict): 
      return Configuration ( ret, self.threads )
    else:
      return ret 

  def __getattr__ (self, key):
    if key not in self._cfg.keys():
      raise AttributeError (key) 
    return self.__getitem__ (key) 

  def __setitem__ (self, key, newval):
    if key not in self._cfg.keys():
      self._cfg[key] = newval 
    elif type(newval) == self._cfg[key]:
      self._cfg[key] = newval 
    else:
      raise TypeError("Changing type of configuration entries is not permitted")


