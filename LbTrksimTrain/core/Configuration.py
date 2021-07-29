import yaml
from logging import getLogger as logger
try:
  from yaml import CLoader as yamlLoader
except ImportError :
  from yaml import Loader as yamlLoader 

class Configuration:
  def __init__ (self, configuration, threads=1):
    self.threads = threads 

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


