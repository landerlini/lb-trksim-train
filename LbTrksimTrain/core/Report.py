import base64
import numpy as np 
import html_reports
from datetime import datetime 

class Report:
  """
  Simple context manager for html_reports 
  """
  def __init__ ( self, title, filename ):
    self.title = title 
    self.filename = filename 
    self.report = html_reports.Report(self)
    self.report.add_markdown ( "# %s" % title )
    self.report.add_markdown ( "Initialized at %s" % str(datetime.now()) )
    self.report.title = title 
    self.report.filename = filename 

  def __enter__ (self):
    return self.report 

  def __exit__ (self, type, value, traceback):
    self.report.add_markdown ( "Saved at %s" % str(datetime.now()) )
    self.report.write_report ( filename = self.filename ) 


class Jscript:
  def __init__ (self, method='onclick'):
    self._method = method
    self._code = [
        "def parse (c,t):\\n  return np.frombuffer(base64.b64decode(c), dtype=t)\\n\\n"
        ]
    self._options = []

  @staticmethod
  def export (name, var):
    var = np.array(var)
    return f'{name} = parse(\\"{str(base64.b64encode(var.tobytes()),"ascii")}\\", \\"{str(var.dtype)}\\")\\n'

  def hist (self, name, contents, boundaries, *_):
    name = name.lower().replace(" ", "_")
    self._code.append (
        self.export(f"{name}_boundaries", boundaries) + 
        self.export(f"{name}_contents", contents) + 
        f"{name}_xAxis      = 0.5*({name}_boundaries[1:]+{name}_boundaries[:-1])\\n"+
        f"""plt.hist({name}_xAxis, bins={name}_boundaries, weights={name}_contents,"""+
        f"""linewidth=2, histtype=\\"step\\", label=\\"{name}\\")\\n"""
        )

    return self

  def errorbar (self, name, x, y, yerr, xerr=None):
    name = name.lower().replace(" ", "_")
    self._code.append (
        self.export(f"{name}_x", x) + 
        self.export(f"{name}_y", y) 
        )

    if xerr is None:
      self._code.append(f"{name}_xerr = None\\n")
    else:
      if not hasattr(xerr, '__iter__'): xerr = np.full_like(x, xerr)
      self._code.append(self.export(f"{name}_xerr", xerr))

    if yerr is None:
      if not hasattr(yerr, '__iter__'): yerr = np.full_like(y, yerr)
      self._code.append(f"{name}_yerr = None\\n")
    else:
      self._code.append(self.export(f"{name}_yerr", yerr))

    self._code.append (
        f"""plt.errorbar({name}_x, {name}_y, {name}_yerr, {name}_xerr, \\".\\", """+
        f"""linewidth=1, label=\\"{name}\\")\\n"""
        )

    return self

  def hist2d (self, name, contents, extent):
    self._code.append(
        self.export(f"{name}_contents", contents)+
        f'plt.imshow ({name}_contents.reshape({contents.shape}), extent={extent}, origin=\\"lower\\")\\n'
        )
    return self

  def width(self, width):
    self._options.append (f"width='{width}'")
    return self 

  def __str__ (self):
    return f"""{self._method}='console.log("{"".join(self._code)}");' {" ".join(self._options)}"""

  def __iadd__ (self, other):
    if isinstance(other, str):
      return other + str(self) 
    
  def __add__ (self, other):
    if isinstance(other, str):
      return str(self) + other
    





