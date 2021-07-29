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



