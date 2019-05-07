from .tools import *  # noqa

# Disable autopadding to avoid polluting the generated example code
from devito import configuration
configuration['autopadding'] = False
