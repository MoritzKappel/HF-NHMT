'''projectpath.py: sets up project directory structure for imports.'''

import os
import sys


# adds source file locations to python path
class context:

    def __enter__(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

    def __exit__(self, *args):
        sys.path.pop(0)
