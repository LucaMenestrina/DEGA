__name__ = "DEGA"
__version__ = "0.1.1"
__author__ = "Luca Menestrina"
__license__ = "GPLv3"
__doc__ = """DEGA: a Python package for differential gene expression analysis
================================================================

**Differential gene expression analysis** is an important tool
for identifying genes that display a significantly altered expression
in response to specific stimuli.
**DEGA** is a Python package for differential expression analysis.
It is an implementation of the core algorithm of the R package DESeq2.
Along with the differential testing algorithm,
DEGA also provides high-level functions for **dataset exploration**
and **results interpretation**.

Notes
-----
For a complete use case, see the [Jupyter Notebook](https://github.com/LucaMenestrina/DEGA/blob/main/validation/DEGA.ipynb).
"""

from DEGA.core import dataset
