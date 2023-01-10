# only for development
# run  with
# python setup.py build_ext --inplace
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(ext_modules=cythonize("estimateDispersion.pyx", language_level=3,
      include_path=[np.get_include()]))  # , annotate=True
setup(ext_modules=cythonize("negativeBinomialGLM.pyx", language_level=3,
      include_path=[np.get_include()]))  # , annotate=True
