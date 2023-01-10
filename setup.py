from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "DEGA.cython.estimateDispersion",
        ["DEGA/cython/estimateDispersion.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "DEGA.cython.negativeBinomialGLM",
        ["DEGA/cython/negativeBinomialGLM.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="DEGA",
    package_dir={"DEGA": "DEGA", "DEGA.cython": "DEGA/cython"},
    packages=["DEGA", "DEGA.cython"],
    ext_modules=cythonize(extensions, language_level=3,
                          include_path=[np.get_include()]),
    install_requires=["cython", "pythran", "numpy", "scipy"],
)
