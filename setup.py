#!/usr/bin/env python
"""Setup file for the library."""

from setuptools import find_packages  # or find_namespace_packages
from setuptools import setup

# https://setuptools.pypa.io/en/latest/userguide/quickstart.html
# https://setuptools.pypa.io/en/latest/references/keywords.html

__version__ = '0.0.1'
__author__ = 'Diego Francisco Valenzuela Iturra'
__author_email__ = 'diegovalenzuelaiturra@gmail.com'
__description__ = 'A simple package for the challenge of the Data Scientist position at NeuralWorks.'
# __description__ = 'Data Scientist Challenge - NeuralWorks'

__version__ = '0.0.1'

setup(
    name='neuralworks',
    version=__version__,
    packages=find_packages(),
    description=__description__,
    author=__author__,
    author_email=__author_email__,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    python_requires='>=3.8',
)
