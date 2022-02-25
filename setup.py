# -*- coding: utf-8 -*-


from setuptools import setup, Extension
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as infile:
    long_description = infile.read()

version = {}
with open(path.join(here, 'ttvfast', 'version.py')) as infile:
    exec(infile.read(), version)

setup(
    name='exohammer',
    version=version['__version__'],
    description='TTV and RV MCMC generator',
    url='https://github.com/nick-juliano/exohammer',
    long_description=long_description,
    author='Nick Juliano',
    author_email='nick_juliano@icloud.com',
    license='GPL',
    packages=['exohammer', ],
    install_requires=['numpy', 'ttvfast', 'emcee'],
    ext_modules=[exohammer, ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)