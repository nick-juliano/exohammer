# -*- coding: utf-8 -*-


from setuptools import setup, Extension
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# ttvfast = Extension("ttvfast._ttvfast",
#             sources=["src/ttvfast_wrapper.c",
#                 "external/TTVFast/c_version/TTVFast.c"],
#             include_dirs=['external/TTVFast/c_version'],
#             extra_compile_args=['-std=c99'],
#             # Debug mode
#             # define_macros=[('DEBUG', True)],
#             )

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
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)