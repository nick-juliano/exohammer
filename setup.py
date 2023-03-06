# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as infile:
	long_description = infile.read()

version = {}
with open(path.join(here, 'exohammer', 'version.py')) as infile:
	exec(infile.read(), version)

with open(path.join(here, 'requirements.txt')) as f:
	required = f.read().splitlines()

setup(
	name='exohammer',
	version=version['__version__'],
	description='TTV and RV MCMC generator',
	url='https://github.com/nick-juliano/exohammer',
	long_description=long_description,
	author='Nick Juliano',
	author_email='nick_juliano@icloud.com',
	license='GPL',
	packages=find_packages(),
	include_package_data=True,
	install_requires=required,
	extras_require={"extras": ["pickle", "matplotlib", "easygui"], }
)
