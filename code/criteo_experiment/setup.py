from os import path
import pip
import re
from setuptools import setup, find_packages
import sys

here = path.abspath(path.dirname(__file__))

pip_version_match = re.search(r'^[0-9]*', pip.__version__)
if pip_version_match:
    if int(pip_version_match.group(0)) < 19:
        sys.exit('Install requires pip version 19 or greater.  ' +
                 'Run pip install --upgrade pip.')
else:
    sys.exit('There was an error getting the pip version number.')

# Parse requirements.txt, ignoring any commented-out or git lines.
with open(path.join(here, 'requirements.txt')) as requirements_file:
    requirements_lines = requirements_file.read().splitlines()

requirements = [line for line in requirements_lines
                if not (line.startswith('#')) ]

setup(
    name='criteo_experiment_jmlr1951',
    version='0.1',
    description='Code to run the Criteo experiments in JMLR v19 51.',
    author='Ryan Giordano',
    author_email='rgiordano@berkeley.edu',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Development Status :: 2 - Pre-Alpha',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
)
