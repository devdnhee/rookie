# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import subprocess

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Rookie',

    version='1.0',

    description='Rookie, teaching computers to play chess through DRL',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/devdnhee/rookie',

    # Author details
    author='Dorian Van den Heede',
    author_email='devdnhee@gmail.com',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',

        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords='Chess, Artificial Intelligence, Deep Learning, Reinforcement Learning, Masters thesis',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['peppercorn', 'scikit-learn', 'tensorflow', 'python-chess', 'numpy'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    } 
)
subprocess.call('sh setup.sh', shell=True)
