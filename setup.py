from setuptools import setup, find_packages
from os import path
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='flares-utov',
    version='0.0.1-beta',    
    description='A Python lib for Solar Flares identification and classification',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='TBD',
    author='Michele Berretti, Simone Mestici, Helio-Tov Team',
    author_email='michele.berretti@unitn.it',
    license='GPL-3.0',
    packages=['asr'],
    install_requires=['pandas',
                      'numpy',
                      'numba',
                      'scipy',
                      'matplotlib',   
                      'tqdm',
                      'typing',
                      ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
)
