#!/usr/bin/env python2
import setuptools


setuptools.setup(
    name='UEDGE_utils',
    packages=setuptools.find_packages(),
    install_requires=['h5py', 'mppl', 'forthon', 'uedge', 'matplotlib', 
                      'numpy', 'scipy', 'pandas', 'shapely'],
    author='Sean Ballinger',
    author_email='sballin@mit.edu',
    url='https://github.com/sballin/UEDGE_utils',
    description='Utilities for the UEDGE plasma simulation code.',
    long_description=open('README.md').read(),
)
