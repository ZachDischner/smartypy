#!/usr/bin/env python
# coding: utf-8

## Taken from: https://github.com/poliastro/poliastro/blob/master/setup.py
# Work in progress...

# http://stackoverflow.com/a/10975371/554319
import io
from setuptools import setup, find_packages


# http://blog.ionelmc.ro/2014/05/25/python-packaging/
setup(
    name="smartypy",
    version='0.0.0',
    description="Python package for Machine Learning",
    author="Zach Dischner",
    author_email="zach.dischner@gmail.com",
    url="https://github.com/ZachDischner/smartypy",
    download_url="hhttps://github.com/ZachDischner/smartypy",
    license="MIT",
    keywords=[
      "machine learning"
    ],
    install_requires=[
        "numpy",
        "numba>=0.25",
        "matplotlib",
        "pandas"
    ],
    tests_require=[
        #"pytest"
    ],
    packages=find_packages('smartypy'),
    package_dir={'': 'smartypy'},
    entry_points={
      'console_scripts': [
          'smartypy = smartypy.cli:main'
      ]
    },
    classifiers=[
      "Development Status :: 4 - Beta",
      "Intended Audience :: Education",
      "Intended Audience :: Science/Research",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
      "Programming Language :: Python",
      "Programming Language :: Python :: 3.5",
      "Programming Language :: Python :: Implementation :: CPython",
      "Topic :: Scientific/Engineering",
    ],
    long_description=io.open('README.md', encoding='utf-8').read(),
    zip_safe=False,
)
