#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:39:14 2020

@author: nozais
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Functionnectome",
    url="https://github.com/NotaCS/Functionnectome",
    version="0.1.0",
    author="Victor Nozais",
    author_email="nozais.victor@gmail.com",
    description="Package containing all the necessary tools to run the Functionnectome method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNUv3 License",
        "Operating System :: OS Independent",
                ],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['FunctionnectomeGUI=Functionnectome.functionnectome_GUI:run_gui',
                                      'Functionnectome=Functionnectome.functionnectome:main'],
                  },
    install_requires=['numpy',
                      'nibabel',
                      'pandas',
                      'h5py'
                      ]
)
