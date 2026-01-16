#!/usr/bin/env python

import os

from setuptools import find_packages, setup

install_requires = [
    "torch>=1.11.0",
    "torchvision>=0.11.0",
    "lightning-bolts",
    "matplotlib",
    "numpy",
    "scipy",
    "scikit-learn",
    "scprep",
    "scanpy",
    "torchdyn",
    "pot",
    "torchdiffeq",
    "absl-py",
    "clean-fid",
]

version_py = os.path.join(os.path.dirname(__file__), "torchcfm", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()
readme = open("README.md", encoding="utf8").read()
setup(
    name="mixflow",
    version=version,
    description="MixFlow: Mixture-Conditioned Flow Matching for Out-of-Distribution Generalization",
    author="Andrea Rubbi",
    author_email="andrea.rubbi.98@gmail.com",
    url="https://github.com/Lotfollahi-lab/mixflow",
    install_requires=install_requires,
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
)
