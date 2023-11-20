#!/usr/bin/env python3
from setuptools import setup, find_packages
from its_jointprobability._version import __version__

setup(
    name="its-jointprobability",
    version=__version__,
    description="A Bayesian approach to metadata prediction for IT's Jointly",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        d for d in open("requirements.txt").readlines() if not d.startswith("--")
    ],
    package_dir={"": "."},
    entry_points={
        "console_scripts": [
            "its-jointprobability = its_jointprobability.webservice:main",
            "retrain-model = its_jointprobability.models.prodslda_sep:retrain_model_cli",
        ]
    },
)
