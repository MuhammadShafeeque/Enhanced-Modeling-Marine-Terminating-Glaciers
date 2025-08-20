import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="oggm_marine_terminating",
    version="0.1.0",
    author="Muhammad Shafeeque",
    author_email="shafeequ@uni-bremen.de",
    description="Enhanced modeling of marine-terminating glaciers for OGGM",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/MuhammadShafeeque/Enhanced-Modeling-Marine-Terminating-Glaciers",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "oggm>=1.5.3",
        "numpy>=1.17.0",
        "scipy>=1.3.0",
        "pandas>=1.0.0",
        "matplotlib>=3.1.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Glaciology",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
)
