# setup.py
from setuptools import setup, find_packages

setup(
    name="spectral_decomposition",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "spectral-connectivity"
    ],
    python_requires=">=3.7",
    description="A package for simulating and decomposing PSDs",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/yourname/spectral_decomposition",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
