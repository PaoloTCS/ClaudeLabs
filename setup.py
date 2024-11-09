# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="claudelabs",
    version="0.1.0",
    author="Paolo Pignatelli",
    author_email="your.email@example.com",
    description="Knowledge Framework for tensor-based compression and symmetry detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PaoloTCS/claudelabs",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "networkx>=2.8.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "python-louvain>=0.16",
        "dataclasses>=0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=2.12.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.910",
        ],
    },
)