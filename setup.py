"""Setup script for BeliefDP package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="beliefdp",
    version="1.0.0",
    description="Subgoal Detection, Phase Segmentation, and Online Phase Inference for Robotics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="BeliefDP Research Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "h5py>=3.0.0",
        "pyyaml>=5.4.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
    ],
    extras_require={
        "online": [
            "pytorch-lightning>=1.5.0",
            "tensorboard>=2.8.0",
        ],
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "beliefdp-pipeline=scripts.run_pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
