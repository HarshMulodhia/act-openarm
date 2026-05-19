"""Setup for the openarm_act integration package."""

from setuptools import find_packages, setup

setup(
    name="openarm_act",
    version="0.1.0",
    description="ACT policy integration layer for OpenArm Isaac Lab tasks.",
    author="OpenArm Contributors",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "h5py",
        "numpy",
        "torch",
        "pyyaml",
        "matplotlib",
    ],
    extras_require={
        "dev": ["pytest"],
    },
)
