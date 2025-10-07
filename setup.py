"""Setup script for installing the polarized disinformation package."""
from __future__ import annotations

from pathlib import Path

from setuptools import setup, find_packages


def read_readme() -> str:
    readme_path = Path(__file__).with_name("README.md")
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "Polarized Strategic Disinformation Diffusion Model"


setup(
    name="polarized-disinformation",
    version="0.1.0",
    description="Strategic disinformation diffusion model with Nash and social planner analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Polarized Disinformation Modeling Team",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "pyyaml>=6.0",
    ],
    extras_require={
        "tests": ["pytest>=8.0", "pytest-cov>=4.1"],
    },
)
