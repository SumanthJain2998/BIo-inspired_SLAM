SETUP_PY = """
from setuptools import setup, find_packages

setup(
    name="liquid-slam",
    version="0.1.0",
    description="Event-Based SLAM using Liquid Neural Networks",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.3.0", "black>=23.3.0", "flake8>=6.0.0"],
        "events": ["prophesee-metavision>=3.0.0", "dv-processing>=1.7.0"],
    },
    python_requires=">=3.8",
)
"""