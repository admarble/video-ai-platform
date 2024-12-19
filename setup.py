from setuptools import setup, find_packages

setup(
    name="video-ai-platform",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pytest",
        "pytest-benchmark",
        "psutil",
        "pydantic",
    ],
    python_requires=">=3.8",
) 