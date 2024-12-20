from setuptools import setup, find_packages

setup(
    name="enhanced-config-manager",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0",
        "cryptography>=42.0.0",
        "watchdog>=3.0.0",
        "PyJWT>=2.8.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "pylint>=2.17.0"
        ]
    },
    python_requires=">=3.9",
    author="Your Name",
    description="Enhanced configuration management system with secrets, versioning, and dynamic updates",
    keywords="configuration, secrets, versioning, hot-reload",
) 