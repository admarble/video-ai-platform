from setuptools import setup, find_packages

setup(
    name="video-ai-platform",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "decord>=0.6.0",
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'sphinx>=4.0.0',
            'black>=21.0.0',
            'flake8>=3.9.0',
        ]
    },
    python_requires='>=3.8',
    author="Your Name",
    author_email="your.email@example.com",
    description="A platform for AI-powered video processing",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/video-ai-platform",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
) 