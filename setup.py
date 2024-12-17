from setuptools import setup, find_packages

setup(
    name="video-ai-platform",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "aiohttp>=3.8.0",
        "cv2-headless>=4.8.0",
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "torchvision>=0.16.0",
        "python-dotenv>=0.19.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.5.0",
        "pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "decord-gpu>=0.6.0",
        ],
        "cpu": [
            "decord>=0.6.0",
        ],
    },
    python_requires=">=3.8,<3.9",
) 