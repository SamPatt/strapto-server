"""
Setup configuration for the StrapTo Local Server package.
"""

from setuptools import setup, find_packages

setup(
    name="strapto-server",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "aiortc",
        "aiohttp",
        "pytest",
        "pytest-asyncio",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="StrapTo Local Server for streaming AI model outputs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/strapto-server",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)