from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="automl-classification",
    version="0.1.0",
    author="Sercan Gul",
    author_email="sercan.gul@gmail.com",
    description="Automated Machine Learning Classification Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DataScientistTX/AutoMLClassification",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.2",
        "pandas>=1.1.3",
        "scikit-learn>=0.24.0",
        "streamlit>=0.73.1",
        "matplotlib>=3.3.2",
        "seaborn>=0.11.0",
        "XlsxWriter>=1.3.7",
    ],
    entry_points={
        "console_scripts": [
            "automl-classification=app.main:main",
        ],
    },
)