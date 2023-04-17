from setuptools import find_packages, setup

setup(
    name="Rec2Vec",
    version="0.0.1",
    description="Generic Rec2Vec, 2023SS",
    author="Maximilian Haller",
    author_email="maximilian.haller@tuwien.ac.at",
    license="MIT",
    install_requires=[
        "PyYAML==6.0",
        "pandas==2.0.0",
        "chardet==5.1.0",
        "tqdm==4.65.0"
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
