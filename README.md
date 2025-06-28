# pEYES
![peyes](https://github.com/user-attachments/assets/51d0138d-8e79-4530-96e9-1fce1393dcd3)
## A Python Package for Eye-Tracking Researchers

pEYES is a Python package that enables researchers to perform robust, quantitative comparisons of eye-tracking
segmentation algorithms. It provides implementations of several widely used algorithms and allows users to evaluate
their performance against ground-truth, human-annotated datasets. The package simplifies the process of selecting an
optimal algorithm by offering over 20 metrics to quantify performance, enhancing analysis reliability and reproducibility.

## Overview
pEYES offers several core functionalities designed to facilitate the processing, analysis, and comparison of
eye-tracking data:
- **Downloading & Parsing Datasets**: Provides functions to easily download and parse publicly available,
human-annotated eye-tracking datasets, streamlining the setup process for benchmarking algorithms.
- **Configuring & Running Detection Algorithms**: Allows users to configure various eye-tracking segmentation algorithms
and apply them to either the built-in datasets or their own custom data.
- **Algorithm Comparison & Analysis**: Offers tools to analyze the results of detection algorithms, compare their
performance against human-annotated ground-truth data, or evaluate differences between multiple algorithms.
- **Visualization Tools**: Includes visualization capabilities, such as generating fixation heatmaps and saccade
trajectories, to help users intuitively interpret the results of different detection algorithms.

This functionality makes pEYES a versatile tool for researchers aiming to enhance the accuracy and reliability of their
eye-tracking data analysis.


## Installation Instructions
This package has been created and tested with python ```3.12```.

To install this package as a user, use
```angular2html
pip install peyes
```
To install this package as a developer, clone the package from GitHub and run the following commands:
```angular2html
python -m venv env
git pull
pip install -e .
git checkout -b dev
```

## Usage
This package is intended for scientific use, and is designed to be easy to use for anyone with basic python knowledge.  
Most of the functions in this package are documented, and can be accessed by running:
```angular2html
import peyes
help(peyes)
```
For more detailed information, please refer to the user tutorials provided in the `docs` directory of this repository.

## Citation & License
This package is distributed under the MIT License, but some of the datasets & detection algorithms that are implemented
in this package are distributed under different licenses. Please refer to the documentation of the specific dataset or
detection algorithm for more information.

If you use this package in your research, please cite it as follows:
```angular2html
# TBD
```

If you use a specific dataset or detection algorithm that is implemented in this package, please also cite the original
authors of that dataset or detection algorithm. This information, like the license, can be found in the documentation
of the specific dataset or detection algorithm.

## Acknowledgements
We are grateful for the support of the [Center for Interdisciplinary Data Science Research (CIDR)](https://cidr.huji.ac.il/) at the [Hebrew University of Jerusalem](https://new.huji.ac.il/). In particular, we would like to thank Haimasree Bhattacharya from CIDR for her contributions.
