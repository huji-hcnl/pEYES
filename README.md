![peyes](https://github.com/user-attachments/assets/51d0138d-8e79-4530-96e9-1fce1393dcd3)
# pEYES
## A Python Package for Eye-Tracking Researchers

pEYES is a Python package that enables researchers to perform robust, quantitative comparisons of eye-movement (EM)
detection algorithms, i.e., algorithms that classify raw gaze samples into events such as fixations and saccades.
It provides implementations of several widely used algorithms and allows users to evaluate their performance against
ground-truth, human-annotated datasets. The package simplifies the process of selecting an optimal algorithm by
offering over 20 metrics to quantify performance, enhancing analysis reliability and reproducibility.
<br><br>
Using pEYES, [Nir & Deouell (2026)](https://doi.org/10.3758/s13428-026-02983-5) compared seven detection algorithms
against two human-annotated datasets and found that no single algorithm is universally optimal: performance varied
by dataset, metric, and event type, though adaptive-threshold algorithms (e.g., Engbert's) were consistently among
the top performers. For a detailed overview of the package's functionalities and the full comparison, please refer
to the publication.

## Overview
pEYES offers several core functionalities designed to facilitate the processing, analysis, and comparison of
eye-tracking data:
- **Downloading & Parsing Datasets**: Provides functions to easily download and parse publicly available,
human-annotated eye-tracking datasets, streamlining the setup process for benchmarking algorithms.
- **Configuring & Running Detection Algorithms**: Allows users to configure various eye-movement detection algorithms
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
```bash
pip install peyes
```
To install this package as a developer, clone the repository and install it in editable mode:
```bash
git clone https://github.com/huji-hcnl/pEYES.git
cd pEYES
python -m venv env
env\Scripts\activate  # on Windows; use `source env/bin/activate` on macOS/Linux
pip install -e .
```

Upgrading from 0.1.0? Read [CHANGELOG.md](CHANGELOG.md) first: 0.2.0 is a correctness release, and some
of its fixes change values that earlier versions returned.

## Usage
This package is intended for scientific use, and is designed to be easy to use for anyone with basic python knowledge.  
Most of the functions in this package are documented, and can be accessed by running:
```python
import peyes
help(peyes)
```
For more detailed information, please refer to the user tutorials provided in the `docs` directory of this repository.

## Citation & License
This package is distributed under the MIT License, but some of the datasets & detection algorithms that are implemented
in this package are distributed under different licenses. Please refer to the documentation of the specific dataset or
detection algorithm for more information.

If you use this package in your research, please cite [Nir & Deouell (2026)](https://doi.org/10.3758/s13428-026-02983-5):
```
@article{nir2026systematic,
  title={Systematic classification differences across eye movement detection algorithms},
  author={Nir, Jonathan and Deouell, Leon Y},
  journal={Behavior Research Methods},
  volume={58},
  number={4},
  pages={109},
  year={2026},
  publisher={Springer}
}
```

If you use a specific dataset or detection algorithm that is implemented in this package, please also cite the original
authors of that dataset or detection algorithm. The datasets' licenses and detectors' citations can be found in their
respective documentation (retrieved using the `dataset.load()` call).

## Acknowledgements
We are grateful for the support of the [Center for Interdisciplinary Data Science Research (CIDR)](https://cidr.huji.ac.il/) at the
[Hebrew University of Jerusalem](https://new.huji.ac.il/). In particular, we would like to thank Haimasree Bhattacharya from CIDR for
her assistance in publishing this package.

## Versioning
[Nir & Deouell (2026)](https://doi.org/10.3758/s13428-026-02983-5) is based on pEYES v0.1.0. The current release,
v0.2.0, includes bug fixes and efficiency improvements (see [CHANGELOG.md](CHANGELOG.md) for details) and does not
change the paper's conclusions.
