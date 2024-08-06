# pEYES
## A Python Package for Eye-Tracking Researchers
This package is intended for scientific use when processing and analyzing eye-tracking data.  
It provides easy-to-use functions for:
- Downloading & parsing publicly available annotated eye-tracking datasets.
- Configuring eye-tracking detection algorithms, and running them on the datasets or your own data.
- Analyzing the results of the detection algorithms, and comparing them to ground truth data or to a different detection algorithm.
- Visualizing the results of the detection algorithms.

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
