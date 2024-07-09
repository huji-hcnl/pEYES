# pEYES
## A python package for the eye tracking scientist
This package is intended for scientific use when processing and analyzing eye-tracking data.  
It provides easy-to-use functions for:
- Downloading & parsing publicly available annotated eye-tracking datasets.
- Configuring eye-tracking detection algorithms, and running them on the datasets or your own data.
- Analyzing the results of the detection algorithms, and comparing them to ground truth data or to a different detection algorithm.
- Visualizing the results of the detection algorithms.

## Installation Instructions
This package has been created and tested with python ```3.12```.  
The steps to install this package are -  

```angular2html
python -m venv env 
source ./env/bin/activate (In linux, instructions on other OS will vary)
pip install . (or pip install -e . for installing the editable version)
```

## Acknowledgements
We are grateful for the support of the [Center for Interdisciplinary Data Science Research (CIDR)](https://cidr.huji.ac.il/) at the [Hebrew University of Jerusalem](https://new.huji.ac.il/). In particular, we would like to thank Haimasree Bhattacharya from CIDR for her contributions.
