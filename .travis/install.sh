#!/bin/bash

set -e

pip install pytest
pip install codecov
pip install pytest-cov
pip install numpy
pip install scipy

python setup.py install
