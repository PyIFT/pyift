#!/bin/bash

set -e

pip3 install pytest
pip3 install codecov
pip3 install pytest-cov
pip3 install numpy
pip3 install scipy

python3 setup.py install
