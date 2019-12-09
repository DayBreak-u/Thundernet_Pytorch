#!/usr/bin/env bash

rm -rf ./build/ ./model/_C*
python setup.py build_ext --inplace