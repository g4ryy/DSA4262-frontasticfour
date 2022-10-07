#!/bin/sh
yes | sudo apt install python3-pip
pip install -U scikit-learn
pip install numpy pandas
export PATH=/home/ubuntu/.local/bin:$PATH
