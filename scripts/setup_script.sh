#!/bin/sh
yes | sudo apt install python3-pip
pip install -U scikit-learn
pip install numpy pandas
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
export PATH=/home/ubuntu/.local/bin:$PATH
