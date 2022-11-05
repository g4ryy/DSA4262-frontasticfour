# Introduction
This repository contains the files needed to recreate our group's project on m6A modification detection

# Getting Started
To run some training / inference with the model discussed in our report, follow these instructions : 

1. Start a **new** AWS ubuntu instance (provisioning a new instance avoids conflicts with previously set paths etc), ensure that it is at least a **large** instance type.
2. From the home directory, clone this repo : 
	`git clone https://github.com/g4ryy/DSA4262-frontasticfour.git`
3. Enter the `demo` folder within our repo : 
`cd DSA4262-frontasticfour/demo/`
4. Running the model training / inference requires some setup. We automate this using a shell script. 
	- To grant permissions for the script to run call : 
`chmod +x setup_script.sh`
	- To install all dependencies call : 
	 `source ./setup_script.sh`
	 This may take a few minutes 

## Running Inference
A sample dataset has been provided to run a small prediction / inference demo. From within `DSA4262-frontasticfour/demo/` do the following : 
1. Enter the `m6Anet` folder : 
	`cd m6Anet/`

2. To run the pre-trained model on the sample dataset, call : 
	`python3 run_inference.py ../inference_sample.json`

The resulting csv file with the m6A modification scores will be placed in :
	`DSA4262-frontasticfour/demo/inference_sample_results.csv`

Predictions can be made on any dataset (with the same format) by changing the given datafile path. Call `python3 run_inference.py -h` for more details on the required input arguments. 

## Running Training (Optional)
There is no sample dataset provided to run the training of the model as the training data & label files required are too large to store on github. However if these files have been stored outside the repo it is still easy to do the training : 

1. Ensure you are still in the `DSA4262-frontasticfour/demo/m6Anet` folder
2. Call `python3 run_learner.py <path to data.json file> <path to data.info file>`

The model training results such as the training loss & Validation loss at each epoch will be placed in new folder in the current directory. Call `python3 run_learner.py -h` for more details on required input arguments
