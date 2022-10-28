import argparse

parser = argparse.ArgumentParser(description='Run the training script for m6aNet')
parser.add_argument("ptd", type=str, help="Path to datafile (json format)")
parser.add_argument("ptl", type=str, help="Path to labels")
parser.add_argument("ptp", type=str, help="Path to parameters folder. Where to place model parameter files after each epoch")
parser.add_argument("ptvo", type=str, help="Where to output the validation dataset with attached probability scores (should be a csv file)")
parser.add_argument("pteo", type=str, help="Where to output the dataframe of epoch vs avg loss (should be a csv file)")

parser.add_argument("-nentries", type=int, help="How many entries from the training data to use. Set to 0 to use all data", default=5000)
parser.add_argument("-batchsize", type=int, help="Batchsize for training", default=128)
parser.add_argument("-lr", type=float, help="Learning Rate", default=0.01)
parser.add_argument("-nepoch", type=int, help="Number of Epochs", default=50)

args = parser.parse_args()
print(type(args.nentries))
