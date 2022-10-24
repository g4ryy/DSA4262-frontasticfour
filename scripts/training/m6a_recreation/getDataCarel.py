import json
import numpy as np
import pandas as pd


class getData :
    """
    A class used to obtain the the raw data and label data for the purposes of model training / evaluation

    Attributes
    ---------
    path_to_data : str
        The absolute path to where the data.json file (the raw data) is stored
    path_to_labels : str
        The absolute path to where the data.info file (the labels) is stored
    data : Dict
        The dictionary used to store the parsed data from the raw json file
    label_df : pandas.DataFrame
        DataFrame used to store the class labels parsed from the data.info file

    Methods
    ---------
    get_labels : pandas.DataFrame
        Parses the data.info file and returns the class labels formatted in a pandas DataFrame
    det_data : Dict or pandas.DataFrame
        Parses the data.json file and returns the parsed raw data in either a dictionary or pandas DataFrame format

    """
    def __init__(self,
                 path_to_data="/home/ubuntu/studies/ProjectStorage/data/data.json",
                 path_to_labels="/home/ubuntu/studies/ProjectStorage/data/data.info"):

        ### Raw Data ###
        self.path_to_data = path_to_data
        # Raw Data to return
        self.data = {
            "transcript": [],
            "position": [],
            "k-mer bases": [],
            "values": []
        }

        ### Labels ###
        self.path_to_labels=path_to_labels
        # Labels
        self.label_df = None

        ### Unique K-Mers ###
        self.k_mers = None

    def get_labels(self):
        """
        Get all the label info
        """

        # If the labels haven't yet been parsed
        if self.label_df is None:
            # Open the data.info file and read all the lines. Can do this because the file is relatively small
            with open(self.path_to_labels, "r") as file:
                lines = file.readlines()
            # Cleaning up of text
            df = list(map(lambda x: x.replace("\n", ""), lines))
            df = list(map(lambda x: x.split(","), df))
            # Convert to pandas DataFrame
            df = pd.DataFrame(df)
            # Tidying up column names
            new_header = df.iloc[0]  # grab the first row for the header
            df = df[1:]  # take the data less the header row
            df.columns = new_header  # set the header row as the df header
            # Change data type
            df['transcript_position'] = df['transcript_position'].astype(int)
            df['label'] = df['label'].astype(int)
            # Assign to self to store
            self.label_df = df
        return self.label_df

    def get_data(self, num_entries=10, return_df=True):
        """

        Parameters
        ----------
        num_entries=10 : int
            The number of datapoints in the raw data to read in. By default it is 10 for testing.
            SET TO 0 READ IN ALL THE DATA
        return_df=True : bool
            Whether or not to return the parsed data as a Pandas DataFrame. Otherwise returned as a dictionary

        Returns
        -------
        dict or DataFrame of parsed data
        """

        # Reset the self.data dictionary tracker
        self.data = {key : [] for key in self.data.keys()}
        # Open the data.json file
        with open(self.path_to_data, 'r') as f:
            # Check that the number of desired entries hasn't been reached
            for idx, line in enumerate(f):
                if num_entries != 0 and num_entries < idx+1:
                    break
                # Read in just that line. Cannot read in all lines at once as the data is too large
                line = json.loads(line)
                # Obtain transcript and sub-dictionary
                for transcript, sub1 in line.items():
                    # Add transcript to dictionary
                    self.data['transcript'].append(transcript)
                    # Obtain position number and sub-dict
                    for position, sub2 in sub1.items():
                        # Add position as an integer to the dictionary
                        self.data['position'].append(int(position))
                        # Obtain base and values
                        for bases, values in sub2.items():
                            # Add base to dictionary
                            self.data['k-mer bases'].append(bases)
                            # Convert the values to a numpy array and add to dictionary
                            values = np.array(values)
                            self.data['values'].append(values)

        # Return either the dictionary or pandas.Dataframe representation
        if not return_df:
            return self.data
        else :
            df = pd.DataFrame(self.data)
            return df

    def get_unique_kmers(self):
        """
        Returns a dictionary of 5-mer : unique_index for each of the 66 unique possible 5-mers for the
        m6A modification

        """
        if self.k_mers is None:
            data = set()
            with open(self.path_to_data, 'r') as f:
                # Check that the number of desired entries hasn't been reached
                for idx, line in enumerate(f):
                    # Read in just that line. Cannot read in all lines at once as the data is too large
                    line = json.loads(line)
                    # Obtain transcript and sub-dictionary
                    for transcript, sub1 in line.items():
                        for position, sub2 in sub1.items():
                            for bases, values in sub2.items():
                                indiv_bases = [bases[i:i + 5] for i in range(3)]
                                for tmp in indiv_bases:
                                    data.add(tmp)
            data = {item : idx for idx, item in enumerate(data)}
            self.k_mers = data
        return self.k_mers

