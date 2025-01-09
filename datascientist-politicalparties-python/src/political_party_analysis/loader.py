from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas as pd


class DataLoader:
    """Class to load the political parties dataset"""

    data_url: str = "https://www.chesdata.eu/s/CHES2019V3.dta"

    def __init__(self):
        self.party_data = self._download_data()
        self.non_features = []
        self.index = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        data_path = Path(__file__).parents[2].joinpath("data", "CHES2019V3.dta")
           
    # Check if the file already exists
        if not data_path.exists():
        # Download the file if it does not exist
            urlretrieve(self.data_url, data_path)
    
    # Load the data from the file
        return pd.read_stata(data_path)
        data_path, _ = urlretrieve(
            self.data_url,
            Path(__file__).parents[2].joinpath(*["data", "CHES2019V3.dta"]),
        )
        return pd.read_stata(data_path)

    def remove_duplicates(self, df_original: pd.DataFrame) -> pd.DataFrame:
        """Write a function to remove duplicates in a dataframe"""
        ##### YOUR CODE GOES HERE #####
        df= df_original.copy() 
        df = df.drop_duplicates()
    # print('Shape of df before and after removing duplicates: df_orginal.shape, df.shape')
        print(f"Shape of DataFrame before and after removing duplicates: {df_original.shape}, {df.shape}")
        return df

    def remove_nonfeature_cols(
        self, df_original: pd.DataFrame, non_features: List[str], index: List[str]
    ) -> pd.DataFrame:
        """Write a function to remove certain features cols and set certain cols as indices
        in a dataframe"""
        df= df_original.set_index(index)
        # df= df.drop(non_features, axis=1)
        if non_features is not None:
            df= df.drop(non_features, axis=1)
        # df_original.set_index(non_features, inplace=True)
        return df

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to handle NaN values in a dataframe"""
        ##### YOUR CODE GOES HERE #####
        df.fillna(df.mean(), inplace=True)
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to normalise values in a dataframe. Use StandardScaler."""
        ##### YOUR CODE GOES HERE #####
        pass

    def preprocess_data(self, df_original, non_features, index):
        """Write a function to combine all pre-processing steps for the dataset"""
        ##### YOUR CODE GOES HERE #####
        df= self.remove_duplicates(df_original)
        df= self.remove_nonfeature_cols(df, non_features= None, index= index)
        df= self.handle_NaN_values(df)
        return df
