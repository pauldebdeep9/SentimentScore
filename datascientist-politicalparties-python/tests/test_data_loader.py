import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
import sys
import os
# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from src.political_party_analysis.loader import DataLoader

@pytest.fixture
def mock_df() -> pd.DataFrame:
    data = {
        "id": [0, 1, 2, 2],
        "col1": [1, 2, 3, 3],
        "col2": [30.0, np.nan, 30.0, 30.0],
        "col3": [5, 7, 12, 12],
        "non_feature": ["a", "b", "c", "c"],
        "all_nans": [np.nan] * 4,
    }
    df = pd.DataFrame(data=data)
    return df


def test_download_data():
    data_loader = DataLoader()
    assert data_loader.party_data.shape == (277, 55)
    print('Downloaded data dimension is correct')


def test_preprocess_data(mocker, mock_df: pd.DataFrame):
    data_loader = DataLoader()
    mocker.patch.object(data_loader, "party_data", mock_df)
    mocker.patch.object(data_loader, "non_features", ["non_feature"])
    mocker.patch.object(data_loader, "index", "id")
    data_loader.preprocess_data()
    expected_df = pd.DataFrame(
        data={
            "col1": [-1.225, 0, 1.225],
            "col2": [0.0] * 3,
            "col3": [-1.019, -0.340, 1.359],
        },
        index=[0, 1, 2],
    )
    expected_df.index.name = "id"
    assert_frame_equal(data_loader.party_data, expected_df, rtol=3)

if __name__ == "__main__":
    test_download_data()
    print('This is working')