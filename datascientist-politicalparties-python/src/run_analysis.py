
from pathlib import Path
import pandas as pd
from matplotlib import pyplot

# import os
# Get the current directory of the script (run_analysis.py)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Navigate up two levels to the root directory
# root_dir = os.path.abspath(os.path.join(current_dir, ".."))
# # Construct the full path to the file
# data_file_path = os.path.join(root_dir, "data", "CHES2019V3.dta")
# print("Path to the data file:", data_file_path)

from political_party_analysis.loader import DataLoader
from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.visualization import scatter_plot

if __name__ == "__main__":

    data_loader = DataLoader()
    # Data pre-processing step
    ##### YOUR CODE GOES HERE #####
    df_original= data_loader._download_data()
    # df_original.set_index(["party_id", "party", "country"], inplace=True)
    # df= df_original.copy()
    # Drop specific columns
    # columns_to_drop = ["country", "party"]  # Replace with actual column names
    # df = df.drop(columns=columns_to_drop)
    # df = df.fillna(0)


    # df= data_loader.remove_nonfeature_cols(df_original, non_features= None, index= ["party_id", "party", "country"])
    # df= data_loader.remove_duplicates(df_original)
    # df= data_loader.handle_NaN_values(df)
    df= data_loader.preprocess_data(df_original,
                                    non_features= None,
                                    index= ["party_id", "party", "country"])
        
    # Dimensionality reduction step
    ##### YOUR CODE GOES HERE #####
    dimensionality_red= DimensionalityReducer(data= df, model= None, n_components= 2)
    reduced_data = dimensionality_red.reduce_to_2d()
    original_space= dimensionality_red.map_to_original_space(reduced_data)
    # print("Reduced Data:", reduced_data)

    ## Uncomment this snippet to plot dim reduced data
    # pyplot.figure()
    # splot = pyplot.subplot()
    # scatter_plot(
    #     reduced_dim_data,
    #     color="r",
    #     splot=splot,
    #     label="dim reduced data",
    # )
    # pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "dim_reduced_data.png"]))

    # Density estimation/distribution modelling step
    ##### YOUR CODE GOES HERE #####

    # Plot density estimation results here
    ##### YOUR CODE GOES HERE #####
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "density_estimation.png"]))

    # Plot left and right wing parties here
    pyplot.figure()
    splot = pyplot.subplot()
    ##### YOUR CODE GOES HERE #####
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "left_right_parties.png"]))
    pyplot.title("Lefty/righty parties")

    # Plot finnish parties here
    ##### YOUR CODE GOES HERE #####

    print("Analysis Complete")
