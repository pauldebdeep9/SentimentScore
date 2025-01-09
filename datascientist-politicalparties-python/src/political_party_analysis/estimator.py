import pandas as pd
# import os
# # Get the current directory of the script
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Set it as the working directory
# os.chdir(current_dir)
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
# from dim_reducer import DimensionalityReducer

class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    1. Write a function to model the distribution of the political party dataset
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher dimensional
    space as per the previously used dimensionality reduction technique.
    """
    # def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.kde_model = None
        self.feature_names = None

    ##### YOUR CODE GOES HERE #####
# Question 1: Write a function to model the distribution of the political party dataset
    def model_distribution(self, bandwidth=1.0):
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(self.data)
        self.kde_model = kde
# Question 2: Write a function to randomly sample 10 parties from this distribution
    def sample_parties(self, kde_model, n_samples=10):
       if self.kde_model is None:
            raise ValueError("KDE model not fitted. Call `model_distribution` first.")
       samples = self.kde_model.sample(n_samples)
       return samples
    #    return pd.DataFrame(samples, columns=self.data.columns)

# Question 3: Map the randomly sampled 10 parties back to the original higher dimensional space
    def map_to_original_space(reduced_samples, pca_model):
        if hasattr(pca_model, 'inverse_transform'):
            original_space = pca_model.inverse_transform(reduced_samples)
            return original_space
        else:
            raise ValueError("PCA model does not have the `inverse_transform` method.")

# Example Usage
# if __name__ == "__main__":
#     # Assuming you have the reduced data and PCA model from dimensionality reduction
#     reduced_data = np.random.rand(100, 2)  # Example reduced data (replace with actual data)
#     density_estimator= DensityEstimator(reduced_data)
#     # Step 1: Model the distribution
#     kde_model = density_estimator.model_distribution(reduced_data)
#     # Step 2: Randomly sample 10 parties
#     sampled_parties = density_estimator.sample_parties(kde_model, n_samples=10)
#     print("Sampled Parties (Reduced Space):\n", sampled_parties)
    
#     # Step 3: Map back to original space
#     # Assuming `pca_model` is the trained PCA model
#     pca_model = PCA(n_components=2)
#     pca_model.fit(np.random.rand(100, 10))  # Simulated PCA training (replace with actual model)
#     original_space_samples = density_estimator.map_to_original_space(sampled_parties, pca_model)
#     print("Mapped to Original Space:\n", original_space_samples)