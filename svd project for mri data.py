# svd project for mri data

import numpy as np
import matplotlib.pyplot as plt

from nilearn.datasets import fetch_development_fmri
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from nilearn import plotting

data = fetch_development_fmri()

func_files = data.func
confounds = data.confounds
# load data
atlas = fetch_atlas_schaefer_2018(n_rois=100)

atlas_filename = atlas.maps
# extract time series
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

time_series = masker.fit_transform(func_files[0])
# transpose for SVD
X = time_series.T
# SVD
U, S, Vt = np.linalg.svd(X, full_matrices=False)
# plot
plt.plot(S, marker="o")
plt.xlabel("Component")
plt.ylabel("Singular Value")
plt.title("Singular Value Spectrum of Brain Activity")
plt.show()

# quantify variance
variance_explained = (S**2) / np.sum(S**2)

plt.plot(np.cumsum(variance_explained), marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("Variance Explained by SVD Components")
plt.show()

k = 50
X_reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
error = np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X)
print("Reconstruction error:", error)


# k = 50 gives Reconstruction error: 0.10818540121230935

# map the patterns
component = U[:, 0] * S[0]
component_img = masker.inverse_transform(component)

plotting.plot_stat_map(
    component_img,
    title="First SVD Brain Network",
    display_mode="ortho",
    threshold=0.01)

# plotting multiple networks
for i in range(5): # 5 most dominant characteristics
    comp = masker.inverse_transform(U[:, i])
    
    plotting.plot_stat_map(
        comp,
        title=f"SVD Network {i+1}",
        display_mode="ortho",
        threshold=0.01)

    
#Loop through components and plot weighted networks
for i in range(k):
    component = U[:, i] * S[i]               # weight by singular value
    component_img = masker.inverse_transform(component)  # map back to brain
    plotting.plot_stat_map(
        component_img,
        title=f"SVD Network {i+1} ({cumulative_variance[i]*100:.1f}% variance)",
        display_mode="ortho",
        threshold=0.01)

plotting.show()  