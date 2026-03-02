# using ica fmri data analysis

import numpy as np

import matplotlib.pyplot as plt

from nilearn.datasets import fetch_development_fmri, fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from nilearn import plotting

data = fetch_development_fmri()
func_files = data.func
confounds = data.confounds

atlas = fetch_atlas_schaefer_2018(n_rois=100)
atlas_filename = atlas.maps

masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

time_series = masker.fit_transform(func_files[0], confounds=confounds[0])

X = time_series.T  

U, S, Vt = np.linalg.svd(X, full_matrices=False)

plt.plot(S, marker="o")
plt.xlabel("Component")
plt.ylabel("Singular Value")
plt.title("Singular Value Spectrum of Brain Activity")
plt.show()


variance_explained = (S**2) / np.sum(S**2)
cumulative_variance = np.cumsum(variance_explained)  

plt.plot(cumulative_variance, marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("Variance Explained by SVD Components")
plt.show()

# reconstruction error at k=50 
k = 50
X_reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
error = np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X)
print(f"Reconstruction error (k={k}): {error:.5f}")

# top 5 unweighted networks 
for i in range(5):
    comp = U[:, i].reshape(1, -1)  
    comp_img = masker.inverse_transform(comp)
    plotting.plot_stat_map(
        comp_img,
        title=f"SVD Network {i+1}",
        display_mode="ortho",
        threshold=0.01)

#  k weighted networks 
for i in range(k):
    component = (U[:, i] * S[i]).reshape(1, -1)  
    component_img = masker.inverse_transform(component)
    plotting.plot_stat_map(
        component_img,
        title=f"SVD Network {i+1} ({cumulative_variance[i]*100:.1f}% variance)",  
        display_mode="ortho",
        threshold=0.01)

plotting.show()
