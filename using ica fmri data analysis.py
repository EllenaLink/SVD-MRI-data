from sklearn.decomposition import FastICA

# ICA expects (n_timepoints, n_rois) — opposite of SVD
n_components = 20
ica = FastICA(n_components=n_components, random_state=42, max_iter=500)

sources = ica.fit_transform(time_series)
mixing = ica.mixing_

# Plot ICA spatial maps
for i in range(5):
    comp = mixing[:, i].reshape(1, -1)
    comp_img = masker.inverse_transform(comp)
    plotting.plot_stat_map(
        comp_img,
        title=f"ICA Network {i+1}",
        display_mode="ortho",
        threshold=0.01)

# Compare temporal profiles: SVD vs ICA
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(Vt[0, :], label='SVD Component 1')
axes[0].set_title('SVD Temporal Profile')
axes[0].set_xlabel('Timepoint')
axes[1].plot(sources[:, 0], label='ICA Component 1', color='orange')
axes[1].set_title('ICA Temporal Profile')
axes[1].set_xlabel('Timepoint')
plt.tight_layout()
plt.show()

plotting.show()
