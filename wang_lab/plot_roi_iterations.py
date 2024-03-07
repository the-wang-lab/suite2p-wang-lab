# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: suite2p_wang
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
from suite2p.wang_lab import utils as utl

p_out = Path('../data/test_output/sparsedetect')

# %% [markdown]
# This workflow allows us to investigate the intermediate steps of the ROI detection algorithm
# used when `sparse_mode` is set to `True`.
#
# # Running suite2p
# To generate the additional files needed,
# we will need to run suite2p setting `ops["wang:save_roi_iterations"] = True`. This will create additional files in the `'save_path0' directory.
#
# # Generate mean and max projection images
# The function `plot_processing_steps` plots the time-averaged and max-projection-along-time images at the
# individual preprocessing steps.
# This is useful for visualizing the effects of preprocessing parameters on the data.
#
# See `sparsery` function in [`sparsedetect.py`](../suite2p/detection/sparsedetect.py) for details
# on the steps.

# %%
utl.plot_processing_steps(p_out)
# to save plot to file, pass `path` argument

# %% [markdown]
# # Plot ROI masks
#
# The function `plot_roi_iter` visualizes the data the sparsery algorithm uses to initialize and refine ROI masks.
#
# The first row shows the "thresholed standard deviation" (see in `threshold_reduce` [`suite2p/detection/utils.py`](../suite2p/detection/utils.py))
# downsampled at various spatial scales.
# The maximum value is used to initialize a given ROI as a square.
#
# The second row shows how the ROI is refined by iteratively.
# Note that this is not the iteration in the `iter_extend` function in [`sparsedetect.py`](../suite2p/detection/sparsedetect.py),

# %%
utl.plot_roi_iter(p_out, n_roi=0)

# %% [markdown]
# To automatically generate plots for many ROIs, use this loop.

# %%
# total number of ROIs based on number of folders in `p_out`
n_roi_tot = len([ *p_out.glob('roi_*/') ])

for i in range(n_roi_tot):

    print(f'Plotting ROI {i}')
    utl.plot_roi_iter(p_out, n_roi=i, path=p_out / f'roi_{i:04d}/')

    if i > 10: # stop at some point
        break
