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
#     display_name: suite2p-wang-lab
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Alternative ROI detection algorithm
#
# This workflow explains an alternative ROI detection algorithm designed for dopamine sensors in hippocampus.
# It is a modification to the `sparsedetect` algorithm.
# Setting the parameters below in the `ops` file 

# %%
new_ops = {
    "sparse_mode": True, # alternative detection is implemented in sparsedetect.py
    "high_pass": 200, # temporal high pass filter width in frames
    "wang:high_pass_overlapping": True, # use overlapping for high pass filter
    "spatial_hp_detect": 15,
    "spatial_scale": 2,
    "max_iterations": 8, # in sparsedetect, max number of iterations is "max_iterations" * 250
    "wang:bin_size": 1, # force bin size to be 1 (see detect in detect.py)
    "wang:thresh_act_pix": .1, # select more active pixels during ROI extension (default is 0.2)
    "wang:thresh_peak_default": .08, # force peak threshold instead of deriving from spatial scale
    "wang:use_alt_norm": True, # If True, use alternative normalization for sparsedetect
    "wang:width_max": 30, # width of max filter in bins using during alternate normalization
    "wang:downsample_scale": 1, #If > 0, always use x downsampled data for peak detection
}

# %%
import numpy as np
ops = np.load(r'C:\temp\AC918-20231017_02\test_suite2p\suite2p\plane0\ops.npy', allow_pickle=True).item()
# ops["do_registration"] = True
# ops["align_by_chan"] = 2
ops["save_mat"] = True
ops['anatomical_only'] = 0
# ops['wang:save_roi_iterations'] = True
ops.update(new_ops)
from suite2p.wang_lab.wrappers import run_suite2p_no_gui

run_suite2p_no_gui(r'C:\temp\AC918-20231017_02\test_suite2p', ops)

# %%
