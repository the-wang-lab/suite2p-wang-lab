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

# %%
import suite2p
from suite2p.wang_lab.wrappers import run_suite2p_no_gui
from pathlib import Path
import numpy as np

# %% [markdown]
# # define ops dictionary

# %%
# load default ops or from file
ops = suite2p.default_ops()
# ops = np.load(r'C:\temp\AC918-20231017_02\test_suite2p\suite2p\plane0\ops.npy', allow_pickle=True).item()

# optional: change parameters
ops['spatial_scale'] = 2
ops['two_step_registration'] = 1

# %% [markdown]
# # run on single folder
# Note that in order to use motion-corrected data for ROI detection, suite2p requires 
# - `data.bin` and `ops.npy` in the `plane0` folder and
# - `refImg` in the `ops.npy` file

# %%
p_data = Path( r'C:\temp\AC918-20231017_02\test_suite2p')
run_suite2p_no_gui(p_data, ops)

# %% [markdown]
# # loop over folders

# %%
# option 1: manually
folders = [
    r'C:\temp\A214-20221222-02',
    r'C:\temp\A220-20230419-02',
]

# option 2: glob expression
folders = [ *Path(r'C:\temp').glob('AC9*/') ]

for p_data in folders:
    print(f'INFO: Running suite2p for {p_data}')
    run_suite2p_no_gui(p_data, ops)
