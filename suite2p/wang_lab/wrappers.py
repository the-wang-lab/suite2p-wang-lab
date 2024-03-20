from suite2p import version, run_s2p
from pathlib import Path
from contextlib import redirect_stdout    

def run_suite2p_no_gui(p_data, ops, save_folder='suite2p'):
    '''Run suite2p from jupyter notebook or script

    Note that `data_path` and `save_path0` will be set to `p_data`.

    Parameters
    ----------
    p_data : path-like
        Folder containing the tiff files
    ops : dict
        Settings for suite2p
    save_folder : path-like, optional
        Folder to save the output in p_data, by default 'suite2p'
    '''

    # data and output in same folder
    ops.update({
        'data_path': [ str(p_data) ],
        'save_path0': str(p_data),
        'save_folder': save_folder,
    })

    # save output to 'run.log' file (same output folder as defined in run_s2p)
    p_log = Path(p_data) / f'{save_folder}/run.log'
    print(f'INFO: Saving text output to {p_log}')

    with open(p_log, 'w') as f:
        with redirect_stdout(f):
            print(f'Running suite2p v{version} from jupyter notebook')
            run_s2p(ops)