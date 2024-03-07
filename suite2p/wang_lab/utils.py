import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

def post_processing_suite2p_gui(img_orig):
    '''Applies similar post processing to what is done to images in suite2p gui

    Correlation map and max projection include additional post processing steps
    that depend on `ops['xrange']` and `ops['yrange']`. These are not included here.

    Parameters
    ----------
    img_orig : np.ndarray
        Original image

    Returns
    -------
    img_proc : np.ndarray
        Post-processed image
    '''

    # normalize to 1st and 99th percentile
    perc_low, perc_high = np.percentile(img_orig, [1, 99])
    img_proc = (img_orig - perc_low) / (perc_high - perc_low)
    img_proc = np.maximum(0, np.minimum(1, img_proc))

    # convert to uint8
    img_proc *= 255
    img_proc = img_proc.astype(np.uint8)

    return img_proc

def plot_processing_steps(p_out, cmap = 'viridis', path=''):

    fig, axmat = plt.subplots(ncols=5, nrows=2, figsize=(20, 8))
    fig.suptitle('mean images (top) and max projections (bottom)')

    for m, axarr in zip(['mean', 'max'], axmat):

        ax = axarr[0]
        ax.set_title('into sparsery')
        img = post_processing_suite2p_gui(np.load(p_out / f'{m}_img.npy'))
        ax.imshow(img, cmap=cmap)

        ax = axarr[1]
        ax.set_title('after temp HP')
        img = post_processing_suite2p_gui(np.load(p_out / f'{m}_img_hp.npy'))
        ax.imshow(img, cmap=cmap)

        if (p_out / f'img_hp_sd.npy').exists():
            ax = axarr[2]
            ax.set_title('standard deviation')
            img = post_processing_suite2p_gui(np.load(p_out / f'img_hp_sd.npy'))
            ax.imshow(img, cmap=cmap)

            ax = axarr[3]
            ax.set_title('norm by SD')
            img = post_processing_suite2p_gui(np.load(p_out / f'{m}_norm.npy'))
            ax.imshow(img, cmap=cmap)

            ax = axarr[4]
            ax.set_title('norm + spatial LP')
            img = post_processing_suite2p_gui(np.load(p_out / f'{m}_norm_lp.npy'))
            ax.imshow(img, cmap=cmap)
        
        else: # alternative detection
            ax = axarr[2]
            ax.set_title('after rolling max')
            img = post_processing_suite2p_gui(np.load(p_out / f'{m}_img_hp_rmax.npy'))
            ax.imshow(img, cmap=cmap)

            ax = axarr[3]
            ax.set_title('after neuropil subtraction')
            img = post_processing_suite2p_gui(np.load(p_out / f'{m}_lp.npy'))
            ax.imshow(img, cmap=cmap)

    for ax in axmat.flatten():
        ax.axis('off')

    fig.tight_layout()

    if path:
        fig.savefig(path)
        plt.close(fig)
        

def plot_roi_iter(p_out, n_roi, zoom_border=10, color='C3', cmap1='viridis', cmap2='gray', path=''):

    # downsampled standard deviation images with max values
    fig, axmat = plt.subplots(ncols=5, nrows=2, figsize=(15, 7))
    norm = lambda x: (x - min(x.min(), 0)) / x.max()

    for i, ax in enumerate(axmat[0]):
        img = np.load(p_out / f'roi_{n_roi}/sd_down_{i}.npy')
        ax.imshow(img, cmap=cmap1)
        y, x = np.unravel_index(np.argmax(img), img.shape)
        ax.scatter(x, y, color='r', marker='x')
        ax.set_title(f'SD x{i} | max={img.max():1.1f}')

    # ROI iterations on top of mean image
    ax = axmat[1, 0]
    img = post_processing_suite2p_gui(np.load(p_out / 'mean_img.npy'))
    for ax in axmat[1]:
        ax.imshow(img, cmap=cmap2)

    # define RGB plane with alpha channel = 0
    c_rgb = to_rgba(color)
    plane = np.tile(c_rgb, (*img.shape, 1))
    plane[:, :, -1] = 0

    for i in range(4):

        # load ROI data for given iteration
        xpix = np.load(p_out / f'roi_{n_roi}/xpix_{i}.npy')
        ypix = np.load(p_out / f'roi_{n_roi}/ypix_{i}.npy')
        lam = np.load(p_out / f'roi_{n_roi}/lam_{i}.npy') 

        if i == 0:
            # zoom limits for ROI
            xlims = xpix.min() - zoom_border, xpix.max() + zoom_border
            ylims = ypix.min() - zoom_border, ypix.max() + zoom_border     

        # set alpha channel to lam
        r = plane.copy()
        r[ypix, xpix, -1] = norm(lam)

        # plot ROI in zoomed image
        ax = axmat[1, i]
        ax.set_title(f'lam: iter {i}')
        ax.imshow(r)
        
    # plot lam * SD (lam in stat.npy is lam*SD) 
    ax = axmat[1, -1]
    r = plane.copy()
    p_sd = p_out / 'img_hp_sd.npy'
    if p_sd.exists():
        ax.set_title('lam * SD')
        img_sd = np.load(p_sd)
        r[ypix, xpix, -1] = norm(lam * img_sd[ypix, xpix])
    else: # alternative detection
        ax.set_title('lam')
        r[ypix, xpix, -1] = norm(lam)
    ax.imshow(r)

    # zoom in on all but first
    for ax in axmat[1, 1:].flatten():
        ax.set_xlim(xlims)
        ax.set_ylim(ylims[::-1]) # y is down in images

    for ax in axmat.flatten(): # remove axes
        ax.axis('off')

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
