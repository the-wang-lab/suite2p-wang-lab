"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from typing import Tuple, Dict, List, Any
from warnings import warn

import numpy as np
from numpy.linalg import norm

from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.stats import mode as most_common_value

from pathlib import Path

from . import utils

from suite2p.extraction import masks
from suite2p.detection.stats import ROI



def neuropil_subtraction(mov: np.ndarray, filter_size: int) -> None:
    '''Apply spatial low-pass filter to help ignore neuropil
    
    The uniform filter of size "filter_size" is applied to each frame
    and divided by the a 2D plane of ones with feathered edges.
    This is then subtracted from the original frame. 
    

    Parameters
    ----------
    mov : np.ndarray
        Input movie of shape (n_bins, y, x)
    filter_size : int
        Size of filter size for uniform_filter in pixel

    Returns
    -------
    mov_out : np.ndarray
        Low-pass filtered movie of shape (n_bins, y, x)
    '''
    # plane with feathered edges
    _, Ly, Lx = mov.shape
    c1 = uniform_filter(np.ones((Ly, Lx)), size=filter_size, mode="constant")

    mov_out = np.zeros_like(mov)
    for frame_old, frame_new in zip(mov, mov_out):
        frame_filt = uniform_filter(frame_old, size=filter_size, mode="constant") / c1
        frame_new[:] = frame_old - frame_filt
    return mov_out


def square_convolution_2d(mov: np.ndarray, filter_size: int) -> np.ndarray:
    '''Returns movie convolved by uniform kernel with width "filter_size".

    Parameters
    ----------
    mov : np.ndarray
        Input movie of shape (n_bins, y, x)
    filter_size : int
        Size of filter size for uniform_filter in pixel

    Returns
    -------
    mov_out : np.ndarray
        Convolved movie of shape (n_bins, y, x)
    '''
    mov_out = np.zeros_like(mov, dtype=np.float32)
    for frame_old, frame_new in zip(mov, mov_out):
        frame_filt = filter_size * \
            uniform_filter(frame_old, size=filter_size, mode="constant")
        frame_new[:] = frame_filt
    return mov_out


def multiscale_mask(ypix, xpix, lam, Ly_down, Lx_down):
    '''Downsample masks across spatial scales

    Parameters
    ----------
    ypix : np.ndarray
        1D array of y pixel indices
    xpix : np.ndarray
        1D array of x pixel indices
    lam : np.ndarray
        1D array of pixel weights
    Ly_down : list
        List of y dimensions at each spatial scale
    Lx_down : list
        List of x dimensions at each spatial scale

    Returns
    -------
    ypix_down : list
        List of downsampled y pixel indices at each spatial scale
    xpix_down : list
        List of downsampled x pixel indices at each spatial scale
    lam_down : list
        List of downsampled pixel weights at each spatial scale
    '''
    # initialize at original scale
    xpix_down = [xpix]
    ypix_down = [ypix]
    lam_down = [lam]
    for j in range(1, len(Ly_down)):
        ipix, ind = np.unique(
            np.int32(xpix_down[j - 1] / 2) + np.int32(ypix_down[j - 1] / 2) * Lx_down[j],
            return_inverse=True,
        )
        lam_d = np.zeros(len(ipix))
        for i in range(len(xpix_down[j - 1])):
            lam_d[ind[i]] += lam_down[j - 1][i] / 2
        lam_down.append(lam_d)
        ypix_down.append(np.int32(ipix / Lx_down[j]))
        xpix_down.append(np.int32(ipix % Lx_down[j]))
    for j in range(len(Ly_down)):
        ypix_down[j], xpix_down[j], lam_down[j] = extend_mask(
            ypix_down[j], xpix_down[j], lam_down[j], Ly_down[j], Lx_down[j])
    return ypix_down, xpix_down, lam_down


def add_square(yi, xi, lx, Ly, Lx):
    """return square of pixels around peak with norm 1

    Parameters
    ----------------

    yi : int
        y-center

    xi : int
        x-center

    lx : int
        x-width

    Ly : int
        full y frame

    Lx : int
        full x frame

    Returns
    ----------------

    y0 : array
        pixels in y

    x0 : array
        pixels in x

    mask : array
        pixel weightings

    """
    lhf = int((lx - 1) / 2)
    ipix = np.tile(np.arange(-lhf, -lhf + lx, dtype=np.int32), reps=(lx, 1))
    x0 = xi + ipix
    y0 = yi + ipix.T
    mask = np.ones_like(ipix, dtype=np.float32)
    ix = np.all((y0 >= 0, y0 < Ly, x0 >= 0, x0 < Lx), axis=0)
    x0 = x0[ix]
    y0 = y0[ix]
    mask = mask[ix]
    mask = mask / norm(mask)
    return y0, x0, mask


def iter_extend(ypix, xpix, mov, Lyc, Lxc, active_frames, thresh_active):
    """extend mask based on activity of pixels on active frames
    ACTIVE frames determined by threshold

    Parameters
    ----------------

    ypix : array
        pixels in y

    xpix : array
        pixels in x

    mov : 2D array
        binned residual movie [nbinned x Lyc*Lxc]

    active_frames : 1D array
        list of active frames

    Returns
    ----------------
    ypix : array
        extended pixels in y

    xpix : array
        extended pixels in x
    lam : array
        pixel weighting
    """

    # only active frames
    mov_act = mov[active_frames]

    while True:
        npix_old = ypix.size  # roi size before extension

        # extend by 1 pixel on each side
        ypix, xpix = extendROI(ypix, xpix, Lyc, Lxc, 1)

        # mean activity in roi
        roi_act = mov_act[:, ypix * Lxc + xpix]
        lam = roi_act.mean(axis=0)

        # select active pixels
        thresh_lam = max(0, lam.max()) * thresh_active
        pix_act = lam > thresh_lam

        if not np.any(pix_act):  # stop if no pixels are active
            break

        ypix, xpix, lam = ypix[pix_act], xpix[pix_act], lam[pix_act]
        npix_new = ypix.size  # after extension

        if npix_new <= npix_old:  # stop if no pixels were added
            break

        if npix_new >= 10000:  # stop if too many pixels
            break

    # normalize by standard deviation
    lam = lam / np.sum(lam**2) ** 0.5

    return ypix, xpix, lam


def extendROI(ypix, xpix, Ly, Lx, niter=1):
    '''Extend ypix and xpix by `niter` pixel(s) on each side

    Parameters
    ----------
    ypix : np.ndarray
        1D array of y pixel indices
    xpix : np.ndarray
        1D array of x pixel indices
    Ly : int
        y dimension of movie
    Lx : int
        x dimension of movie
    niter : int, optional
        Number of iterations to extend, by default 1

    Returns
    -------
    ypix : np.ndarray
        1D array of extended y pixel indices
    xpix : np.ndarray
        1D array of extended x pixel indices
    '''
    for _ in range(niter):
        yx = (
            (ypix, ypix, ypix, ypix - 1, ypix + 1),
            (xpix, xpix + 1, xpix - 1, xpix, xpix),
        )
        yx = np.array(yx)
        yx = yx.reshape((2, -1))
        yu = np.unique(yx, axis=1)
        ix = np.all((yu[0] >= 0, yu[0] < Ly, yu[1] >= 0, yu[1] < Lx), axis=0)
        ypix, xpix = yu[:, ix]
    return ypix, xpix


def two_comps(mpix0, lam, Th2):
    """check if splitting ROI increases variance explained

    Parameters
    ----------------

    mpix0 : 2D array
        binned movie for pixels in ROI [nbinned x npix]

    lam : array
        pixel weighting

    Th2 : float
        intensity threshold


    Returns
    ----------------

    vrat : array
        extended pixels in y

    ipick : tuple
        new ROI

    """
    # TODO add comments
    mpix = mpix0.copy()
    xproj = mpix @ lam
    gf0 = xproj > Th2

    mpix[gf0, :] -= np.outer(xproj[gf0], lam)
    vexp0 = np.sum(mpix0**2) - np.sum(mpix**2)

    k = np.argmax(np.sum(mpix * np.float32(mpix > 0), axis=1))
    mu = [lam * np.float32(mpix[k] < 0), lam * np.float32(mpix[k] > 0)]

    mpix = mpix0.copy()
    goodframe = []
    xproj = []
    for mu0 in mu:
        mu0[:] /= norm(mu0) + 1e-6
        xp = mpix @ mu0
        mpix[gf0, :] -= np.outer(xp[gf0], mu0)
        goodframe.append(gf0)
        xproj.append(xp[gf0])

    flag = [False, False]
    V = np.zeros(2)
    for _ in range(3):
        for k in range(2):
            if flag[k]:
                continue
            mpix[goodframe[k], :] += np.outer(xproj[k], mu[k])
            xp = mpix @ mu[k]
            goodframe[k] = xp > Th2
            V[k] = np.sum(xp**2)
            if np.sum(goodframe[k]) == 0:
                flag[k] = True
                V[k] = -1
                continue
            xproj[k] = xp[goodframe[k]]
            mu[k] = np.mean(mpix[goodframe[k], :] *
                            xproj[k][:, np.newaxis], axis=0)
            mu[k][mu[k] < 0] = 0
            mu[k] /= 1e-6 + np.sum(mu[k] ** 2) ** 0.5
            mpix[goodframe[k], :] -= np.outer(xproj[k], mu[k])
    k = np.argmax(V)
    vexp = np.sum(mpix0**2) - np.sum(mpix**2)
    vrat = vexp / vexp0
    return vrat, (mu[k], xproj[k], goodframe[k])


def extend_mask(ypix, xpix, lam, Ly, Lx):
    """extend mask into 8 surrrounding pixels"""
    # TODO add docstring and comments
    nel = len(xpix)
    yx = (
        (ypix, ypix, ypix, ypix - 1, ypix - 1,
         ypix - 1, ypix + 1, ypix + 1, ypix + 1),
        (xpix, xpix + 1, xpix - 1, xpix, xpix +
         1, xpix - 1, xpix, xpix + 1, xpix - 1),
    )
    yx = np.array(yx)
    yx = yx.reshape((2, -1))
    yu, ind = np.unique(yx, axis=1, return_inverse=True)
    LAM = np.zeros(yu.shape[1])
    for j in range(len(ind)):
        LAM[ind[j]] += lam[j % nel] / 3
    ix = np.all((yu[0] >= 0, yu[0] < Ly, yu[1] >= 0, yu[1] < Lx), axis=0)
    ypix1, xpix1 = yu[:, ix]
    lam1 = LAM[ix]
    return ypix1, xpix1, lam1



def estimate_spatial_scale(I: np.ndarray) -> int:
    '''Estimate spatial scale based on max projection

    Parameters
    ----------
    I : np.ndarray
        Array with upsampled max proj images across time of shape (n_scales, y, x)

    Returns
    -------
    im : int
        Best spatial scale
    '''
    # TODO add comments
    I0 = I.max(axis=0)
    imap = np.argmax(I, axis=0).flatten()
    ipk = np.abs(I0 - maximum_filter(I0, size=(11, 11))).flatten() < 1e-4
    isort = np.argsort(I0.flatten()[ipk])[::-1]
    im, _ = most_common_value(imap[ipk][isort[:50]], keepdims=False)
    return im


def find_best_scale(maxproj_splined: np.ndarray, spatial_scale: int, max_scale=4) -> Tuple[int, str]:
    '''Find best spatial

    Returns best scale (between 1 and `max_scale`) 
    and estimation method ("FORCED" or "estimated").

    Parameters
    ----------
    maxproj_splined : np.ndarray
        Array with upsampled max proj images across time of shape (n_scales, y, x)
    spatial_scale : int
        If > 0, use this as the spatial scale, otherwise estimate it

    Returns
    -------
    scale : int
        Best spatial scale
    mode : str
        Estimation mode
    '''
    modes = { # to mirror former Enum class
        'frc': "FORCED",
        'est': "estimated",
    }
    if spatial_scale > 0:
        scale = max(1, min(max_scale, spatial_scale))
        mode = modes['frc']
    else:
        scale = estimate_spatial_scale(maxproj_splined)
        mode = modes['est']

    if not scale > 0:
        warn(
            "Spatial scale estimation failed.  Setting spatial scale to 1 in order to continue."
        )
        scale = 1
        mode = modes['frc']

    return scale, mode


def spatially_downsample(mov: np.ndarray, n_scales: int, filter_size=3) -> np.ndarray:
    '''Downsample movie at multiple spatial scales

    Spatially downsample the movie `n_scales` times by a factor of 2.
    Applies smoothing with 2D uniform filter of size `filter_size` before
    each downsampling step.
    Also returns meshgrid of downsampled grid points.

    Parameters
    ----------
    mov : np.ndarray
        Input movie of shape (n_bins, y, x)
    n_scales : int
        Number of times to downsample

    Returns
    -------
    mov_down : list
        List of downsampled movies
    grid_down : list
        List of downsampled grid points
    '''
    _, Lyc, Lxc = mov.shape
    grid_list = np.meshgrid(range(Lxc), range(Lyc))
    grid = np.array(grid_list).astype("float32")

    # variables to be downsampled
    mov_d, grid_d = mov, grid

    # collect downsampled movies and grids
    mov_down, grid_down = [], []

    # downsample multiple times
    for _ in range(n_scales):
        # smooth (downsampled) movie
        smoothed = square_convolution_2d(mov_d, filter_size=filter_size)
        mov_down.append(smoothed)

        # downsample movie TODO why x2?
        mov_d = 2 * utils.downsample(mov_d, taper_edge=True)

        # downsample grid
        grid_down.append(grid_d)
        grid_d = utils.downsample(grid_d, taper_edge=False)

    return mov_down, grid_down


def spline_over_scales(mov_down, grid_down):
    '''Spline approximation of max projection of downsampled movies

    Uses RectBivariateSpline to upsample the downsampled max projection
    across time at each spatial scale.

    Parameters
    ----------
    mov_down : list
        List of downsampled movies
    grid_down : list
        List of downsampled grid points

    Returns
    -------
    img_up : np.ndarray
        Array with upsampled max proj images across time of shape (n_scales, y, x)
    '''

    grid = grid_down[0]

    img_up = []
    for mov_d, grid_d in zip(mov_down, grid_down):
        img_d = mov_d.max(axis=0)
        upsample_model = RectBivariateSpline(
            x=grid_d[1, :, 0],
            y=grid_d[0, 0, :],
            z=img_d,
            kx=min(3, grid_d.shape[1] - 1),
            ky=min(3, grid_d.shape[2] - 1),
        )
        up = upsample_model(grid[1, :, 0], grid[0, 0, :])
        img_up.append(up)

    img_up = np.array(img_up)

    return img_up


def scale_in_pixel(scale):
    "Convert scale integer to number of pixels"
    return int(3 * 2**scale)

def set_scale_and_thresholds(mov_norm_down, grid_down, spatial_scale, threshold_scaling):
    '''Find best spatial scale and set thresholds for ROI detection

    Parameters
    ----------
    mov_norm_down : list
        List of downsampled movies
    grid_down : list
        List of downsampled grid points
    spatial_scale : int
        If > 0, use this as the spatial scale, otherwise estimate it
    threshold_scaling : float
        

    Returns
    -------
    scale_pix : int
        Spatial scale in pixels
    thresh_peak : float
        Threshold: `threshold_scaling` * 5 * `scale`
    thresh_multiplier : float
        Threshold multiplier: max(1, n_bins / 1200)
    vcorr : np.ndarray
        Correlation map
    '''
    
    # spline approximation of max projection of downsampled movies
    maxproj_splined = spline_over_scales(mov_norm_down, grid_down)
    corr_map = maxproj_splined.max(axis=0)

    scale, estimate_mode = find_best_scale(maxproj_splined, spatial_scale=spatial_scale)
    # TODO: scales from cellpose (?)
    #    scales = 3 * 2 ** np.arange(5.0)
    #    scale = np.argmin(np.abs(scales - diam))


    # define thresholds based on spatial scale
    scale_pix = scale_in_pixel(scale)
    # threshold for accepted peaks (scale it by spatial scale) TODO why hardcode 5
    thresh_peak = threshold_scaling * 5 * scale
    # TODO why hardcode 1200
    thresh_multiplier = max(1, mov_norm_down[0].shape[0] / 1200)
    print(
        "NOTE: %s spatial scale ~%d pixels, time epochs %2.2f, threshold %2.2f "
        % (
            estimate_mode,
            scale_pix,
            thresh_multiplier,
            thresh_multiplier * thresh_peak,
        )
    )

    return scale_pix, thresh_peak, thresh_multiplier, corr_map


def sparsery(
    mov: np.ndarray,
    high_pass: int,
    neuropil_high_pass: int,
    batch_size: int,
    spatial_scale: int,
    threshold_scaling: float,
    max_iterations: int,
    use_overlapping: bool,
    use_alt_norm: bool,
    thresh_peak_default,
    thresh_act_pix: float,
    width: int,
    downsample_scale,
    rolling: str,
    neuropil_lam: bool,
    norm: str,
    lam_percentile: float,
    inner_neuropil_radius: int,
    min_neuropil_pixels: int,
    circular: bool,
    aspect: bool,
    diameter: int,
    do_crop: bool,   
    percentile=0., # TODO add to documentation (docs/settings.rst)
    n_scales=5,
    n_iter_refine=3,
    thresh_split=1.25,
    # extract_patches=False, TODO patches and seeds are unused
    save_path="",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Algorithm for ROI detection if `sparse_mode` is True
    
    Defined here
    ------------

    Statistics for each ROI:
    - ypix: y pixel indices
    - xpix: x pixel indices
    - lam: thresholded mean pixel intensity times pixel intensity standard deviation
    - med: location of peak in downsampled standard deviation map
    - footprint: spatial scale in which the ROI exhibits the largest standard deviation
    
    Items to be added to the ops dict:
    - Vcorr: max projection across spatial scales (see spline_over_scales function)
    - spatial_scale: best spatial scale in pixels
    - Vmap: standard deviation map across spatial scales
    - Vmax: max value of standard deviation map for each ROI
    - ihop: spatial scale in which the ROI exhibits the largest standard deviation
    - Vsplit: ratio of variance explained by splitting ROI into two components


    Outline of algorithm
    --------------------
    TODO make sparsery part of the docs/celldetection.rst
    
    Preprocessing
    - high-pass filter movie with kernel width `high_pass` (uniform or Gaussian)
    - normalize movie by standard deviation across time
    - low-pass filter movie with kernel width `neuropil_high_pass` (uniform)
    - downsample movie at `n_scales` spatial scales

    Spatial scale
    - for automatic detection (if `spatial_scale` == 0):
        - max project across time for each spatial scale
        - spline approximation back to original resolution
        - estimate spatial scale based on `find_best_scale`
    - if `spatial_scale` > 0: use `spatial_scale` as the best spatial scale

    ROI detection
    - initialization
        - calculate standard deviation of pixels across time relative to 0 while
        filtering out pixels values < `threshold_scaling` * 5 * `spatial_scale`
        - find peaks in downsampled standard deviations: maximum is defined by (scale, x, y)
        - place square around peak as initial ROI (size of square is 3 * 2**scale pixels )
        - average frames across square, define active frames based on threshold
    - refine `n_iter_refine` times:
        - extend ROI by 1 pixel on each side until
        - caclulate average correlation between pixel intensities and ROIs 
        - mean across only active frames, select pixels based on 1/5 of max value
        - repeat until
            (i) no pixels are active
            (ii) no pixels were added (only number, not identity)
            (iii) ROI becomes too large (10,000 pixels)
    - split ROI if ratio of variance explained after and before splitting is > `thresh_split`
    - subtract ROI from data
    - repeat until
        (i) no active frames detected
        (ii) maximum value in downsampled standard deviation map < thresh_multiplier * thresh_peak
        (iii) `max_iterations` iterations reached

    Parameters
    ----------
    mov : np.ndarray
        Input movie of shape (n_bins, y, x)
    high_pass : int
        Width of temporal high-pass filter in bins
    neuropil_high_pass : int
        Width of spatial low-pass filter in pixels
    batch_size : int    
        Frames in each bin
    spatial_scale : int
        If > 0, use this as the spatial scale, otherwise estimate it
    threshold_scaling : float
        Threshold scaling factor, higher values lead to more ROIs detected
    max_iterations : int
        Maximum number of ROIs detected
    use_overlapping : bool
        If True, use overlapping instead of non-overlapping rolling mean for high-pass filter
        Behavior depends on `high_pass` value, see utils.temporal_high_pass_filter for details
    use_alt_norm : bool
        If True, use alternative normalization method (see code for detail)
    thresh_peak_default : float or None
        If not None, use this as the threshold for `thresh_peak` instead of the value calculated in `set_scale_and_thresholds`
    thresh_act_pix : float
        Threshold for active pixels as fraction of max lam value
    width : int
        Width of max or mean filter in bins, only used if `use_alt_norm` is True
    downsample_scale : int or None
        If not None, always look for ROI peaks at this spatial scale (0 is the original scale)
    percentile : float, optional
        If > 0, use percentile as dynamic threshold for active frames, by default 0.
    n_scales : int, optional
        Number of spatial scales to downsample, by default 5
    n_iter_refine : int, optional
        Number of iterations to extend ROI, by default 3
    thresh_split : float, optional
        Threshold for variance explained ratio to split ROI, by default 1.25
    save_path : str, optional
        If not empty, save masks and quality control images used during ROI iterations, by default ""
    rolling : str
        Choose which method to use for rolling binning, 'max' for max projection, 'mean' for mean projection
    norm : str 
        Choose which method to use for movie normalization, if 'max', movie is normalized by max projection; if 'max-min', mov_norm = (mov-mov.min)/(mov.max-mov-min)
    
        ----------following parameters are used for generating neuropil mask and calculate neuropil lam---------

    neuropil_lam : bool
        If true, in sparsedetect generate neuropil mask and calculate neuropil lam
    lam_percentile: int, default: 50
        Percentile of Lambda within area to ignore when excluding cell pixels for neuropil extraction
    inner_neuropil_radius : int, default: 2
        Number of pixels to keep between ROI and neuropil donut
    min_neuropil_pixels : int, default: 350
        Minimum number of pixels used to compute neuropil for each cell
    circular : bool, default: Flase
        Whetehr to expend neuropil in circular manner
    aspect : float, default: 1.0
        Ratio of um/pixels in X to um/pixels in Y (ONLY for correct aspect ratio in GUI, not used for other processing)
    diameter : int, default: 0
        Diameter that will be used for cellpose. If set to zero, diameter is estimated.
    do_crop = bool, default: True
        Specifies whether to crop dendrites for cell classification stats

    Returns
    -------
    new_ops : dict
        Dictionary with new items to be added to ops
    stats : list
        List of dictionaries with statistics for each ROI        
    """
    new_ops = {}  # initialize new ops

    if save_path:
        # create output folder
        p_out = Path(save_path) / 'sparsedetect'
        p_out.mkdir(exist_ok=True, parents=True)
        print(f'INFO saving ROI iterations to {p_out}')

        def save_array(name, arr):
            p = p_out / name
            p.parent.mkdir(exist_ok=True, parents=True)
            return np.save(p, arr)
    else:
        def save_array(name, arr):
            pass

    ###############
    # Preprocessing
    # save_array("mov_intosparsery.npy", mov)
    
    save_array("mean_img.npy", mov.mean(axis=0))
    save_array("max_img.npy", mov.max(axis=0))

    # high-pass filter movie
    mov = utils.temporal_high_pass_filter(mov=mov, width=int(high_pass), use_overlapping=use_overlapping)
    new_ops["max_proj"] = mov.max(axis=0)
    save_array("mean_img_hp.npy", mov.mean(axis=0))
    save_array("max_img_hp.npy", mov.max(axis=0))


    if use_alt_norm:
        # rolling max filter:
        if rolling == 'max':
            mov = utils.max_filter(mov=mov, width=int(width))
        if rolling == 'mean':
            mov = utils.mean_filter(mov=mov, width=int(width))
        # save_array("rolling_bin.npy", mov)
        save_array("mean_img_hp_rmax.npy", mov.mean(axis=0))
        save_array("max_img_hp_rmax.npy", mov.max(axis=0))

        # subtract spatially low-pass per frame
        mov = neuropil_subtraction(
            mov=mov, filter_size=neuropil_high_pass)
        save_array("mean_lp.npy", mov.mean(axis=0))
        save_array("max_lp.npy", mov.max(axis=0))

        # normalization 
        if norm == 'max':
            # Find the maximum pixel intensity across the entire image stack
            mov_max_intensity = np.max(mov)
            # Normalize each image by the maximum intensity
            mov_norm = mov / mov_max_intensity
        elif norm == 'max-min':
            mov_norm = (mov-np.min(mov))/(np.max(mov)-np.min(mov)) #new normalization - Jingyu 3/28/2024
        
        # save_array("mov_max_intensity", mov_max_intensity)
        # save_array("mov_norm.npy", mov_norm)
        save_array("mean_norm.npy", mov_norm.mean(axis=0))
        save_array("max_norm.npy", mov_norm.max(axis=0))
                    
    else:
        # normalize by standard deviation
        mov_sd = utils.standard_deviation_over_time(mov, batch_size=batch_size)
        mov_norm = mov / mov_sd
        save_array("img_hp_sd.npy", mov_sd)
        save_array("mean_norm.npy", mov_norm.mean(axis=0))
        save_array("max_norm.npy", mov_norm.max(axis=0))

        # subtract spatially low-pass per frame
        mov_norm = neuropil_subtraction(
            mov=mov_norm, filter_size=neuropil_high_pass)
        save_array("mean_norm_lp.npy", mov_norm.mean(axis=0))
        save_array("max_norm_lp.npy", mov_norm.max(axis=0))

    # downsample movie at various spatial scales
    mov_norm_down, grid_down = spatially_downsample(
        mov=mov_norm, n_scales=n_scales)

    # xy dimensions original movie
    _, Ly, Lx = mov_norm.shape
    # xy dimensions downsampled movies
    Ly_down = [m.shape[-2] for m in mov_norm_down]
    Lx_down = [m.shape[-1] for m in mov_norm_down]

    # NOTE all variables ending with *_down are lists of the variable at different spatial scales

    ##########################
    # Spatial scale estimation
    scale_pix, thresh_peak, thresh_multiplier, vcorr = set_scale_and_thresholds(
        mov_norm_down, grid_down, spatial_scale, threshold_scaling)
    new_ops["Vcorr"] = vcorr
    new_ops["spatscale_pix"] = scale_pix

    ###############
    # ROI detection

    if thresh_peak_default is not None:
        thresh_peak = thresh_peak_default

    # get standard deviation for pixels for all values > Th2
    mov_norm_sd_down = [
        utils.threshold_reduce(m, thresh_peak) for m in mov_norm_down
    ]
    # needed so that scipy.io.savemat doesn't fail in runpipeline with latest numpy (v1.24.3).
    # dtype="object" is needed to have numpy array with elements having diff sizes
    new_ops["Vmap"] = np.asanyarray(mov_norm_sd_down, dtype="object").copy()

    # flatten x and y dimensions: (n_frames, x, y) -> (n_frames, x*y)
    mov_norm_down = [m.reshape(m.shape[0], -1) for m in mov_norm_down]
    mov_norm = mov_norm.reshape(mov_norm.shape[0], -1)

    # initialize variables
    max_val_per_roi = np.zeros(max_iterations)
    ratio_vexp_split = np.zeros(max_iterations)
    max_scale_per_roi = np.zeros(max_iterations)
    stats = []  # collect stats for each ROI

    # if extract_patches: TODO unused
    #     patches, seeds = [], []
    #     mask_window = ((scale_pix * 1.5) // 2) * 2

    for n in range(max_iterations):

        for i, arr in enumerate(mov_norm_sd_down):
            save_array(f'roi_{n}/sd_down_{i}.npy', arr)
        save_array(f'roi_{n}/mean_img.npy', mov_norm.mean(axis=0))
        save_array(f'roi_{n}/max_img.npy', mov_norm.max(axis=0))

        ############
        # FIND PEAKS

        # max value at each scale
        max_val_per_scale = np.array([m.max() for m in mov_norm_sd_down])
        if downsample_scale is not None:
            max_val = max_val_per_scale[downsample_scale]
            max_scale = downsample_scale
        else:
            # max value across scales
            max_val = max_val_per_scale.max()
            # scale with max value
            max_scale = np.argmax(max_val_per_scale)
        # index of max value at that scale
        max_idx = np.argmax(mov_norm_sd_down[max_scale])

        # add to output lists
        max_val_per_roi[n] = max_val
        max_scale_per_roi[n] = max_scale

        # check if peak is larger than threshold * max(1,nbinned/1200)
        if max_val < thresh_peak:
            break

        # downsampled position of peak
        max_idx_y_down, max_idx_x_down = np.unravel_index(
            max_idx, (Ly_down[max_scale], Lx_down[max_scale])
        )

        # original position of peak
        max_idx_y = int(
            grid_down[max_scale][1, max_idx_y_down, max_idx_x_down])
        max_idx_x = int(
            grid_down[max_scale][0, max_idx_y_down, max_idx_x_down])
        med = [max_idx_y, max_idx_x]

        ################
        # INITIALIZE ROI

        # make square of initial pixels based on spatial scale of peak
        ypix, xpix, lam = add_square(
            max_idx_y, max_idx_x, scale_in_pixel(max_scale), Ly, Lx)

        # project movie into square to get time series
        # this is the correlation between the movie and the mean activity per pixel in the ROI
        roi_corr = (mov_norm[:, ypix * Lx + xpix] * lam[0]).sum(axis=-1)

        # threshold active frames based on percentile or fixed value
        if percentile > 0:
            thresh_active = min(thresh_peak, np.percentile(roi_corr, percentile))
        else:
            thresh_active = thresh_peak
        active_frames = np.flatnonzero(roi_corr > thresh_active)

        # if extract_patches: # get square around seed TODO unused
        #     mask = mov[active_frames].mean(axis=0).reshape(Lyc, Lxc)
        #     patches.append(utils.square_mask(mask, mask_window, yi, xi))
        #     seeds.append([yi, xi])

        save_array(f'roi_{n}/ypix_0.npy', ypix)
        save_array(f'roi_{n}/xpix_0.npy', xpix)
        save_array(f'roi_{n}/lam_0.npy', lam)

        ############
        # EXTEND ROI
        active_frames_rec = []
        for i in range(n_iter_refine):
            # extend mask based on mean activity in active frames
            ypix, xpix, lam = iter_extend(
                ypix, xpix, mov_norm, Ly, Lx, active_frames, thresh_act_pix)
            
            save_array(f'roi_{n}/ypix_{i+1}.npy', ypix)
            save_array(f'roi_{n}/xpix_{i+1}.npy', xpix)
            save_array(f'roi_{n}/lam_{i+1}.npy', lam)

            # select pixels in ROI
            mov_norm_roi = mov_norm[:, ypix * Lx + xpix]
            # correlation of bins with lam within ROI
            roi_corr = mov_norm_roi @ lam
            # reselect active frames
            active_frames = np.flatnonzero(roi_corr > thresh_active)
            
            active_frames_rec.append(active_frames)
            if not active_frames.size:  # stop ROI extension if no active frames
                break
            
        if not active_frames.size:  # stop ROI detection if no active frames
            break
        
        #save index of active frames
        act_frames = active_frames_rec[-1]
        
        ###########
        # SPLIT ROI
        
        # check if splitting ROI increases variance explained
        ratio_vexp_split[n], ipack = two_comps(mov_norm_roi, lam, thresh_active)
        if ratio_vexp_split[n] > thresh_split:
            # if greater than threshold, update ROI to include 
            # only the component with higher variance explained
            lam, xp, active_frames = ipack
            roi_corr[active_frames] = xp
            ix = lam > (lam.max() * thresh_act_pix)
            xpix = xpix[ix]
            ypix = ypix[ix]
            lam = lam[ix]

            # determine med for new ROI
            ymed = np.median(ypix)
            xmed = np.median(xpix)
            imin = np.argmin((xpix - xmed) ** 2 + (ypix - ymed) ** 2)
            med = [ypix[imin], xpix[imin]]

        ########################
        # SUBTRACT ROI FROM DATA

        # from original movie:
        x = np.outer(roi_corr[active_frames], lam)
        # indices for active frames and pixels in ROI
        idx = np.ix_(active_frames, xpix + ypix * Lx)
        mov_norm[idx] -= x

        # from downsampled movie:
        # get ypix, xpix, and lam at each downsampled spatial scale
        ypix_down, xpix_down, lam_down = multiscale_mask(
            ypix, xpix, lam, Ly_down, Lx_down
        )
        for j in range(n_scales):

            # unpack variables at each scale
            ypix_d, xpix_d, lam_d, Lx_d, mov_norm_d, mov_norm_sd_d = (
                ypix_down[j],
                xpix_down[j],
                lam_down[j],
                Lx_down[j],
                mov_norm_down[j],
                mov_norm_sd_down[j],
            )

            # same as above
            x = np.outer(roi_corr[active_frames], lam_d)
            idx = np.ix_(active_frames, xpix_d + Lx_d * ypix_d)
            mov_norm_d[idx] -= x

            # update standard deviation
            Mx = mov_norm_d[:, xpix_d + Lx_d * ypix_d]
            x = (Mx**2 * (Mx > thresh_active)).sum(axis=0) ** 0.5
            mov_norm_sd_d[ypix_d, xpix_d] = x
        
        stats.append({ # add to list of ROI stats
            "ypix": ypix.astype(int),
            "xpix": xpix.astype(int),
            "lam": lam if use_alt_norm else lam * mov_sd[ypix, xpix],
            "med": med,
            "footprint": max_scale_per_roi[n],
            "act_frames": act_frames
        })
    
        if n % 1000 == 0:
            print("%d ROIs, score=%2.2f" % (n, max_val))
        
    if neuropil_lam:
        print('***neuropil_lam=True, calculating neuropil_lam...***')
        
        d0 = 10 if diameter is None or (isinstance(diameter, int) and
                                    diameter == 0) else diameter
        if aspect is not None:
            diameter = int(d0[0]) if isinstance(d0, (list, np.ndarray)) else int(d0)
            dy, dx = int(aspect * diameter), diameter
        else:
            dy, dx = (int(d0),
                      int(d0)) if not isinstance(d0, (list, np.ndarray)) else (int(d0[0]),
                                                                               int(d0[0]))
        rois = [
        ROI(ypix=s["ypix"], xpix=s["xpix"], lam=s["lam"], med=s["med"], do_crop=do_crop)
        for s in stats
        ]
        for roi, s in zip(rois, stats):
            ellipse = roi.fit_ellipse(dy, dx)
            s["npix"] = roi.n_pixels
            s["radius"] = ellipse.radius
        
        cell_pix = masks.create_cell_pix(stats, Ly=Ly, Lx=Lx,
                                   lam_percentile=lam_percentile)
        neuropil_masks = masks.create_neuropil_masks(ypixs=[stat["ypix"] for stat in stats],
                                    xpixs=[stat["xpix"] for stat in stats], cell_pix=cell_pix,
                                    inner_neuropil_radius = inner_neuropil_radius,
                                    min_neuropil_pixels = min_neuropil_pixels, circular=circular)
        
        for k in range(len(stats)):
            stats[k]["neuropil_mask"] = neuropil_masks[k]
        
        for stat in stats:
            act_frames = stat['act_frames']
            mov_act = mov_norm[act_frames]
            neuropil = np.unravel_index(stat['neuropil_mask'], (Ly,Lx))
            ypix_neu = neuropil[0]
            xpix_neu = neuropil[1]
            # npix = len(neuropil[1])
            roi_act = mov_act[:, ypix_neu * Lx + xpix_neu]
            lam = roi_act.mean(axis=0)
            lam = lam / np.sum(lam**2) ** 0.5
            stat['lam_neu'] = lam
            lam, neuropil_mask = zip(*sorted(zip(stat['lam_neu'], stat['neuropil_mask']), reverse=True)[:stat['npix']])
            lam = np.asarray(lam)
            stat['neuropil_mask'] = np.asarray(neuropil_mask)
            
            if norm == 'max': 
                lam_normed = (lam-lam.min())/(lam.max()-lam.min()) #normalization to scale lam > 0          
                stat['lam_neu'] = np.asarray(lam_normed)
            
            else:
                stat['lam_neu'] = lam
                
    
    else:
        print('***neuropil_lam=False***')


    new_ops.update({ # new items to be added to ops
        "Vmax": max_val_per_roi,
        "ihop": max_scale_per_roi,
        "Vsplit": ratio_vexp_split,
    })

    return new_ops, stats
