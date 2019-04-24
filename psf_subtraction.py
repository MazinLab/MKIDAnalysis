import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
from astropy.io import fits
from scipy.ndimage import rotate, zoom
import argparse
# from vip_hci import pca
from mkidpipeline.imaging.drizzler import form, pretty_plot, write_fits
import mkidpipeline
from mkidpipeline.utils.plottingTools import plot_array as pa
import mkidcore.corelog as pipelinelog


from photutils import DAOStarFinder, centroids, centroid_2dg, centroid_1dg, centroid_com, CircularAperture, aperture_photometry
from analysis_utils import *


def rudimentaryPSF_subtract(ditherframe_file, PSF_file, target_info, npos=25, combination_mode='median'):
    dither_frames=np.load(ditherframe_file)
    PSF=np.load(PSF_file)
    for i in range(npos):
        dither_frames[i,:,:]=dither_frames[i,:,:]-PSF
    if combination_mode=='median':
        outfilesub = target_info+ '_medianstacked_PSFSub'
        PSF_sub=median_stack(dither_frames)
    if combination_mode=='mean':
        outfilesub = target_info + '_meanstacked_PSFSub'
        PSF_sub = mean_stack(dither_frames)
    np.save(outfilesub, PSF_sub)
    pa(PSF_sub)


def SDI():
    """

    Median collapse a tesseract along the time dimension to produce a spectral cube with minimal hot pixels. Then
    radially scale the channels and median collapse to produce the reference PSF. Scale and subtract that PSF from the
    spectral cube and median collapse to form an image. Needs to be verified

    """

    matplotlib.use('QT5Agg', force=True)
    matplotlib.rcParams['backend'] = 'Qt5Agg'

    parser = argparse.ArgumentParser(description='Photon Drizzling Utility')
    parser.add_argument('cfg', type=str, help='The configuration file')
    parser.add_argument('-wl', type=float, dest='wvlMin', help='', default=850)
    parser.add_argument('-wh', type=float, dest='wvlMax', help='', default=1100)
    parser.add_argument('-t0', type=int, dest='startt', help='', default=0)
    parser.add_argument('-it', type=int, dest='intt', help='', default=60)
    args = parser.parse_args()

    # set up logging
    mkidpipeline.logtoconsole()
    log_format = "%(levelname)s : %(message)s"
    pipelinelog.create_log('mkidpipeline', console=True, fmt=log_format, level="INFO")

    getLogger('mkidpipeline.hdf.photontable').setLevel('info')

    # load as a task configuration
    cfg = mkidpipeline.config.load_task_config(args.cfg)

    wvlMin = args.wvlMin
    wvlMax = args.wvlMax
    startt = args.startt
    intt = args.intt
    pixfrac = cfg.drizzler.pixfrac
    dither = cfg.dither

    nwvlbins = 5
    wsamples = np.linspace(wvlMin, wvlMax, nwvlbins + 1)
    scale_list = wsamples[::-1] * 2. / (wvlMax + wvlMin)

    # Get tesseract of data
    tess, drizwcs = form(dither, 'temporal', virPixStar=(20, 20), wvlMin=wvlMin, wvlMax=wvlMax,
                         startt=startt, intt=intt, pixfrac=pixfrac, nwvlbins=nwvlbins)

    # Get median spectral cube
    mask_tess = np.ma.masked_where(tess == 0, tess)
    medDither = np.ma.median(mask_tess, axis=0).filled(0)

    # Inspect the spectral cube
    for i in range(nwvlbins):
        show = True if i == nwvlbins - 1 else False
        pretty_plot(medDither[i], drizwcs.wcs.cdelt[0], drizwcs.wcs.crval, vmin=1, vmax=10, show=show)

    fits.writeto(cfg.dither.name + '_med.fits', medDither, drizwcs.to_header(), overwrite=True)

    # Using PCA doesn't appear to work well
    # SDI = pca.pca(medDither, angle_list=np.zeros((medDither.shape[0])), scale_list=scale_list)

    # Do it manually
    scale_cube = np.zeros_like(medDither)
    for i in range(nwvlbins):
        scale_cube[i] = clipped_zoom(medDither[i], scale_list[i])
        show = True if i == nwvlbins - 1 else False
        pretty_plot(scale_cube[i], drizwcs.wcs.cdelt[0], drizwcs.wcs.crval, vmin=1, vmax=10, show=show)

    ref = np.median(scale_cube, axis=0)
    SDI = medDither - ref

    pretty_plot(SDI, drizwcs.wcs.cdelt[0], drizwcs.wcs.crval, vmin=1, vmax=10)

    fits.writeto(cfg.dither.name + '_SDI.fits', SDI, drizwcs.to_header(), overwrite=True)