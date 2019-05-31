import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import rotate, zoom
import argparse
from mkidpipeline.imaging.drizzler import form
import mkidpipeline
from mkidpipeline.utils.plottingTools import plot_array as pa
from MKIDAnalysis.analysis_utils import *
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
from astroplan import Observer
import astropy
import datetime


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


def rot_array(img, pivot,angle):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]


def clipped_zoom(img, zoom_factor, **kwargs):
    """ Courtesy of
    https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions """
    img=nans_to_zeros(img)
    img[img==np.inf]=0
    h, w = img.shape[:2]
    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:
        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        # Zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:
        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        if trim_top < 0 or trim_left < 0:
            temp = np.zeros_like(img)
            temp[:out.shape[0], :out.shape[1]] = out
            out = temp
        else:
            out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out



def SDI(plot_diagnostics=False, integration_time=100, number_time_bins=10):
    """
    Median collapse a tesseract along the time dimension to produce a spectral cube with minimal hot pixels. Then
    radially scale the channels and median collapse to produce the reference PSF. Scale and bubtract that PSF from the
    spectral cube and median collpase to form an image. Needs to be verified
    """

    parser = argparse.ArgumentParser(description='Photon Drizzling Utility')
    parser.add_argument('cfg', type=str, help='The configuration file')
    args = parser.parse_args()
    cfg = mkidpipeline.config.load_task_config(args.cfg)

    fitsname = 'spec_cube'

    nwvlbins = 5
    wvlMin = 850
    wvlMax = 1100
    wsamples = np.linspace(wvlMin, wvlMax, nwvlbins + 1)
    scale_list = wsamples[::-1] * 2. / (wvlMax + wvlMin)

    fullfitsname = fitsname+'.fits'
    if os.path.exists(fullfitsname):
        hdul = fits.open(fullfitsname)
        spec_cube = hdul[0].data
    else:
        drizzle = form(cfg.dither, mode='temporal', rotation_center=cfg.drizzler.rotation_center,
                       pixfrac=cfg.drizzler.pixfrac, wvlMin=wvlMin,
                       wvlMax=wvlMax, intt=integration_time, ntimebins=number_time_bins, device_orientation=cfg.drizzler.device_orientation, nwvlbins=nwvlbins,
                       derotate=True)

        tess = drizzle.data
        spec_cube = np.sum(tess, axis=0) / drizzle.image_weights
        fits.writeto(fullfitsname, spec_cube, overwrite=True)

    if plot_diagnostics:
        # Inspect the spectral cube
        for i in range(nwvlbins):
            plt.figure()
            plt.imshow(spec_cube[i])
            if i == nwvlbins - 1:
                plt.show(block=True)

    # Using PCA doesn't appear to work well
    # SDI = pca.pca(spec_cube, angle_list=np.zeros((spec_cube.shape[0])), scale_list=scale_list)

    # Do it manually

    # stretch the spec_cube so the speckles line up and the planet moves radially
    scale_cube = np.zeros_like(spec_cube)

    for i in range(nwvlbins):
        scale_cube[i] = clipped_zoom(spec_cube[i], scale_list[i])

    if plot_diagnostics:
        for i in range(nwvlbins):
            plt.figure()
            plt.imshow(scale_cube[i])
            if i == nwvlbins - 1:
                plt.show(block=True)

    # median collapse to create the static speckle reference
    ref = np.median(scale_cube, axis=0)
    plt.imshow(ref)
    plt.show(block=True)

    # scale that so the ref image matches the static speckles at each wavelength and the planet doesn't move
    ref_cube = np.zeros_like(spec_cube)
    for i in range(nwvlbins):
        ref_cube[i] = clipped_zoom(ref, scale_list[-i])

    if plot_diagnostics:
        for i in range(nwvlbins):
            ref_cube[i] = clipped_zoom(ref, scale_list[-i])
            plt.figure()
            plt.imshow(ref_cube[i])
            if i == nwvlbins - 1:
                plt.show(block=True)

    # perform the subtraction and mean collapse to build S/N of planet
    SDI=np.nanmean(spec_cube - ref_cube, axis=0)

    plt.imshow(SDI)
    np.save('./SDI',SDI)
    plt.show()

def ADI_check(target='HD 984', year=2018, month=12, day=21, observatory='subaru'):

    st = datetime.datetime(year=year, month=month, day=day, hour=3, tzinfo=datetime.timezone.utc)  # sunset is 5 pm Hawaii time + 10 hours
    et = datetime.datetime(year=year, month=month, day=day, hour=18, tzinfo=datetime.timezone.utc)  # sunrise is 8 am Hawaii time + 10 hours

    uts = np.int_([st.timestamp(), et.timestamp()])
    apo = Observer.at_site(observatory)
    times = range(uts[0], uts[1], 60)
    site = EarthLocation.of_site(observatory)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    fig.autofmt_xdate()
    dtobs = [datetime.datetime.fromtimestamp(time) for time in times]

    coords = SkyCoord.from_name(target)
    altaz = apo.altaz(astropy.time.Time(val=times, format='unix'), coords)
    earthrate = 360 / u.sday.to(u.second)

    parallactic_angles = apo.parallactic_angle(astropy.time.Time(val=times, format='unix'), SkyCoord.from_name(target)).value

    lat = site.geodetic.lat.rad
    az = altaz.az.radian
    alt = altaz.alt.radian

    rot_rates = earthrate * np.cos(lat) * np.cos(az) / np.cos(alt)  # Smart 1962
    axs[0].plot(dtobs, rot_rates*3600)
    axs[0].set_ylabel('rot rate (arcsec/s)')
    axs[0].set_title(target)
    axs[1].plot(dtobs, parallactic_angles)
    axs[1].set_ylabel('Parallactic Angles')
    axs[1].set_title(target)
    plt.show()




    plt.show()


if __name__ == '__main__':
    SDI(plot_diagnostics=True)