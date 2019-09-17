import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import rotate, zoom
import argparse
from mkidpipeline.imaging.drizzler import form
import mkidpipeline
from mkidpipeline.utils.plottingTools import plot_array as pa
from mkidpipeline.config import config
import mkidpipeline.imaging.drizzler as drizzler
from mkidpipeline import badpix as bp
import mkidpipeline as pipe
from mkidanalysis.analysis_utils import *
import vip_hci as vip
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
from astroplan import Observer
import astropy
import datetime
import pytz
import mkidcore.corelog as pipelinelog
from vip_hci.preproc import cube_derotate, frame_rotate

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

def ADI(file='./ADI_HPM.fits', datafile = '/mnt/data0/isabel/mec/HD1160/data_HD1160.yml', outfile = '/mnt/data0/isabel/mec/HD1160/out_HD1160.yml',
        cfgfile = '/mnt/data0/isabel/mec/HD1160/pipe_HD1160.yml', temporal_file_derot='/mnt/data0/isabel/mec/HD1160/HD1160_temporal.fits',
        temporal_file='/mnt/data0/isabel/mec/HD1160/HD1160_temporal_NDR.fits'):
    """
    First median collapsing a time cube of all the raw dither images to a get a static PSF of the virtual grid. For
    each dither derotate the static PSF and isolate the relevant area. Subtract that static reference from the
    median of the derorated dither
    :return:
    """

    pipe.configure_pipeline(cfgfile)

    out_collection = pipe.load_output_description(outfile, datafile=datafile)
    outputs = out_collection.outputs

    scidata_static = fits.open(temporal_file)
    tess_static = scidata_static[1].data

    y = np.ma.masked_where(tess_static[:, 0] == 0, tess_static[:, 0])
    static_map = np.ma.median(y, axis=0).filled(0)

    scidata = fits.open(temporal_file_derot)
    tess = scidata[1].data

    y2 = np.ma.masked_where(tess[:, 0] == 0, tess[:, 0])
    rot_map = np.ma.median(y2, axis=0).filled(0)
    tcube = y2.data

    timesteps=len(tcube[:, 0, 0])

    HAs = make_angle_list(target=outputs[0].data.target, obs_start=outputs[0].data.obs[0].start,
                         int_time=int_time, nsteps=timesteps, observatory=outputs[0].data.observatory)


    # derotate the static psf for each dither time
    derot_static = np.zeros([timesteps, static_map.shape[0], static_map.shape[1]])
    starxy = [int(scidata[1].header['CRPIX1']), int(scidata[1].header['CRPIX2'])]
    for ia, ha in enumerate(HAs):
        print(ia, ha)
        derot_static[ia] = rot_array(static_map, starxy, -(np.rad2deg(HAs[0])-np.rad2deg(ha)))

    for i, image in enumerate(tcube):
        # shrink the reference psf to the relevant area
        print(i)
        derot_static[i][image == 0] = 0

        #subtract the reference
        image -= derot_static[i]

    # sum collapse the differential
    diff = np.ma.mean(tcube, axis=0).filled(0)
    plt.imshow(diff, origin='lower')
    plt.show()

    hdul = fits.HDUList([fits.PrimaryHDU(header=scidata[1].header),
                         fits.ImageHDU(data=diff, header=scidata[1].header)])

    hdul.writeto(file, overwrite=True)

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

def Sky_Rotation(target='* Kap And', year=2018, month=12, day=24, observatory='subaru'):

    st = datetime.datetime(year=year, month=month, day=day, hour=3, tzinfo=datetime.timezone.utc)  # sunset is 5 pm Hawaii time + 10 hours
    et = datetime.datetime(year=year, month=month, day=day, hour=18, tzinfo=datetime.timezone.utc)  # sunrise is 8 am Hawaii time + 10 hours

    uts = np.int_([st.timestamp(), et.timestamp()])
    apo = Observer.at_site(observatory)
    times = range(uts[0], uts[1], 30)
    site = EarthLocation.of_site(observatory)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    fig.autofmt_xdate()
    dtobs = [datetime.datetime.fromtimestamp(time, pytz.timezone('UTC')) for time in times]

    coords = SkyCoord.from_name(target)
    altaz = apo.altaz(astropy.time.Time(val=times, format='unix'), coords)
    earthrate = 360 / u.sday.to(u.second)

    parallactic_angles = apo.parallactic_angle(astropy.time.Time(val=times, format='unix'), SkyCoord.from_name(target)).value

    lat = site.geodetic.lat.rad
    az = altaz.az.radian
    alt = altaz.alt.radian

    rot_rates = earthrate * np.cos(lat) * np.cos(az) / np.cos(alt)  # Smart 1962
    axs[0].plot(dtobs, rot_rates)
    axs[0].set_ylabel('rot rate (deg/s)')
    axs[0].set_title(target)
    axs[1].plot(dtobs, parallactic_angles)
    axs[1].set_ylabel('Parallactic Angles')
    axs[1].set_title(target)
    plt.show()


def ADI_check(target='* kap And', obs_start=1545626973, obs_end=1545627075, observatory='subaru', target_sep=1000, scale_factor=10):

    uts = np.int_([obs_start, obs_end])
    apo = Observer.at_site(observatory)
    times = range(uts[0], uts[1], 1)
    site = EarthLocation.of_site(observatory)


    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    fig.autofmt_xdate()
    dtobs = [datetime.datetime.fromtimestamp(time) for time in times]
    coords = SkyCoord.from_name(target)
    altaz = apo.altaz(astropy.time.Time(val=times, format='unix'), coords)
    earthrate = 360 / u.sday.to(u.second)
    interval = obs_end - obs_start

    parallactic_angles = apo.parallactic_angle(astropy.time.Time(val=times, format='unix'), SkyCoord.from_name(target)).value
    #These are in radians!

    lat = site.geodetic.lat.rad
    az = altaz.az.radian
    alt = altaz.alt.radian
    rot_rates = earthrate * np.cos(lat) * np.cos(az) / np.cos(alt)  # Smart 1962

    delta_pa=abs(parallactic_angles[-1]-parallactic_angles[0])

    milliarcsec_traced_y=np.sin(delta_pa)*target_sep
    pix_traced_y=milliarcsec_traced_y/scale_factor

    milliarcsec_traced_x=(1-np.cos(delta_pa))*target_sep
    pix_traced_x=milliarcsec_traced_x/scale_factor

    degrees_traced=abs(np.median(rot_rates)*(times[-1]-times[0]))
    milliarcsec_traced_y_2=np.sin(np.deg2rad(degrees_traced))*target_sep
    pix_traced_y_2 = milliarcsec_traced_y_2 / scale_factor

    degrees_traced=np.median(rot_rates)*(times[-1]-times[0])
    milliarcsec_traced_x_2=(1-np.cos(np.deg2rad(degrees_traced)))*target_sep
    pix_traced_x_2 = milliarcsec_traced_x_2 / scale_factor

    axs[0].plot(dtobs, rot_rates)
    axs[0].set_ylabel('rot rate (deg/s)')
    axs[0].set_xlabel('Time (UTC)')
    axs[0].set_title(target + ' Shift_1=' + str(np.around(pix_traced_y_2, decimals=2)) + ' Shift_2=' + str(
        np.around(pix_traced_x_2, decimals=2)))
    axs[1].plot(dtobs, parallactic_angles)

    axs[1].set_ylabel('Parallactic Angles (Rad)')
    axs[1].set_xlabel('Time (UTC)')
    axs[1].set_title(target + ' Shift_1=' + str(np.around(pix_traced_y, decimals=2)) + ' Shift_2=' + str(
        np.around(pix_traced_x, decimals=2)))

    plt.show()

def make_angle_list(target='* kap And', obs_start=1545626973, int_time=100, nsteps=25, observatory='subaru'):

    uts = np.int_([obs_start, obs_start+(nsteps*int_time)])
    apo = Observer.at_site(observatory)
    times = range(uts[0], uts[1], int_time)
    parallactic_angles = apo.parallactic_angle(astropy.time.Time(val=times, format='unix'), SkyCoord.from_name(target)).value
    #These are in radians!

    return(np.array(parallactic_angles))

def median_sub_VIP(dither_stack_file, mode='fullfr', collapse='median', target='* kap And',
               obs_start=1545626973, int_time=100, nsteps=25, observatory='subaru'):

    data_stack=np.load(dither_stack_file)
    angle_list=make_angle_list(target=target, obs_start=obs_start, int_time=int_time, nsteps=nsteps, observatory=observatory)

    angle_list=-angle_list[0]+angle_list

    delta_pa = angle_list[-1]-angle_list[0]
    print(delta_pa)


    resid_cube, resid_cube_derot, collapse_frame=vip.medsub.medsub_source.median_sub_MEC(
        cube=data_stack, angle_list=angle_list, scale_list=None,fwhm=2.5, asize=2.5, full_output=True,
        collapse=collapse, mode=mode, verbose=True)

    return(resid_cube, resid_cube_derot, collapse_frame)

def derotate_cube(cube, target='* kap And', obs_start=1545626973, int_time=100, nsteps=25, observatory='subaru'):
    cube = nans_to_zeros(cube)
    angle_list = make_angle_list(target=target, obs_start=obs_start, int_time=int_time, nsteps=nsteps,
                                 observatory=observatory)
    derotated_cube = cube_derotate(cube, angle_list)
    derotated_cube = zeros_to_nans(derotated_cube)

    return(derotated_cube)

def derotate_frame(frame, target='* kap And', obs_start=1545626973, int_time=100, nsteps=25, observatory='subaru'):
    frame = nans_to_zeros(frame)
    angle_list = make_angle_list(target=target, obs_start=obs_start, int_time=int_time, nsteps=nsteps,
                                 observatory=observatory)
    angle = angle_list[-1] - angle_list[0]
    rotated_frame = frame_rotate(frame, angle)
    rotated_frame = zeros_to_nans(rotated_frame)

    return(rotated_frame)

def derotate_cube_CPS(int_time_file, stack_file, HPM = True, target='* kap And', obs_start=1545626973, int_time=100, nsteps=25, observatory='subaru'):
    int_time_array=np.load(int_time_file)
    stack=np.load(stack_file)

    if HPM:
        stack=threshold_HPM_stack(stack, save=False, factor=3)

    stack_derotated = derotate_cube(stack, target=target, obs_start=obs_start, int_time=int_time, nsteps=nsteps, observatory=observatory)
    stacked_derotated = np.nansum(stack_derotated, axis=0)

    int_time_array_derotated = derotate_cube(int_time_array, target=target, obs_start=obs_start, int_time=int_time, nsteps=nsteps, observatory=observatory)
    int_time_derotated = np.nansum(int_time_array_derotated, axis=0)

    stacked_normalized =  stacked_derotated/int_time_derotated

    pa(stacked_normalized)
    return(stacked_normalized, int_time_array_derotated, stacked_derotated)



if __name__ == '__main__':
    ADI()