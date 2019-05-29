import glob
import numpy as np

from mkidpipeline import badpix as bp
from mkidpipeline.hdf.photontable import ObsFile as obs
from mkidpipeline.utils.plottingTools import plot_array as pa
from scipy.optimize import curve_fit
import mkidcore.pixelflags as pixelflags
from mkidcore.instruments import CONEX2PIXEL

from scipy.ndimage.filters import median_filter
import matplotlib.pyplot as plt

from photutils import DAOStarFinder, centroids, centroid_2dg, centroid_1dg, centroid_com, CircularAperture, aperture_photometry
from astropy.modeling.models import AiryDisk2D
from MKIDAnalysis.analysis_utils import *


def align_stack_image(output_dir, target_info, int_time, xcon, ycon, median_combine=False, make_numpy=True, save_shifts=True, hpm_again=True):
    """
    output_dir='/mnt/data0/isabel/microcastle/51Eri/51Eriout/dither3/'
    target_info='51EriDither3'
    int_time=60
    xcon = [-0.1, -0.1, -0.1, -0.1, -0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.35, 0.35, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5, 0.5, 0.5]
    ycon = [-0.5, -0.25, 0.0, 0.25, 0.5, -0.5, -0.25, 0.0, 0.25, 0.5, -0.5, -0.25, 0.0, 0.25, 0.5, -0.5, -0.25, 0.0, 0.25, 0.5, -0.5, -0.25, 0.0, 0.25, 0.5]
    """

    obsfile_list = glob.glob(output_dir + '15*.h5')
    npos = len(obsfile_list)
    wvlStart = 850  # True for parasiting
    wvlStop = 1100

    numpyfxnlist = []

    if make_numpy:
        for i in range(npos):
            obsfile = obs(obsfile_list[i], mode='write')
#            img = obsfile.getPixelCountImage(firstSec=0, integrationTime=int_time, applyWeight=True, flagToUse=0,
#                                             wvlStart=wvlStart, wvlStop=wvlStop)
#            print(
#            'Running getPixelCountImage on ', 0, 'seconds to ', int_time, 'seconds of data from wavelength ', wvlStart,
#            'to ', wvlStop)
            img = obsfile.getPixelCountImage(wvlStart=wvlStart, wvlStop=wvlStop)
            print('GOTIMAGE', i)
            saveim = np.transpose(img['image'])
#            usable_mask = np.array(obsfile.beamFlagImage) == pixelflags.GOODPIXEL
#            usable_mask=usable_mask.T
#            saveim*=usable_mask

            obsfile.file.close()

            outfile = output_dir + target_info + 'HPMasked%i.npy' % i

            if hpm_again:
                print('applying HPM to ', i)
                saveim=zeros_to_nans(saveim)
                saveimHP=quick_hpm(saveim, outfile, save=False)
                np.save(outfile, saveimHP)
            else:
                np.save(outfile, saveim)

            numpyfxnlist.append(outfile)
    else:
        for i in range(npos):
            outfile = output_dir + target_info + 'HPMasked%i.npy' % i
            numpyfxnlist.append(outfile)
        print(numpyfxnlist)

    rough_shiftsx = []
    rough_shiftsy = []
    centroidsx = []
    centroidsy = []

    pad_fraction = 0.8

    refpointx=CONEX2PIXEL(xcon[0], ycon[0])[0]
    refpointy=CONEX2PIXEL(xcon[0], ycon[0])[1]

    # load dithered science frames
    dither_frames = []
    eff_int_time_frames = []
    ideal_int_time_frames = []
    dither_frame_list=[]


    for i in range(npos):
        xpos = CONEX2PIXEL(xcon[i], ycon[i])[0]
        ypos = CONEX2PIXEL(xcon[i], ycon[i])[1]

        dx = refpointx - xpos
        dy = refpointy - ypos

        image=np.load(numpyfxnlist[i])
        image[image == 0] = ['nan']
        rough_shiftsx.append(dx)
        rough_shiftsy.append(dy)
        centroidsx.append(refpointx - dx)
        centroidsy.append(refpointy - dy)
        
        if median_combine:
            padded_frame = embed_image(image/int_time, framesize=pad_fraction, pad_value=np.nan)
            shifted_frame = rotate_shift_image(padded_frame, 0, dx, dy)
        else:
            padded_frame = embed_image(image, framesize=pad_fraction, pad_value=np.nan)
            shifted_frame = rotate_shift_image(padded_frame, 0, dx, dy)
            eff_int_time = np.full_like(padded_frame, fill_value=int_time)
            ideal_int_time = np.full_like(padded_frame, fill_value=int_time)
            eff_int_time[np.isnan(shifted_frame)] = [0]

            eff_int_time_frames.append(eff_int_time)
            ideal_int_time_frames.append(ideal_int_time)
        
        dither_frames.append(shifted_frame)

        if save_shifts:
            shifted_file=output_dir + target_info + 'HPMasked_Shifted%i.npy' % i
            np.save(shifted_file, shifted_frame)
            dither_frame_list.append(shifted_file)

    if median_combine:
        final_image = median_stack(np.array(dither_frames))
        outfile = output_dir + target_info + '_medianstacked_exptimeNORM'
        outfilestack = output_dir + target_info + '_stack_exptimeNORM'

    else:
        final_image=np.nansum(np.array(dither_frames), axis=0)
        outfile = output_dir + target_info + '_stacked'
        outfilestack = output_dir + target_info + '_stack'

        eff_int_time_frame = np.sum(np.array(eff_int_time_frames), axis=0)
        ideal_int_time_frame = np.sum(np.array(ideal_int_time_frames), axis=0)
        int_time_ratio = eff_int_time_frame / ideal_int_time_frame
        outfileinttimeratio = output_dir + target_info + '_intTimeRatio'
        np.save(outfileinttimeratio, int_time_ratio)
        outfile_effinttime = output_dir + target_info + '_effIntTime'
        outfile_idealinttime = output_dir + target_info + '_idealIntTime'
        np.save(outfile_effinttime, eff_int_time_frame)
        np.save(outfile_idealinttime, ideal_int_time_frame)

    if hpm_again:
        final_image=quick_hpm(final_image, outfile, save=False)
        outfile=outfile+'_HPMAgain'
    pa(final_image)
    np.save(outfile, final_image)
    np.save(outfilestack, dither_frames)



def quick_hpm(image, outfilename, save=True):
    dead_mask = np.isnan(image)
    reHPM = bp.hpm_flux_threshold(image, fwhm=4, dead_mask=dead_mask)
    if save:
        np.save(outfilename, reHPM['image'])
        pa(reHPM['image'])
    else:
        return reHPM['image']

def flux_estimator(datafileFLUX, xcentroid_flux, ycentroid_flux, sat_spot_bool=False, ND_filter_bool=True,
                   sat_spotcorr=5.5, ND_filtercorr=1, fwhm_est=2.5):
    data = np.load(datafileFLUX)
    fig, axs = plt.subplots(1, 1)
    axs.imshow(data, origin='lower', interpolation='nearest')

    marker = '+'
    ms, mew = 30, 2.
    box_size = fwhm_est * 5

    mask = np.isnan(data)

    axs.plot(xcentroid_flux, ycentroid_flux, color='red', marker=marker, ms=ms, mew=mew)  # check how we did
    positions = centroids.centroid_sources(data, xcentroid_flux, ycentroid_flux, box_size=int(box_size),
                                           centroid_func=centroid_com, mask=mask)
    axs.plot(positions[0], positions[1], color='blue', marker=marker, ms=ms, mew=mew)  # check how the fit did
    plt.title('Estimate in red, fit in blue')
    plt.show()

    xpos = positions[0]
    ypos = positions[1]

    positions_ap = [(xpos[0], ypos[0])]
    apertures = CircularAperture(positions_ap, r=fwhm_est)
    phot_table = aperture_photometry(data, apertures, mask=mask)

    fig, axs = plt.subplots(1, 1)
    axs.imshow(data, origin='lower', interpolation='nearest')
    apertures.plot(color='white', lw=1)
    plt.show()

    norm = phot_table['aperture_sum'].data

    if sat_spot_bool:
        norm = norm * (10 ** (sat_spotcorr / 2.5))
        factor = (1 / (10 ** (sat_spotcorr / 2.5)))

    if ND_filter_bool:
        factor = 10 ** (-ND_filtercorr)
        norm = norm / factor

    return {'norm': norm, 'xpos': xpos, 'ypos': ypos, 'factor': factor}


def PSF_estimator(datafilePSF, xcentroid_PSF, ycentroid_PSF, norm_by_exptime=True,
                  temporalfilePSF='/mnt/data0/isabel/microcastle/51Eri/51Eriout/dither1/CoronOut/51EriDither1CoronOut_effIntTime.npy', trueFWHM=2.5):

    data=np.load(datafilePSF)
    if norm_by_exptime:
        time=np.load(temporalfilePSF)
        data/=time
    
    fig, axs = plt.subplots(1, 1)
    axs.imshow(data, origin='lower', interpolation='nearest')

    marker = '+'
    ms, mew = 30, 2.

    axs.plot(xcentroid_PSF, ycentroid_PSF, color='red', marker=marker, ms=ms, mew=mew)  # check how we did
    plt.show()
    
    slicedatavert=data[:, xcentroid_PSF]
    slicedatavert[np.where(slicedatavert == np.nanmax(slicedatavert))] = (2000 / 0.0175)
    slicedatavertnorm = slicedatavert / np.nanmax(slicedatavert)

    slicedatahoriz=data[ycentroid_PSF, :]
    slicedatahoriz[np.where(slicedatahoriz == np.nanmax(slicedatahoriz))] = (1300 / 0.0175)
    slicedatahoriznorm = slicedatahoriz / np.nanmax(slicedatahoriz)

    slicedatavert_average=np.nanmean(data[:, xcentroid_PSF-1:xcentroid_PSF+1], axis=1)
    slicedatavertnorm_average=slicedatavert_average/np.nanmax(slicedatavert_average)

    slicedatahoriz_average=np.nanmean(data[ycentroid_PSF-1:ycentroid_PSF+1, :], axis=0)
    slicedatahoriznorm_average=slicedatahoriz_average/np.nanmax(slicedatahoriz_average)


    a2d=AiryDisk2D(radius=trueFWHM, x_0=xcentroid_PSF, y_0=ycentroid_PSF)
    pt=a2d(*np.mgrid[0:data.shape[1], 0:data.shape[0]])
    airy=pt.T

    sliceairyvert=airy[:, xcentroid_PSF-1]
    sliceairyvertnorm=sliceairyvert/np.nanmax(sliceairyvert)
    noisevert = np.random.normal(0, .001, sliceairyvert.shape)

    sliceairyhoriz=airy[ycentroid_PSF+1, :]
    sliceairyhoriznorm=sliceairyhoriz/np.nanmax(sliceairyhoriz)
    noisehoriz = np.random.normal(0, .001, sliceairyhoriz.shape)

    plt.plot(sliceairyhoriznorm)
#    plt.plot(slicedatahoriznorm_average)
    plt.plot(slicedatahoriznorm)
    plt.title('Ideal 1D Airy PSF with FWHM 1.22 lambda/D with actual PSF from a single dither (one slice)')
    plt.ylabel('Normalized Mean Counts/60sec')
    plt.xlabel('Pixels (horizontal)')
    plt.show()

    plt.plot(sliceairyvertnorm)
 #   plt.plot(slicedatavertnorm_average)
    plt.plot(slicedatavertnorm)
    plt.title('Ideal 1D Airy PSF with FWHM 1.22 lambda/D with actual PSF from a single dither (one slice)')
    plt.ylabel('Normalized Mean Counts/60sec')
    plt.xlabel('Pixels (vertical)')
    plt.show()


def prepare_forCC(datafile, outfile_name, interp=True, smooth=True, xcenter=242, ycenter=180, box_size=100):
    data = np.load(datafile)

    if interp:
        datainterp = interpolate_image(data)
        if smooth:
            proc_data = median_filter(datainterp, size=3)

    elif smooth:
        proc_data = median_filter(data, size=3)

    else:
        proc_data = data

    actualcenterx = int(data.shape[1] / 2)
    actualcentery = int(data.shape[0] / 2)
    roll = [actualcenterx - xcenter, actualcentery - ycenter]
    print(actualcenterx, actualcentery)
    proc_data_centered = np.roll(proc_data, roll[1], 0)
    proc_data_centered = np.roll(proc_data_centered, roll[0], 1)

    data_cropped = proc_data_centered[actualcentery - box_size:actualcentery + box_size,
                   actualcenterx - box_size:actualcenterx + box_size]
    pa(data_cropped)

    np.save(outfile_name, data_cropped)


def make_CoronagraphicProfile(datafileCC, unocculted=False, unoccultedfile='/mnt/data0/isabel/microcastle/51Eri/51EriProc/51EriUnocculted.npy',
                              normalize=1, fwhm_est=2.5, nlod=20, **fluxestkwargs):
    if unocculted:
        normdict = flux_estimator(unoccultedfile, **fluxestkwargs)
        norm = normdict['norm']
        factor = normdict['factor']
    else:
        norm = normalize

    speckles = np.load(datafileCC)
    lod = fwhm_est

    sep = np.arange(nlod + 1)

    # pixel coords of center of images.  Assume images are already centered
    centerx = int(speckles.shape[1] / 2)
    centery = int(speckles.shape[0] / 2)

    dead_mask = np.isnan(speckles)

    positions_ap1 = []
    for i in np.arange(nlod) + 1:
        positions_ap1.append([centerx, centery - i * lod])
    apertures1 = CircularAperture(positions_ap1, r=lod / 2)
    phot_table1 = aperture_photometry(speckles, apertures1, mask=dead_mask)
    print(phot_table1)

    fig, axs = plt.subplots(1, 1)
    axs.imshow(speckles, origin='lower', interpolation='nearest')
    apertures1.plot(color='white', lw=1)
    plt.show()

    fig, ax1 = plt.subplots()

    ax1.plot(sep[1:], phot_table1['aperture_sum'] / normalize, linewidth=2, label=r'Coronagraphic PSF Profile')
    ax1.plot(sep[1:], np.sqrt(phot_table1['aperture_sum']) / norm, linestyle='-.', linewidth=2,
             label=r'1-$\sigma$ Photon noise')

    ax1.set_xlabel(r'Separation ($\lambda$/D)', fontsize=14)
    ax1.set_ylabel(r'Normalized Azimuthally Averaged Intensity', fontsize=14)
    ax1.set_xlim(0, nlod)

    ax1.set_ylim(2e-5, 1)
    ax1.set_yscale('log')
    ax1.legend()
    plt.show()


def make_CC(datafileCC, unocculted=False, unoccultedfile='/mnt/data0/isabel/microcastle/51Eri/51EriProc/51EriUnocculted.npy',
            calc_flux=False, normalize=1, fwhm_est=2.5, nlod=20, plot=False, verbose=False, **fluxestkwargs):
    if unocculted and calc_flux:
        normdict = flux_estimator(unoccultedfile, **fluxestkwargs)
        norm = normdict['norm']
        factor = normdict['factor']
    else:
        norm = normalize

    speckles = np.load(datafileCC)
    lod = fwhm_est
    sep_full = np.arange(nlod + 1)

    # pixel coords of center of images.  Assume images are already centered
    centerx = int(speckles.shape[1] / 2)
    centery = int(speckles.shape[0] / 2)

    dead_mask = np.isnan(speckles)

    spMeans0 = [0]
    spMeans = [0]
    spStds = [0]
    spSNRs = [0]

    for i in np.arange(nlod) + 1:
        sourcex = centerx
        sourcey = centery - i * lod
        sep = dist(centery, centerx, sourcey, sourcex)

        angle = np.arcsin(lod / 2. / sep) * 2
        number_apertures = int(np.floor((2) * np.pi / angle))
        yy = np.zeros((number_apertures))
        xx = np.zeros((number_apertures))
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        xx[0] = sourcex - centerx
        yy[0] = sourcey - centery
        for j in range(number_apertures - 1):
            xx[j + 1] = cosangle * xx[j] + sinangle * yy[j]
            yy[j + 1] = cosangle * yy[j] - sinangle * xx[j]

        xx[:] += centerx
        yy[:] += centery
        rad = lod / 2.
        apertures = CircularAperture((xx, yy), r=rad)  # Coordinates (X,Y)
        fluxes = aperture_photometry(speckles, apertures, method='exact')
        fluxes = np.array(fluxes['aperture_sum'])

        f_source = fluxes[0].copy()
        fluxes = fluxes[1:]
        n2 = fluxes.shape[0]
        snr = (normalize - np.nanmean(fluxes)) / (np.nanstd(fluxes) * np.sqrt(1 + (1 / n2)))

        if plot:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(speckles, origin='lower', interpolation='nearest', alpha=0.5,
                      cmap='gray')
            for i in range(xx.shape[0]):
                # Circle takes coordinates as (X,Y)
                aper = plt.Circle((xx[i], yy[i]), radius=lod / 2., color='r', fill=False, alpha=0.8)
                ax.add_patch(aper)
                cent = plt.Circle((xx[i], yy[i]), radius=0.8, color='r', fill=True, alpha=0.5)
                ax.add_patch(cent)
                aper_source = plt.Circle((sourcex, sourcey), radius=0.7, color='b', fill=True, alpha=0.5)
                ax.add_patch(aper_source)
            ax.grid(False)
            plt.show()

        spMeans0.append(f_source)
        spMeans.append(np.nanmean(fluxes))
        spStds.append(np.nanstd(fluxes))
        spSNRs.append(snr)

    spMeans0 = np.array(spMeans0)
    spMeans = np.array(spMeans)
    spStds = np.array(spStds)
    spSNRs = np.array(spSNRs)

    if verbose:
        print('Source Flux', spMeans0)
        print('Mean of the Fluxes in the Lamda/D circle', spMeans)
        print('Std Dev of the Fluxes in the Lambda/D circle', spStds)
        print('SNRs', 1 / spSNRs)

    if unocculted:
        psf=np.load(unoccultedfile)

        # pixel coords of center of images.  Assume images are already centered
        centerx = int(psf.shape[1] / 2)
        centery = int(psf.shape[0] / 2)

        dead_mask = np.isnan(psf)

        psfMeans0 = [0]
        psfMeans = [0]
        psfStds = [0]
        psfSNRs = [0]

        for i in np.arange(nlod) + 1:
            sourcex = centerx
            sourcey = centery - i * lod
            sep = dist(centery, centerx, sourcey, sourcex)

            angle = np.arcsin(lod / 2. / sep) * 2
            number_apertures = int(np.floor((2) * np.pi / angle))
            yy = np.zeros((number_apertures))
            xx = np.zeros((number_apertures))
            cosangle = np.cos(angle)
            sinangle = np.sin(angle)
            xx[0] = sourcex - centerx
            yy[0] = sourcey - centery
            for j in range(number_apertures - 1):
                xx[j + 1] = cosangle * xx[j] + sinangle * yy[j]
                yy[j + 1] = cosangle * yy[j] - sinangle * xx[j]

            xx[:] += centerx
            yy[:] += centery
            rad = lod / 2.
            apertures = CircularAperture((xx, yy), r=rad)  # Coordinates (X,Y)
            fluxes = aperture_photometry(psf, apertures, method='exact')
            fluxes = np.array(fluxes['aperture_sum'])

            f_source = fluxes[0].copy()
            fluxes = fluxes[1:]
            n2 = fluxes.shape[0]
            snr = (normalize - np.nanmean(fluxes)) / (np.nanstd(fluxes) * np.sqrt(1 + (1 / n2)))

            if plot:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(psf, origin='lower', interpolation='nearest', alpha=0.5,
                          cmap='gray')
                for i in range(xx.shape[0]):
                    # Circle takes coordinates as (X,Y)
                    aper = plt.Circle((xx[i], yy[i]), radius=lod / 2., color='r', fill=False, alpha=0.8)
                    ax.add_patch(aper)
                    cent = plt.Circle((xx[i], yy[i]), radius=0.8, color='r', fill=True, alpha=0.5)
                    ax.add_patch(cent)
                    aper_source = plt.Circle((sourcex, sourcey), radius=0.7, color='b', fill=True, alpha=0.5)
                    ax.add_patch(aper_source)
                ax.grid(False)
                plt.show()

            psfMeans0.append(f_source)
            psfMeans.append(np.nanmean(fluxes))
            psfStds.append(np.nanstd(fluxes))
            psfSNRs.append(snr)

        psfMeans0 = np.array(psfMeans0)
        psfMeans = np.array(psfMeans)
        psfStds = np.array(psfStds)
        psfSNRs = np.array(psfSNRs)

    fig, ax1 = plt.subplots()

    ax1.plot(sep_full[1:], spMeans[1:] / norm, linewidth=2, label=r'Azimuthally Averaged Mean Coronagraphic Intensity')
    ax1.plot(sep_full[1:], spStds[1:] / norm, linestyle='-.', linewidth=2, label=r'Azimuthal Standard Deviation')
    ax1.plot(sep_full[1:], np.sqrt(spMeans[1:]) / norm, linestyle='-.', linewidth=2,
             label=r'Square Root of the Azimuthally Averaged Mean Coronagraphic Intensity')

    if unocculted:
        ax1.plot(sep_full[1:], (psfMeans[1:]/factor) / norm, linewidth=2, label=r'Unocculted PSF Profile')

    ax1.set_xlabel(r'Separation ($\lambda$/D)', fontsize=14)
    ax1.set_ylabel(r'Normalized by Unocculted PSF Intensity', fontsize=14)
    ax1.set_xlim(0, nlod)

    ax1.set_ylim(2e-6, 1)
    ax1.set_yscale('log')
    ax1.legend()
    plt.show()
