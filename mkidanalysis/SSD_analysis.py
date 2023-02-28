'''
Code to Perform Stochastic Speckle Discrimination (SSD) on MEC data

Example Usage:

ssd = SSDAnalyzer(h5_files=h5_files, fits_dir=save_folder + 'fits/', component_dir=save_folder, plot_dir=save_folder,
                  ncpu=10, save_plot=True, set_ip_zero=False, binned=False)
ssd.run_ssd()

'''
import matplotlib.pyplot as plt
import numpy as np
import os
from mkidanalysis.speckle.binned_rician import muVar_to_IcIs, maxBinMRlogL, getLightCurve
from mkidanalysis.speckle.binfree_rician import optimize_IcIsIr2
from matplotlib import gridspec
from mkidcore.corelog import getLogger
from progressbar import ProgressBar
from matplotlib.patches import Circle
from multiprocessing import Pool
from functools import partial
from drizzle.drizzle import Drizzle
import warnings
from mkidpipeline.photontable import Photontable
from astropy.io import fits
import matplotlib
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from mkidpipeline.utils import pdfs
from mkidanalysis.lightCurves import histogramLC
from mkidanalysis.lucky_imaging import get_lucky
import astropy.units as u
from mkidcore.instruments import CONEX2PIXEL
from mkidpipeline.utils.snr import get_num_apertures, get_aperture_center

warnings.filterwarnings("ignore")
getLogger().setLevel('INFO')

PROBLEM_FLAGS = ('pixcal.hot', 'pixcal.cold', 'pixcal.dead', 'beammap.noDacTone', 'wavecal.bad')


class SSDManager:
    def __init__(self, output=None, fits_dir='', component_dir='', ncpu=1, prior=None, prior_sig=None, set_ip_zero=False,
                 binned=True, drizzle=True, bin_size=0.01, read_noise=None, deadtime=None, lucky=False,
                 adi_mode=False):
        self.outputs = [o for o in output]
        self.dataset = output.dataset
        self.names = [o.data.name for o in self.outputs]
        self.offsets = [o.start_offset for o in self.outputs]
        self.durations = [o.duration for o in self.outputs]
        self.timesteps = [o.timestep for o in self.outputs]
        self.fits_dir = fits_dir
        self.component_dir = component_dir
        self.ncpu = ncpu
        self.prior = prior
        self.prior_sig = prior_sig
        self.set_ip_zero = set_ip_zero
        self.binned = binned
        self.drizzle = drizzle
        self.bin_size = bin_size
        self.read_noise = read_noise
        self.deadtime = deadtime
        self.lucky = lucky
        self.photontables = None
        self.intt = None
        self.adi_mode = adi_mode

    def run(self):
        for i, name in enumerate(self.names):
            h5s = [o.h5 for o in self.dataset.datadict[name].obs]
            self.photontables = [Photontable(h5) for h5 in h5s]
            self.intt = self.photontables[0].query_header('EXPTIME')
            ssd = SSDAnalyzer(h5_files=h5s, fits_dir=self.fits_dir, component_dir=self.component_dir,
                              ncpu=self.ncpu, prior=self.prior, prior_sig=self.prior_sig, set_ip_zero=self.set_ip_zero,
                              binned=self.binned, drizzle=self.drizzle, bin_size=self.bin_size,
                              read_noise=self.read_noise, deadtime=self.deadtime, startt=self.offsets[i],
                              duration=self.durations[i], lucky=self.lucky, timestep=self.timesteps[i],
                              adi_mode=self.adi_mode)
            ssd.run_ssd()


class SSDAnalyzer:
    """Class for running binned or binfree SSD analysis on MKID data (calibrated h5 files)"""
    def __init__(self, h5_files=None, fits_dir='', component_dir='', ncpu=1, prior=None, prior_sig=None, set_ip_zero=False,
                 binned=True, drizzle=True, bin_size=0.01, read_noise=None, deadtime=None, startt=None, duration=None,
                 lucky=False, timestep=None, adi_mode=False):
        self.fits_dir = fits_dir
        self.component_dir = component_dir
        self.ncpu = ncpu
        self.set_ip_zero = set_ip_zero
        self.prior = prior
        self.prior_sig = prior_sig
        self.binned = binned
        self.h5_files = h5_files
        self.drizzle = drizzle
        self.bin_size = bin_size
        self.read_noise = read_noise
        self.deadtime = deadtime
        self.duration = duration
        self.startt = startt
        self.photontables = [Photontable(h5) for h5 in self.h5_files]
        self.intt = self.photontables[0].query_header('EXPTIME')
        self.lucky = lucky
        self.timestep = timestep
        self.data = None
        self.adi_mode = adi_mode
        self.ic_cube = None
        self.is_cube = None
        self.ip_cube = None 

    def run_ssd(self):
        """
        Runs SSD code for all outputs in the specified ou.yaml file adhering to specified durations, timesteps, and
        starttimes. Saves outputs in the component_dir and fits files in fits_dir
        :return:
        """
        getLogger(__name__).info('Calculating Ic, Ip, and Is')
        self.data = calculate_components(self.h5_files, self.component_dir, ncpu=self.ncpu, prior=self.prior,
                                         prior_sig=self.prior_sig, set_ip_zero=self.set_ip_zero, binned=self.binned,
                                         bin_size=self.bin_size, deadtime=self.deadtime, startt=self.startt,
                                         duration=self.duration, lucky=self.lucky, timestep=self.timestep,
                                         adi_mode=self.adi_mode)
        if self.drizzle:
            getLogger(__name__).info(f'Creating and saving component drizzles in {self.fits_dir}')
            if self.binned:
                components = ['ic_image', 'is_image']
            else:
                components = ['ic_image', 'is_image', 'ip_image']
            for i, component in enumerate(components):
                self.drizzle_components(component=component)
            self.write_fits(self.fits_dir)

    def write_fits(self, fits_dir, overwrite=True):
        """
        Writes and saves output fits files which will be either two dimensional (if timestep=0) or three dimensional
        with a time axis if timestep > 0. Temporal cibes are designed to interface with ADI if desired
        :param fits_dir: str, directory to save fits files to
        :param overwrite: bool, if True will overwrite existing fits file in the fits_dir
        :return:
        """
        if self.timestep and not self.binned:
            cubes = [self.ic_cube, self.is_cube, self.ip_cube]
            components = ['Ic', 'Is', 'Ip']
            for i, cube in enumerate(cubes):
                hdul = fits.HDUList([fits.PrimaryHDU()])
                for j, data in enumerate(cube):
                    hdul.append(fits.ImageHDU(name='cps', data=data))
                hdul.writeto(fits_dir + components[i] + '_drizzle', overwrite=overwrite)
                print(('FITS file {} saved'.format(fits_dir + components[i] + '_drizzle')))
        elif not self.binned:
            cubes = [np.sum(self.ic_cube, axis=0), np.sum(self.is_cube, axis=0), np.sum(self.ip_cube, axis=0)]
            components = ['Ic', 'Is', 'Ip']
            for i, cube in enumerate(cubes):
                hdul = fits.HDUList([fits.PrimaryHDU()])
                hdul.append(fits.ImageHDU(name='cps', data=cube))
                hdul.writeto(fits_dir + components[i] + '_drizzle.fits', overwrite=overwrite)
                print(('FITS file {} saved'.format(fits_dir + components[i] + '_drizzle')))
        elif self.binned and not self.timestep:
            i_c = np.sum(self.ic_cube, axis=0)
            i_s = np.sum(self.is_cube, axis=0)
            cubes = [i_c, i_s, i_c / i_s]
            components = ['Ic', 'Is', 'Ic_div_Is']
            for i, cube in enumerate(cubes):
                hdul = fits.HDUList([fits.PrimaryHDU()])
                hdul.append(fits.ImageHDU(name='cps', data=cube))
                hdul.writeto(fits_dir + components[i] + '_drizzle.fits', overwrite=overwrite)
                print(('FITS file {} saved'.format(fits_dir + components[i] + '_drizzle.fits')))
        elif self.timestep and self.binned:
            cubes = [self.ic_cube, self.is_cube, self.ic_cube / self.is_cube]
            components = ['Ic', 'Is', 'Ic_div_Is']
            for i, cube in enumerate(cubes):
                hdul = fits.HDUList([fits.PrimaryHDU()])
                for j, data in enumerate(cube):
                    hdul.append(fits.ImageHDU(name='cps', data=data))
                hdul.writeto(fits_dir + components[i] + '_drizzle.fits', overwrite=overwrite)
                print(('FITS file {} saved'.format(fits_dir + components[i] + '_drizzle')))
        else:
            print('Not a Valid Mode')

    def drizzle_components(self, pixfrac=0.5, component=''):
        """
        Drizzles specified Ic, Is and (if binned=False) Ip components to later be saved to fits files
        :param pixfrac:
        :param component:
        :return:
        """
        pt = Photontable(self.h5_files[0])
        if pt.query_header('E_PLTSCL') < 1e-4:
            target, pltscl = pt.query_header('OBJECT'), pt.query_header('E_PLTSCL')
        else:
            target, pltscl = pt.query_header('OBJECT'), (pt.query_header('E_PLTSCL') * u.arcsec).to(u.deg).value
        try:
            coords = SkyCoord(pt.query_header('D_IMRRA').values[0], pt.query_header('D_IMRDEC').values[0],
                              unit=(u.hourangle, u.deg))
        except (IndexError, KeyError) as e:
            coords = SkyCoord.from_name(target)
        ref_wcs = get_canvas_wcs(target, coords=coords, platescale=pltscl)
        n_times = (len(self.data) * len(self.data[0]['wcs_seq']))
        dithhyper = np.zeros((n_times, 500, 500), dtype=np.float32)
        for pos, dither_pos in enumerate(self.data):
            for wcs_i, wcs_sol in enumerate(dither_pos['wcs_seq']):
                cps = dither_pos[component][wcs_i]
                if self.binned:
                    cps[cps > 1e4] = 0
                    cps[cps < 1] = 0
                driz = Drizzle(outwcs=ref_wcs, pixfrac=pixfrac)
                inwht = cps.astype(bool).astype(int)
                wcs_sol.pixel_shape = (146, 140)
                driz.add_image(cps, wcs_sol, inwht=inwht, in_units='cps')
                # for a single wcs timestep
                dithhyper[pos * len(dither_pos['wcs_seq']) + wcs_i, :,
                :] = driz.outsci  # sum all counts in same exposure bi
        if component == 'ic_image':
            self.ic_cube = dithhyper
        elif component == 'is_image':
            self.is_cube = dithhyper
        elif component == 'ip_image':
            self.ip_cube = dithhyper
    

def calculate_components(fn_list, component_dir, ncpu=1, prior=None, prior_sig=None, set_ip_zero=False, binned=False,
                         bin_size=None, read_noise=0.0, deadtime=None, startt=None, duration=None, lucky=False,
                         timestep=None, adi_mode=False):
    """
    wrapper for running the binned or binfree SSD code
    :param fn_list: list of h5 files
    :param component_dir: location to save the component .npy files
    :param ncpu: number of cpus to use
    :param prior: prior for the binfree SSD analysis - defaults to np.nan for all values
    :param prior_sig: std of the prior for the binfree SSD analysis - defaults to np.nan for all values
    :param set_ip_zero: if binned=False and True will run the binfree SSD enforcinf Ip to 0. Ignored if binned=True
    :param binned: If True will run binned SSD otherwise will run binfree SSD
    :param bin_size: Size of time bins for binned SSD analysis. Ignored if binned=False
    :param read_noise: option to add poisson noise to the SSD calculation to simulate results for non-MKID detectors
    like EMCCDs
    :return: None
    """
    if binned:
        p = Pool(ncpu)
        f = partial(binned_ssd, save=True, save_dir=component_dir, bin_size=bin_size, read_noise=read_noise,
                    use_lucky=lucky, timestep=timestep, startt=startt, duration=duration, adi_mode=adi_mode)
        data = p.map(f, fn_list)
        data.sort(key=lambda k: fn_list.index(k['file']))
        return data
    else:
        p = Pool(ncpu)
        f = partial(binfree_ssd, save=True, save_dir=component_dir, IptoZero=set_ip_zero, prior=prior,
                    prior_sig=prior_sig, deadtime=deadtime, startt=startt, duration=duration, use_lucky=lucky,
                    timestep=timestep, adi_mode=adi_mode)
        data = p.map(f, fn_list)
        data.sort(key=lambda k: fn_list.index(k['file']))
        return data


def binned_ssd(fn, save=True, save_dir='', bin_size=0.01, read_noise=0.0, use_lucky=False, timestep=None,
               adi_mode=False, startt=None, duration=None):
    """
    function for running binned SSD

    :param fn: file to run binned SSD on
    :param save: if True will save the result as a .npy file to save_dir. The name of the file will correspond to the
    UNIX timestamp name given to the h5 file.
    :param save_dir: directory where to save the Ic and Is .npy arrays
    :param name_ext: optional name extension to apply to the end of the file to be used as an identifier
    :param bin_size: time bin size for the SSD (seconds)
    :return: Ic and Is images
    """
    pt = Photontable(fn)
    if timestep:
        wcs_times = pt.start_time + np.arange(startt, startt + duration, timestep)  # This is in unixtime
    else:
        wcs_times = [pt.start_time + startt]
    wcs = pt.get_wcs(derotate=not adi_mode, sample_times=wcs_times)
    ntimes = len(wcs_times)
    Ic_image = np.zeros((ntimes, 140, 146))
    Is_image = np.zeros((ntimes, 140, 146))
    if os.path.exists(save_dir + 'Ic/' + fn[-13:-3] + '.npy'):
        print('Ic, Is, and Ip already calculated for {}'.format(fn))
        Ic_image = np.load(save_dir + 'Ic/' + fn[-13:-3] + '.npy')
        Is_image = np.load(save_dir + 'Is/' + fn[-13:-3] + '.npy')
    else:
        for j in range(ntimes):
            print(f'Running time {j} of {fn}')
            num_pix = len([i for i in pt.resonators(exclude=PROBLEM_FLAGS, pixel=True)])
            bar = ProgressBar(maxval=num_pix).start()
            bari = 0
            if use_lucky:
                use_cubes = lucky_images(pt, bin_size)
            with pt.needed_ram():
                for pix, resID in pt.resonators(exclude=PROBLEM_FLAGS, pixel=True):
                    ts = pt.query(start=wcs_times[j] if startt > 0 else None,
                                  intt=timestep if timestep > 0 else None, resid=resID, column='time')
                    if len(ts) > 0:
                        if use_lucky:
                            lc_counts = np.array([int(c[pix[0], pix[1]]) for c in use_cubes])
                        else:
                            lc_counts, lc_intensity, lc_times, _ = getLightCurve(ts / 10 ** 6, effExpTime=bin_size)
                        mu = np.mean(lc_counts)
                        var = np.var(lc_counts)
                        if read_noise and read_noise != 0:
                            # for poisson noise
                            # noise = np.random.poisson(read_noise, len(lc_counts))
                            # lc_counts = np.array([lc_counts[i] + noise[i] for i in range(len(lc_counts))])
                            noise = np.random.normal(loc=read_noise, size=len(lc_counts))
                            noise = np.around(noise).astype(int)
                            lc_counts = np.array([lc_counts[i] + noise[i] for i in range(len(lc_counts))])
                            np.clip(lc_counts, 0, 1e5, out=lc_counts)
                        try:
                            IIc, IIs = np.asarray(muVar_to_IcIs(mu, var, bin_size)) * bin_size
                        except ValueError:
                            IIc = mu / 2  # just create a reasonable seed
                            IIs = mu - IIc
                        Ic, Is, res = maxBinMRlogL(lc_counts, Ic_guess=IIc, Is_guess=IIs, effExpTime=bin_size)
                        if np.isfinite(Ic) and np.isfinite(Is):
                            Ic_image[j][pix[0]][pix[1]] = Ic
                            Is_image[j][pix[0]][pix[1]] = Is
                        else:
                            pass
                    else:
                        pass
                    bari += 1
                    bar.update(bari)
            bar.finish()
    if save:
        try:
            np.save(save_dir + 'Ic/' + fn[-13:-3], Ic_image)
            np.save(save_dir + 'Is/' + fn[-13:-3], Is_image)
        except FileNotFoundError:
            os.mkdir(save_dir + 'Ic/')
            os.mkdir(save_dir + 'Is/')
            np.save(save_dir + 'Ic/' + fn[-13:-3], Ic_image)
            np.save(save_dir + 'Is/' + fn[-13:-3], Is_image)
    return {'file': fn, 'ic_image': Ic_image, 'is_image': Is_image, 'wcs_seq': wcs}


def binfree_ssd(fn, save=True, save_dir='', IptoZero=False, prior=None, prior_sig=None, deadtime=None,
                use_lucky=True, startt=None, duration=None, timestep=None, adi_mode=False):
    """
    Runs the binfree SSD
    :param fn: file to run binned SSD on
    :param save: if True will save the result as a .npy file to save_dir. The name of the file will correspond to the
    UNIX timestamp name given to the h5 file.
    :param save_dir: directory where to save the Ic and Is and Ip (if desired) .npy arrays
    :param IptoZero: If True will set Ip to 0 for the determination of Ic and Is
    :param prior: prior for the SSD analysis - defaults to np.nan for all values
    :param prior_sig: std of the prior for the SSD analysis - defaults to np.nan for all values
    :param name_ext: optional name extension to apply to the end of the file to be used as an identifier
    :param deadtime:
    :return: Ic, Is and Ip images
    """
    if prior is None:
        use_prior = None
        use_prior_sig = None
    pt = Photontable(fn)
    if timestep:
        wcs_times = pt.start_time + np.arange(startt, startt + duration, timestep)  # This is in unixtime
    else:
        wcs_times = [pt.start_time + startt]
    wcs = pt.get_wcs(derotate=not adi_mode, sample_times=wcs_times)
    ntimes = len(wcs_times)
    Ic_image = np.zeros((ntimes, 140, 146))
    Is_image = np.zeros((ntimes, 140, 146))
    Ip_image = np.zeros((ntimes, 140, 146))
    if os.path.exists(save_dir + 'Ic/' + fn[-13:-3] + '.npy'):
        print('Ic, Is, and Ip already calculated for {}'.format(fn))
        Ic_image = np.load(save_dir + 'Ic/' + fn[-13:-3] + '.npy')
        Is_image = np.load(save_dir + 'Is/' + fn[-13:-3] + '.npy')
        Ip_image = np.load(save_dir + 'Ip/' + fn[-13:-3] + '.npy')
    else:
        for j in range(ntimes):
            print(f'Running time {j} of {fn}')
            if use_lucky:
                use_ranges = get_lucky(pt, 15, 30, startt=startt, duration=duration, bin_width=0.1, percent_best=0.3)
            num_pix = len([i for i in pt.resonators(exclude=PROBLEM_FLAGS, pixel=True)])
            bar = ProgressBar(maxval=num_pix).start()
            bari = 0
            for pix, resID in pt.resonators(exclude=PROBLEM_FLAGS, pixel=True):
                if use_lucky:
                    all_ts = pt.query(start=startt, intt=duration, resid=resID, column='time')
                    dt = np.array([])
                    for i, rnge in enumerate(use_ranges):
                        idxs = np.where(np.logical_and(all_ts > rnge[0] * 1e6, all_ts < rnge[1] * 1e6))[0]
                        ts = all_ts[idxs]
                        dt_new = np.diff(np.sort(ts)) / 1e6
                        dt = np.append(dt, dt_new)
                else:
                    ts = pt.query(start=wcs_times[j], intt=timestep if timestep else duration, resid=resID,
                                  column='time')
                    ts = np.sort(ts)
                    dt = np.diff(ts) / 1e6
                if deadtime and len(dt) > 0:
                    dt = np.array([t if t > deadtime else t + dt[i + 1] for (i, t) in enumerate(dt[:-1])])
                if prior:
                    use_prior = [[prior[0][0][pix[0], pix[1]] if np.any(~np.isnan(prior[0][0])) else np.nan][0], np.nan,
                                 np.nan]
                    use_prior_sig = [
                        [prior_sig[0][0][pix[0], pix[1]] if np.any(~np.isnan(prior_sig[0][0])) else np.nan][0],
                        np.nan, np.nan]
                if len(dt) > 0:
                    model = optimize_IcIsIr2(dt, prior=use_prior, prior_sig=use_prior_sig, forceIp2zero=IptoZero,
                                             deadtime=deadtime if deadtime else 1.e-5)
                    Ic, Is, Ip = model.x
                    Ic_image[j][pix[0]][pix[1]] = Ic
                    Is_image[j][pix[0]][pix[1]] = Is
                    Ip_image[j][pix[0]][pix[1]] = Ip
                bari += 1
                bar.update(bari)
            bar.finish()
    if save:
        try:
            np.save(save_dir + 'Ic/' + fn[-13:-3], Ic_image)
            np.save(save_dir + 'Is/' + fn[-13:-3], Is_image)
        except FileNotFoundError:
            os.mkdir(save_dir + 'Ic/')
            os.mkdir(save_dir + 'Is/')
            os.mkdir(save_dir + 'Ip/')
            np.save(save_dir + 'Ic/' + fn[-13:-3], Ic_image)
            np.save(save_dir + 'Is/' + fn[-13:-3], Is_image)
        if not IptoZero:
            np.save(save_dir + 'Ip/' + fn[-13:-3], Ip_image)
    return {'file': fn, 'ic_image': Ic_image, 'is_image': Is_image, 'ip_image': Ip_image, 'wcs_seq': wcs}


def quickstack(file_path, make_fits=False, axes=None, v_max=30000):
    """
    plotting function to stack all numpy 3D arrays in a given file_path
    :param file_path: location of the numpy files
    :param make_fits: if True will save a fits file containing the stacked image data
    :param axes: axes on which to put the plot
    :param v_max: maximum value for the plot colorbar
    :return: matplotlib Axes object
    """
    stacked_im = np.zeros((140, 146))
    for i, fn in enumerate(os.listdir(file_path)):
        if fn.endswith('.npy'):
            data = np.load(file_path + fn)
            stacked_im += data
    if make_fits:
        hdu = fits.PrimaryHDU(stacked_im)
        hdu.writeto(file_path + 'stacked.fits', overwrite=True)
    else:
        im = axes.imshow(stacked_im, vmin=0, vmax=v_max, cmap='magma')  # norm=LogNorm(vmin=0.1, vmax=100), cmap=my_cmap)
        plt.colorbar(im, ax=axes)
        axes.set_title(file_path[-3:-1] + ' stack', fontsize=10)
        return axes


def get_canvas_wcs(target, coords=None, platescale=None):
    npixx = 500
    npixy = 500
    if coords is None:
        coords = SkyCoord.from_name(target)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = np.array([npixx/2., npixy/2.])
    wcs.wcs.crval = [coords.ra.deg, coords.dec.deg]
    wcs.wcs.ctype = ["RA--TAN", "DEC-TAN"]
    wcs.pixel_shape = (npixx, npixy)
    wcs.wcs.pc = np.eye(2)
    wcs.wcs.cdelt = [platescale, platescale] if platescale else [2.888e-6,
                                                                 2.888e-6]  # corresponds to a 10.4 mas platescale
    wcs.wcs.cunit = ["deg", "deg"]
    getLogger(__name__).debug(wcs)
    return wcs


def get_annulus_pixels(pt, annulus_idx):
    xCon, yCon = pt.query_header('E_CONEXX'), pt.query_header('E_CONEXY')
    slopes = (pt.query_header('E_DPDCX'), pt.query_header('E_DPDCY'))
    ref_pix = (pt.query_header('E_PREFX'), pt.query_header('E_PREFY'))
    ref_con = (pt.query_header('E_CXREFX'), pt.query_header('E_CXREFY'))
    center_pix = CONEX2PIXEL(xCon, yCon, slopes, ref_pix, ref_con)
    n_apers, _ = get_num_apertures(0.5, annulus_idx + 1)
    angular_size = 2 * np.arctan(0.5 / (annulus_idx + 1))
    centers = []
    shape = np.shape(pt.beamImage)
    for i in range(n_apers):
        aperture_center = get_aperture_center(0, i * angular_size, center_pix, annulus_idx + 1)
        if int(np.round(aperture_center[0])) < 0 or int(np.round(aperture_center[0])) >= shape[0] or int(
                np.round(aperture_center[1])) < 0 or int(np.round(aperture_center[1])) >= shape[1]:
            pass
        else:
            centers.append((int(np.round(aperture_center[0])), int(np.round(aperture_center[1]))))
    return centers


def plot_intensity_histogram(data, object_name='object', N=400, span=[0, 300], axes=None, fit_poisson=True):
    """
    for a given 3D data array will plot the count rate histograms and perform a modified rician fit
    :param data: 3D data array
    :param object_name: str name of the objectget
    :param axes: axes on which to make the plot
    :param fit_poisson: if True will fit a poisson distribution to the histogram in addiiton to a MR
    :return: matplotlib Axes object
    """

    histS_o, binsS_o = histogramLC(data, centers=True, N=N, span=span)
    histS = histS_o[1:]
    binsS = binsS_o[1:]
    guessIc = np.mean(data) * 2
    guessIs = np.mean(data) * 1.1
    guessLam = np.mean(data)

    fitIc, fitIs = pdfs.fitMR(binsS, histS, guessIc, guessIs)
    fitMR = pdfs.modifiedRician(binsS, fitIc, fitIs)
    fitLam = pdfs.fitPoisson(binsS, histS, guessLam)
    fitPoisson = pdfs.poisson(binsS, fitLam)

    axes.step(binsS, histS, color='grey', label=r'Histogram of intensities', where='mid')
    axes.plot(binsS, fitMR, color='black', linestyle='-.',
              label=r'MR fit to histogram: Ic=%2.2f, Is=%2.2f' % (fitIc, fitIs))
    if fit_poisson:
        axes.plot(binsS, fitPoisson, color='red', linestyle='-.',
                  label=r'Poisson fit to histogram' )
    axes.set_xlabel('counts/20ms')
    axes.set_ylabel('Probability')
    axes.set_title(object_name)
    axes.legend(prop={'size': 6}, loc='upper right')
    # axes.set_ylim(0, 0.2)
    return axes


def lucky_images(pt, bin_size):
    """

    :param pt:
    :param bin_size:
    :return:
    """
    plt.clf()
    temporal_cube = pt.get_fits(cube_type='time', bin_width=bin_size, rate=False)[1].data
    median_flux = np.median(np.sum(temporal_cube, axis=(1, 2)))
    plt.hist(np.sum(temporal_cube, axis=(1, 2)), bins=20, histtype='step')
    std = np.std(np.sum(temporal_cube, axis=(1, 2)))
    plt.axvline(x=median_flux, color='r', linestyle='--')
    # plt.savefig('/data/steiger/MEC/20220222/SSD/HIP36152/binned_20ms_lucky/test.pdf')
    use_cubes = []
    for i, cube in enumerate(temporal_cube):
        if np.sum(cube) > std + median_flux:
            # if np.sum(cube) > median_flux:
            continue
        else:
            use_cubes.append(cube)
    return use_cubes
