'''
Code to Perform Stochastic Speckle Discrimination (SSD) on MEC data

Example Usage:

ssd = SSDAnalyzer(h5_dir=h5_folder, fits_dir=save_folder + 'fits/', component_dir=save_folder, plot_dir=save_folder,
                  ncpu=10, save_plot=True, set_ip_zero=False, binned=False)
ssd.run_ssd()

'''
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import os
from mkidanalysis.speckle.binned_rician import muVar_to_IcIs, maxBinMRlogL, getLightCurve
from mkidanalysis.speckle.binfree_rician import optimize_IcIsIr2
from matplotlib import gridspec
from mkidcore.corelog import getLogger
from progressbar import ProgressBar
from mkidcore.pixelflags import FlagSet
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
warnings.filterwarnings("ignore")
getLogger().setLevel('INFO')


PROBLEM_FLAGS = ('pixcal.hot', 'pixcal.cold', 'pixcal.dead', 'beammap.noDacTone', 'wavecal.bad',
                 'wavecal.failed_validation', 'wavecal.failed_convergence', 'wavecal.not_monotonic',
                 'wavecal.not_enough_histogram_fits', 'wavecal.no_histograms',
                 'wavecal.not_attempted')

class SSDAnalyzer:
    """Class for running binned or binfree SSD analysis on MKID data (calibrated h5 files)"""
    def __init__(self, h5_dir='', fits_dir='', component_dir='',  ncpu=1, prior=None, prior_sig=None, set_ip_zero=False,
                 binned=True, drizzle=True, bin_size=0.01, read_noise=None):
        self.h5_dir = h5_dir
        self.fits_dir = fits_dir
        self.component_dir = component_dir
        self.ncpu = ncpu
        self.set_ip_zero = set_ip_zero
        self.prior = prior
        self.prior_sig = prior_sig
        self.binned = binned
        self.h5_files = []
        self.drizzle = drizzle
        self.bin_size = bin_size
        self.read_noise = read_noise
        print(f'read noise is {self.read_noise}')
        for i, fn in enumerate(os.listdir(h5_dir)):
            if fn.endswith('.h5') and fn.startswith('1'):
                self.h5_files.append(h5_dir + fn)
        self.photontables = [Photontable(h5) for h5 in self.h5_files]
        self.intt = self.photontables[0].query_header('EXPTIME')
        self.wcs_list = [pt.get_wcs(derotate=True, wcs_timestep=self.intt) for pt in self.photontables]

    def run_ssd(self):
        """
        runs the SSD code on every h5 file in the h5_dir. Creates fits files which are saved in the fits_dir
        and adds the Ic, Is, and Ip images onto the end of the fits file.

        Also saves a drizzled Ic, Is, and Ip image in the fits_dir if drizzle=True.
        """
        getLogger(__name__).info('Calculating Ic, Ip, and Is')
        calculate_components(self.h5_dir, self.component_dir, ncpu=self.ncpu, prior=self.prior, prior_sig=self.prior_sig,
                      set_ip_zero=self.set_ip_zero, binned=self.binned, bin_size=self.bin_size)
        getLogger(__name__).info('Initializing fits files')
        initialize_fits(self.h5_dir, self.fits_dir, ncpu=self.ncpu, intt=self.intt)
        if self.binned:
            component_types = ['Ic', 'Is']
            for i, ct in enumerate(component_types):
                update_fits(self.component_dir + ct + '/', self.fits_dir, ext=ct)
        else:
            component_types = ['Ic', 'Is', 'Ip']
            for i, ct in enumerate(component_types):
                update_fits(self.component_dir + ct + '/', self.fits_dir, ext=ct)
        if self.drizzle:
            getLogger(__name__).info(f'Creating and saving component drizzles in {self.fits_dir}')
            self.save_drizzles()

    def save_drizzles(self):
        """wrapper for drizzle_components"""
        if self.binned or self.set_ip_zero:
            components = ['Ic', 'Is', 'SCIENCE']
        else:
            components = ['Ic', 'Is', 'Ip', 'SCIENCE']
        for ct in components:
            self.drizzle_components(plot_type=ct)

    def drizzle_components(self, plot_type='Ic'):
        """
        plots and saves the result of running each of the Ic, Is, (and Ip if relevant) frames through the Drizzle
        algorithm. Files are saved in a fits file format in self.fits_dir

        https://github.com/spacetelescope/drizzle

        :param plot_type: component type to drizzle - 'Ic', 'Is', or 'Ip'
        :return: None
        """
        infiles = []
        for i, fn in enumerate(os.listdir(self.fits_dir)):
            if fn.endswith('.fits') and fn.startswith('1'):
                infiles.append(self.fits_dir + fn)
        target = fits.open(infiles[0])[0].header['OBJECT']
        ref_wcs = get_canvas_wcs(target)
        driz = Drizzle(outwcs=ref_wcs, pixfrac=0.5, wt_scl='')
        for i, infile in enumerate(infiles):
            try:
                imlist = fits.open(infile)
                try:
                    image = imlist[plot_type].data
                except IndexError:
                    getLogger(__name__).info(f'Fits files in {self.fits_dir} have no Ic, Is, and Ip data attached')
                    return
                flags = imlist['FLAGS'].data
                file_flags = self.photontables[i].flags
                bad_bit_mask = file_flags.bitmask(PROBLEM_FLAGS)
                weight_arr = np.ones((140,146))
                for (x, y), flag in np.ndenumerate(flags):
                    if flag & bad_bit_mask != 0:
                        weight_arr[x, y] = 0
                    else:
                        weight_arr[x, y] = 1
                cps = image/self.intt
                if plot_type == 'Is':
                    tcps = cps.T
                    is_lim = np.nanmedian(tcps[tcps != 0])
                    tcps[tcps > (100 * is_lim)] = 0
                    bad_idx = np.where(tcps <= 0)
                    weight_arr[bad_idx] = 0
                if plot_type == 'Ic':
                    tcps = cps.T
                    ic_lim = np.nanmedian(tcps[tcps != 0])
                    tcps[tcps > (100 * ic_lim)] = 0
                    bad_idx = np.where(tcps <= 0)
                    weight_arr[bad_idx] = 0
                cps[np.isnan(cps)] = 0
                image_wcs = WCS(imlist[0].header)
                image_wcs.pixel_shape = (146, 140)
                driz.add_image(cps.T, inwcs=image_wcs, in_units='cps', inwht=weight_arr)
            except ValueError:
                pass
        driz.write(self.fits_dir + plot_type + '_drizzle.fits')
        return None


def initialize_fits(in_file_path, out_file_path, intt, ncpu=1):
    """
    wrapper function for running multiprocessing with init_fits
    :param in_file_path: location fo the input h5 files
    :param out_file_path: location to save the output fits files to
    :param ncpu: number of cores to ue for multiprocessing
    :return: None
    """
    fn_list = []
    for i, fn in enumerate(os.listdir(in_file_path)):
        if os.path.exists(out_file_path + fn[0:-3] + '.fits'):
            getLogger(__name__).info(
                f'fits file {fn[0:-3]}.fits already exists in the specified directory, not creating a new one')
        if fn.endswith('.h5') and not os.path.exists(out_file_path + fn[0:-3] + '.fits'):
            fn_list.append(in_file_path + fn)
    p = Pool(8)
    f = partial(init_fits, out_file_path=out_file_path, intt=intt)
    p.map(f, fn_list)


def init_fits(fn, out_file_path, intt):
    """
    creates a fits file from h5 file, fn by calling Photontable.get_fits().
    :param fn: h5 file to make a fits file from
    :param out_file_path: directory to save the fits file to
    :return: None
    """
    pt = Photontable(fn)
    with pt.needed_ram():
        hdu = pt.get_fits(rate=False, exclude_flags=PROBLEM_FLAGS)
        hdu.writeto(out_file_path + fn[-13:-3] + '.fits')
        hdu.close()
        getLogger(__name__).info(f'Initialized fits file for {out_file_path + fn[-13:-3]}.fits')


def calculate_components(data_path, component_dir, ncpu=1, prior=None, prior_sig=None, set_ip_zero=False, binned=False,
                         bin_size=None, read_noise=0.0):
    """
    wrapper for running the binned or binfree SSD code

    :param data_path: location of the h5 files
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
    fn_list = []
    for i, fn in enumerate(os.listdir(data_path)):
        if fn.endswith('.h5') and fn.startswith('1'):
            fn_list.append(data_path + fn)
    if binned:
        p = Pool(ncpu)
        f = partial(binned_ssd, save=True, save_dir=component_dir, bin_size=bin_size, read_noise=read_noise)
        p.map(f, fn_list)
    else:
        p = Pool(ncpu)
        f = partial(binfree_ssd, save_dir=component_dir, IptoZero=set_ip_zero, prior=[prior], prior_sig=[prior_sig])
        p.map(f, fn_list)


def binned_ssd(fn, save=True, save_dir='', name_ext='', bin_size=0.01, read_noise=0.0):
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
    Ic_image = np.zeros((140, 146))
    Is_image = np.zeros((140, 146))
    if os.path.exists(save_dir + 'Ic/' + fn[-13:-3] + name_ext + '.npy'):
        getLogger(__name__).info('Ic and Is already calculated for {}'.format(fn))
        return
    bar = ProgressBar(maxval=20439).start()
    bari = 0
    with pt.needed_ram():
        for pix, resID in pt.resonators(exclude=PROBLEM_FLAGS, pixel=True):
            ts = pt.query(pixel=pix, column='time')
            if len(ts) > 0:
                lc_counts, lc_intensity, lc_times = getLightCurve(ts/10**6, effExpTime=bin_size)
                mu = np.mean(lc_counts)
                var = np.var(lc_counts)
                if read_noise and read_noise !=0:
                    # for poisson noise
                    # noise = np.random.poisson(read_noise, len(lc_counts))
                    # lc_counts = np.array([lc_counts[i] + noise[i] for i in range(len(lc_counts))])
                    noise = np.random.normal(loc=read_noise, size=len(lc_counts))
                    noise=np.around(noise).astype(int)
                    lc_counts = np.array([lc_counts[i] + noise[i] for i in range(len(lc_counts))])
                    np.clip(lc_counts, 0, 1e5, out=lc_counts)
                try:
                    IIc, IIs = np.asarray(muVar_to_IcIs(mu, var, bin_size)) * bin_size
                except ValueError:
                    IIc = mu / 2  # just create a reasonable seed
                    IIs = mu - IIc
                Ic, Is, res = maxBinMRlogL(lc_counts, Ic_guess=IIc, Is_guess=IIs)
                Ic_image[pix[0]][pix[1]] = Ic
                Is_image[pix[0]][pix[1]] = Is
            else:
                Ic_image[pix[0]][pix[1]] = 0
                Is_image[pix[0]][pix[1]] = 0
            bari += 1
            bar.update(bari)
        bar.finish()
    if save:
        try:
            np.save(save_dir + 'Ic/' + fn[-13:-3] + name_ext, Ic_image.T)
            np.save(save_dir + 'Is/' + fn[-13:-3] + name_ext, Is_image.T)
        except FileNotFoundError:
            os.mkdir(save_dir + 'Ic/')
            os.mkdir(save_dir + 'Is/')
            np.save(save_dir + 'Ic/' + fn[-13:-3] + name_ext, Ic_image.T)
            np.save(save_dir + 'Is/' + fn[-13:-3] + name_ext, Is_image.T)
    return Ic_image, Is_image


def binfree_ssd(fn, save=True, save_dir='', IptoZero=False, prior=None, prior_sig=None, name_ext=''):
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
    :return: Ic, Is and Ip images
    """
    if prior is None:
        prior = [np.nan, np.nan, np.nan]
        prior_sig = [np.nan, np.nan, np.nan]
    pt = Photontable(fn)
    Ic_image = np.zeros((140, 146))
    Is_image = np.zeros((140, 146))
    Ip_image = np.zeros((140, 146))
    if os.path.exists(save_dir + 'Ic/' + fn[-13:-3] + name_ext + '.npy'):
        getLogger(__name__).info('Ic, Is, and Ip already calculated for {}'.format(fn))
        return
    bar = ProgressBar(maxval=20439).start()
    bari = 0
    for pix, resID in pt.resonators(exclude=PROBLEM_FLAGS, pixel=True):
        ts = pt.query(pixel=pix, column='Time')
        ts = np.sort(ts)
        dt = np.diff(ts) / 10e6
        use_prior = [[prior[0][0][pix[0], pix[1]] if np.any(~np.isnan(prior[0][0])) else np.nan][0], np.nan, np.nan]
        use_prior_sig = [[prior_sig[0][0][pix[0], pix[1]] if np.any(~np.isnan(prior_sig[0][0])) else np.nan][0], np.nan, np.nan]
        if len(dt) > 0:
            model = optimize_IcIsIr2(dt, prior=use_prior, prior_sig=use_prior_sig, forceIp2zero=IptoZero)
            Ic, Is, Ip = model.x
            Ic_image[pix[0]][pix[1]] = Ic
            Is_image[pix[0]][pix[1]] = Is
            Ip_image[pix[0]][pix[1]] = Ip
        bari += 1
        bar.update(bari)
    bar.finish()
    if save:
        if not os.path.isfile(save_dir + 'Ic/'):
            os.mkdir(save_dir + 'Ic/')
            os.mkdir(save_dir + 'Is/')
            if not IptoZero:
                os.mkdir(save_dir + 'Ip/')
        np.save(save_dir + 'Ic/' + fn[-13:-3] + name_ext, Ic_image)
        np.save(save_dir + 'Is/' + fn[-13:-3] + name_ext, Is_image)
        if not IptoZero:
            np.save(save_dir + 'Ip/' + fn[-13:-3] + name_ext, Ip_image)
    return Ic_image, Is_image, Ip_image


def update_fits(data_file_path, fits_file_path, ext='UNKNOWN'):
    """
    takes a series of 3D .npy arrays in data_file_path and appends them to the end of the appropriate fits file
    in fits_file_path
    :param data_file_path: location of the .npy arrays
    :param fits_file_path: location of the fits files
    :param ext: EXTNAME of HDU to be added to the header of the HDU where the data is being added
    :return:
    """
    getLogger(__name__).info('Adding SSD data to fits files')
    for i, fn in enumerate(os.listdir(data_file_path)):
        for j, fit in enumerate(os.listdir(fits_file_path)):
            if fn.endswith('.npy'):
                if fit.endswith('.fits'):
                    if fit[0:-5] == fn[0:-4]:
                        try:
                            hdul = fits.open(fits_file_path+fit)
                            data = np.load(data_file_path + fn)
                            hdr = fits.Header()
                            hdu = fits.ImageHDU(data=data, header=hdr, name=ext)
                            hdul.append(hdu)
                            hdul.writeto(fits_file_path+fit, overwrite=True)
                        except OSError:
                            getLogger(__name__).info('Error trying to append ImageHdu {}'.format(fn[0:-4]))
                            pass


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


def get_canvas_wcs(target):
    npixx = 500
    npixy = 500
    coords = SkyCoord.from_name(target)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = np.array([npixx/2., npixy/2.])
    wcs.wcs.crval = [coords.ra.deg, coords.dec.deg]
    wcs.wcs.ctype = ["RA--TAN", "DEC-TAN"]
    wcs.pixel_shape = (npixx, npixy)
    wcs.wcs.pc = np.eye(2)
    wcs.wcs.cdelt = [2.888888889e-06, 2.88888889e-06] #corresponds to a 10.4 mas platescale
    wcs.wcs.cunit = ["deg", "deg"]
    getLogger(__name__).debug(wcs)
    return wcs


def plot_icisip(fn, Ic, Is, Ip):
    """
    Easy plotting function to compare Ic, Ip and Is
    :param fn: File path to save the finished plot to
    :param Ic: 2D array of Ic
    :param Is: 2D array of Ic
    :param Ip: 2D array of Ic
    :return:
    """
    fig, axes = plt.subplots(2, 2)
    im1 = axes[0][0].imshow(Ic, vmin=0, vmax=30)
    axes[0][0].set_title('Ic')
    im2 = axes[0][1].imshow(Is, vmin=0, vmax=30)
    axes[0][1].set_title('Is')
    im3 = axes[1][0].imshow(Ip, vmin=0, vmax=30)
    axes[1][0].set_title('Ip')
    plt.colorbar(im1, ax=axes[0][0])
    plt.colorbar(im2, ax=axes[0][1])
    plt.colorbar(im3, ax=axes[1][0])
    plt.tight_layout()
    plt.savefig(fn)


def plot_ic_div_is(Ic, Is, axes=None):
    """
    plotting function to display ic/is
    :param Ic: 3D array of Ic values
    :param Is: 3D array of Is values
    :param axes: axes on which to plot the result
    :return: matplotlib Axes object
    """
    div_image = np.zeros_like(Ic)
    for coords, idx in np.ndenumerate(Ic):
        x = coords[0]
        y = coords[1]
        if 0 < Ic[x][y] < 100 and 0.3 < Is[x][y]:
            div_image[x][y] = Ic[x][y]/Is[x][y]
        else:
            div_image[x][y] = 0
    my_cmap = matplotlib.cm.get_cmap('viridis')
    im = axes.imshow(div_image, vmin=0, vmax=100, cmap=my_cmap)# norm=LogNorm(vmin=0.1, vmax=100), cmap=my_cmap)
    plt.colorbar(im, ax=axes)
    axes.set_title('Ic/Is')
    return axes


def plot_intensity_histogram(data, object_name='object', N=400, span=[0, 300], axes=None, fit_poisson=True):
    """
    for a given 3D data array will plot the count rate histograms and perform a modified rician fit
    :param data: 3D data array
    :param object_name: str name of the object
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


def mr_hist_summary_plot(cube, object_location, field_location, box_size=2, N=40, Ic=None, Is=None, object_name='',
                         save_dir='', span=[0, 300]):
    """
    Plots an image of the data given by cube and also histograms of the counts contained in a box centered on
    object_location and field_location

    :param cube:
    :param object_location: location of the object of interest
    :param field_location: location of a speckle in the field to compare
    :param box_size: size of the aperture around the object center for which you would like to consider data (in pixels)
    :param Ic: 3D numpy array - If given will add a plot of Ic/Is to the plot
    :param Is: 3D numpy array - If given will add a plot of Ic/Is to the plot
    :param object_name: name of the object for the plot title (str)
    :param save_dir: directory in which to save the plot
    :param span:
    :return: None
    """
    x = object_location[0]
    y = object_location[1]
    a = field_location[0]
    b = field_location[1]
    xmin = int(x-box_size/2)
    xmax = int(x+box_size/2)
    ymin = int(y-box_size/2)
    ymax = int(y+box_size/2)
    amin = int(a-box_size/2)
    amax = int(a+box_size/2)
    bmin = int(b-box_size/2)
    bmax = int(b+box_size/2)

    figure = plt.figure()
    gs = gridspec.GridSpec(2, 2)
    axes_list = np.array([figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1]),
                          figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1])])

    cmap = plt.cm.viridis
    cmap.set_bad(color='r')
    try:
        im = axes_list[3].imshow(np.sum(cube['cube'], axis=2), cmap=cmap, vmin=0, vmax=3000, interpolation='nearest')
        object_intensities = np.sum(cube['cube'][xmin:xmax, ymin:ymax, :], axis=(0, 1))
        field_intensities = np.sum(cube['cube'][amin:amax, bmin:bmax, :], axis=(0, 1))
    except IndexError:
        im = axes_list[3].imshow(np.sum(cube, axis=2), cmap=cmap, vmin=0, vmax=8000, interpolation='nearest')
        object_intensities = np.sum(cube[xmin:xmax, ymin:ymax, :], axis=(0, 1))
        field_intensities = np.sum(cube[amin:amax, bmin:bmax, :], axis=(0, 1))

    plt.colorbar(im, ax=axes_list[3])
    axes_list[3].set_title('Total Intensity')

    plot_intensity_histogram(object_intensities, object_name=object_name, axes=axes_list[0], N=N, span=span)
    plot_intensity_histogram(field_intensities, object_name='Field', axes=axes_list[1], N=N, span=span)
    if Ic and Is:
        plot_ic_div_is(Ic, Is, axes=axes_list[2])
    circ_obj = Circle((object_location[1], object_location[0]), box_size, fill=False, color='red')
    circ_field = Circle((field_location[1], field_location[0]), box_size, fill=False, color='orange')
    axes_list[3].add_patch(circ_obj)
    axes_list[3].add_patch(circ_field)
    plt.tight_layout()
    plt.savefig(save_dir + 'mr_summary.pdf')


