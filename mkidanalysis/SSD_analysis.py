'''
Author: Alex Walter and Clint Bockstiegel - adapted for the MKIDPipeline by Sarah Steiger
Date: February 2020

Code to Perform SSD on MEC data

Example Usage:

ssd = SSDAnalyzer(h5_dir=h5_folder, fits_dir=save_folder + 'fits/', component_dir=save_folder, plot_dir=save_folder,
                  ncpu=10, save_plot=True, set_ip_zero=False, binned=False)
ssd.run_ssd()

'''


from mkidpipeline.speckle.binFreeRicianEstimate import *
# from DarknessPipeline.Examples.speckleStudy import lightCurves as lc
# import DarknessPipeline.Utils.pdfs as pdfs
from matplotlib import gridspec
from mkidcore.corelog import getLogger
from progressbar import *
from matplotlib.patches import Circle
from mkidpipeline.speckle.binned_rician import *
from multiprocessing import Pool
from functools import partial
from astropy.wcs import WCS
from drizzle.drizzle import Drizzle
import warnings
warnings.filterwarnings("ignore")
getLogger().setLevel('INFO')
from mkidpipeline.imaging.drizzler import *


class SSDAnalyzer:
    def __init__(self, h5_dir='', fits_dir='', component_dir='', plot_dir='', target_name='',  ncpu=1, save_plot=False,
                 prior=None, prior_sig=None, set_ip_zero=False, binned=False, startw=850, stopw=1375):
        self.h5_dir = h5_dir
        self.fits_dir = fits_dir
        self.component_dir = component_dir
        self.plot_dir = plot_dir
        self.ncpu = ncpu
        self.startw = startw
        self.stopw = stopw
        self.set_ip_zero = set_ip_zero
        self.save_plot = save_plot
        self.prior = prior
        self.prior_sig = prior_sig
        self.binned = binned
        self.target_name = target_name
        self.h5_files = []
        for i, fn in enumerate(os.listdir(h5_dir)):
            if fn.endswith('.h5') and fn.startswith('1'):
                self.h5_files.append(h5_dir + fn)
        self.intt = Photontable(self.h5_files[0]).getFromHeader('expTime')

    def run_ssd(self):
        """
        runs the binfree SSD code on every h5 file in the h5_dir. Creates fits files which are saved in the fits_dir
        and adds the Ic, Is, and Ip images onto the end of the fits file.
        Also creates and saves a summary plot of the stacked Ic, Is and Ip images as well as a drizzled Ip image and
        drizzled total intensity image in plot_dir if save_plot=True
        """
        getLogger(__name__).info('Calculating Ic, Ip, and Is')
        calc_ic_is_ip(self.h5_dir, self.component_dir, ncpu=self.ncpu, prior=self.prior, prior_sig=self.prior_sig,
                      set_ip_zero=self.set_ip_zero, binned=self.binned)
        getLogger(__name__).info('Initializing fits files')
        initialize_fits(self.h5_dir, self.fits_dir, startw=self.startw, stopw=self.stopw, ncpu=self.ncpu)
        if self.binned:
            component_types = ['Ic/', 'Is/']
            for i, ct in enumerate(component_types):
                update_fits(self.component_dir + ct, self.fits_dir)
        else:
            component_types = ['Ic/', 'Is/', 'Ip/']
            for i, ct in enumerate(component_types):
                update_fits(self.component_dir + ct, self.fits_dir)
        if self.save_plot:
            getLogger(__name__).info('Creating and saving summary plot in {}'.format(self.plot_dir))
            self.summary_plot()

    def summary_plot(self):
        """
        saves a summary plot of stacked Ic, Is and Ip, a the drizzled Ip image and a total intensity drizzle

        If binned= True will insetad save a drizzled Ic and a drizzled Is image
        """
        figure = plt.figure()
        gs = gridspec.GridSpec(3, 3)
        axes_list = np.array([figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1]),
                              figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1]), figure.add_subplot(gs[1, 2])])
        for ax in axes_list:
            ax.set_xticklabels('')
            ax.set_yticklabels('')
        if self.binned or self.set_ip_zero:
            component_types = ['Ic', 'Is']
            for i, ct in enumerate(component_types):
                quickstack(self.component_dir + ct + '/', axes=axes_list[i])
            plot_drizzle_binned(self.fits_dir, plot_type='Ic', axes=axes_list[3], target=self.target_name,
                                intt=self.intt)
            plot_drizzle_binned(self.fits_dir, plot_type='Is', axes=axes_list[4], target=self.target_name,
                                intt=self.intt)
        else:
            component_types = ['Ic', 'Is', 'Ip']
            for i, ct in enumerate(component_types):
                quickstack(self.component_dir + ct + '/', axes=axes_list[i])
            plot_drizzle(self.fits_dir, plot_type='Ip', axes=axes_list[3], target=self.target_name, intt=self.intt)
            plot_drizzle(self.fits_dir, plot_type='Intensity', axes=axes_list[4], target=self.target_name,
                         intt=self.intt)
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 8}
        matplotlib.rc('font', **font)
        plt.tight_layout()
        plt.savefig(self.plot_dir + 'summary_plot.pdf')


def initialize_fits(in_file_path, out_file_path, startw=850, stopw=1375, ncpu=1):
    """
    wrapper function for running multiprocessing with init_fits
    :param in_file_path: location fo the inpput h5 files
    :param out_file_path: location to save the output fits files to
    :param startw: start wavelength (nm)
    :param stopw: stop wavelength (nm)
    :param ncpu: number of cores to ue for multiprocessing
    :return: None
    """
    fn_list = []
    for i, fn in enumerate(os.listdir(in_file_path)):
        if os.path.exists(out_file_path + fn[0:-3] + '.fits'):
            getLogger(__name__).info(
                'fits file {} already exists in the specified directory, not creating a new one'
                .format(fn[0:-3] + '.fits'))
        if fn.endswith('.h5') and not os.path.exists(out_file_path + fn[0:-3] + '.fits'):
            fn_list.append(in_file_path + fn)
    p = Pool(ncpu)
    f = partial(init_fits, out_file_path=out_file_path, startw=startw, stopw=stopw)
    p.map(f, fn_list)


def init_fits(fn, out_file_path, startw=850, stopw=1375):
    """
    creates a fits file from h5 file, fn.
    :param fn: h5 file to make a fits file from
    :param out_file_path: directory to save the fits file to
    :param startw: start wavelength (nm)
    :param stopw: stop wavelength (nm)
    :return: None
    """
    obs = Photontable(fn)
    hdu = obs.getFits(wvlStart=startw, wvlStop=stopw, applyWeight=False)
    hdu.writeto(out_file_path + fn[-13:-3] + '.fits')
    getLogger(__name__).info('Initialized fits file for {}'.format(out_file_path + fn[-13:-3] + '.fits'))


def calc_ic_is_ip(data_path, component_dir, ncpu=1, prior=None, prior_sig=None, set_ip_zero=False, binned=False):
    """
    wrapper for running binfreeSSD
    :param data_path: location of the h5 files to run SSD on
    :param component_dir: directory to which you want the Ic, Is, and Ip data saved
    :param ncpu: number of cpus to use for parallel processing
    :return: None
    """
    fn_list = []
    for i, fn in enumerate(os.listdir(data_path)):
        if fn.endswith('.h5') and fn.startswith('1'):
            fn_list.append(data_path + fn)
    if binned:
        p = Pool(ncpu)
        f = partial(calculate_icis_binned, save=True, savefile=component_dir)
        p.map(f, fn_list)
    else:
        p = Pool(ncpu)
        f = partial(calculate_icisip, savefile=component_dir, IptoZero=set_ip_zero, prior=[prior], prior_sig=[prior_sig])
        p.map(f, fn_list)


def calculate_icisip(fn, savefile, IptoZero=False, save=True, prior=[np.nan, np.nan, np.nan],
                     prior_sig=[np.nan, np.nan, np.nan], name_ext=''):
    """
    Function to calculate Ic, Is, and Ip from an h5 file. Uses binfreeSSD
    :param fn: Filename of the observation for which you would like to find Ic, Is, and Ip
    :param IptoZero: If True will set Ip to 0 to calculate Ic and Is
    :param save: If True will save the Ic, Is, and Ip results to .npy files
    :param savefile: file path to save the .npy files. This function will add an _Ix.pdf extension for each parameter
    :param name_ext: name extension to add to the savefile
    :return: Ic, Is, and Ip images
    """

    obs = Photontable(fn)
    exclude_flags = pixelflags.PROBLEM_FLAGS
    Ic_image = np.zeros((140, 146))
    Is_image = np.zeros((140, 146))
    Ip_image = np.zeros((140, 146))
    if os.path.exists(savefile + 'Ic/' + fn[-13:-3] + name_ext + '.npy'):
        getLogger(__name__).info('Ic, Is, and Ip already calculated for {}'.format(fn))
        return
    bar = ProgressBar(maxval=20439).start()
    bari = 0
    for (x, y), dt in np.ndenumerate(obs.beamImage):
        if obs.flagMask(exclude_flags, (x, y)) and any(exclude_flags):
            continue
        ts = np.sort(obs.getPixelPhotonList(xCoord=x, yCoord=y)['Time'])
        dt = np.diff(ts)
        dt = dt / 10. ** 6
        use_prior = [[prior[0][0][x, y] if np.any(~np.isnan(prior[0][0])) else np.nan][0], np.nan, np.nan]
        use_prior_sig = [[prior_sig[0][0][x, y] if np.any(~np.isnan(prior_sig[0][0])) else np.nan][0], np.nan, np.nan]
        if len(dt) > 0:
            model = optimize_IcIsIr2(dt, prior=use_prior, prior_sig=use_prior_sig, forceIp2zero=IptoZero)
            Ic, Is, Ip = model.x
            Ic_image[x][y] = Ic
            Is_image[x][y] = Is
            Ip_image[x][y] = Ip
        bari += 1
        bar.update(bari)
    bar.finish()
    if save:
        try:
            np.save(savefile + 'Ic/' + fn[-13:-3] + name_ext, Ic_image)
            np.save(savefile + 'Is/' + fn[-13:-3] + name_ext, Is_image)
            if not IptoZero:
                np.save(savefile + 'Ip/' + fn[-13:-3] + name_ext, Ip_image)
        except FileNotFoundError:
            os.mkdir(savefile + 'Ic/')
            os.mkdir(savefile + 'Is/')
            os.mkdir(savefile + 'Ip/')
            np.save(savefile + 'Ic/' + fn[-13:-3] + name_ext, Ic_image)
            np.save(savefile + 'Is/' + fn[-13:-3] + name_ext, Is_image)
            if not IptoZero:
                np.save(savefile + 'Ip/' + fn[-13:-3] + name_ext, Ip_image)
    return Ic_image, Is_image, Ip_image


def update_fits(data_file_path, fits_file_path):
    """
    takes a series of 3D .npy arrays in data_file_path and appends them to the end of the appropriate fits file
    in fits_file_path
    :param data_file_path: location of the .npy arrays
    :param fits_file_path: location of the fits files
    :return:
    """
    for i, fn in enumerate(os.listdir(data_file_path)):
        for j, fit in enumerate(os.listdir(fits_file_path)):
            if fn.endswith('.npy'):
                if fit.endswith('.fits'):
                    if fit[0:-5] == fn[0:-4]:
                        try:
                            hdu = fits.open(fits_file_path + fit)
                            data = np.load(data_file_path + fn)
                            image = fits.ImageHDU(data)
                            hdu.append(image)
                            hdu.writeto(fits_file_path + fit, overwrite=True)
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


def plot_drizzle(file_path, plot_type='Intensity', axes=None, target='', intt=30):
    """
    plots and saves the result of the Drizzle algorithm
    https://github.com/spacetelescope/drizzle
    Assumes the HDUs in the fits file are in the format as outputted by Photontable.getFits()

    :param file_path: location of the fits files to be drizzled
    :param plot_type: 'Intensity' plots the total intensity drizzle, 'Ip' plots the drizzled Ip data from the SSD
    decomposition
    :param axes: axes on which to put the plot
    :return: matplotlib Axes object
    """
    infiles = []
    for i, fn in enumerate(os.listdir(file_path)):
        if fn.endswith('.fits') and fn.startswith('1'):
            infiles.append(file_path + fn)
    ref_wcs = get_canvas_wcs(target)
    driz = Drizzle(outwcs=ref_wcs, pixfrac=0.5)
    for i, infile in enumerate(infiles):
        try:
            imlist = fits.open(infile)
            if plot_type == 'Ip':
                try:
                    image = imlist[-1].data
                except IndexError:
                    getLogger(__name__).info('Fits files in {} have no Ic, Is, and Ip data attached'.format(file_path))
            else:
                image = imlist[1].data
            image_wcs = WCS(imlist[1].header)
            bad_mask = pixelflags.problem_flag_bitmask(flag_list=pixelflags.FLAG_LIST)
            flags = imlist[3].data
            weight_arr = np.zeros((140, 146))
            for (x, y), flag in np.ndenumerate(flags):
                if flags[x][y] & bad_mask != 0:
                    weight_arr[x][y] = 0
                else:
                    weight_arr[x][y] = 1
            cps = image/intt
            driz.add_image(cps, inwcs=image_wcs, in_units='cps', expin=intt, wt_scl=weight_arr)
            # driz.write('/mnt/data0/steiger/MEC/20200731/Hip5319/dither_2/SSD/fits/' + 'drizzler_step_' + str(i).zfill(3) + '.fits')
        except ValueError:
            pass
    driz.write(file_path + plot_type + '_drizzle.fits')
    ffile = fits.open(file_path + plot_type + '_drizzle.fits')
    data_im = ffile[1].data
    im = axes.imshow(data_im, vmin=0, vmax=1000, cmap='magma')
    plt.colorbar(im, ax=axes)
    axes.set_title(plot_type + ' drizzle', fontsize=10)
    return axes


def plot_drizzle_binned(file_path, plot_type='Ic', axes=None, target='', intt=15):
    """
    Drizzles either an Ic or Is image that has been appended to the end of a fits file. NOTE! This assumes that the Ic
    image was appended first
    :param file_path: location of the fits files to be drizzled
    :param plot_type: 'Ic' or 'Is' - which image to drizzle
    :param axes: Matplotlib axes object
    :param intt: integration time of observation
    :return: Matplotlib axes object
    """
    infiles = []
    for i, fn in enumerate(os.listdir(file_path)):
        if fn.endswith('.fits') and fn.startswith('1'):
            infiles.append(file_path + fn)
    ref_wcs = get_canvas_wcs(target)
    driz = Drizzle(outwcs=ref_wcs, pixfrac=0.5)
    for i, infile in enumerate(infiles):
        try:
            imlist = fits.open(infile)
            if plot_type == 'Is':
                try:
                    image = imlist[-1].data
                except IndexError:
                    getLogger(__name__).info('Fits files in {} have no Ic and Is data attached'.format(file_path))
            else:
                # drizzle IC data
                image = imlist[-2].data
            image_wcs = WCS(imlist[1].header)
            bad_mask = pixelflags.problem_flag_bitmask(flag_list=pixelflags.FLAG_LIST)
            flags = imlist[3].data
            weight_arr = np.zeros((140, 146))
            for (x, y), flag in np.ndenumerate(flags):
                if flags[x][y] & bad_mask != 0:
                    weight_arr[x][y] = 0
                else:
                    weight_arr[x][y] = 1
            cps = image/intt
            driz.add_image(cps, inwcs=image_wcs, in_units='cps', expin=intt, wt_scl=weight_arr)
            # driz.write('/mnt/data0/steiger/MEC/20200731/Hip5319/dither_2/SSD/fits/' + 'drizzler_step_' + str(i).zfill(3) + '.fits')
        except ValueError:
            pass
    driz.write(file_path + plot_type + '_drizzle.fits')
    ffile = fits.open(file_path + plot_type + '_drizzle.fits')
    data_im = ffile[1].data
    im = axes.imshow(data_im, vmin=0, vmax=1000, cmap='magma')
    plt.colorbar(im, ax=axes)
    axes.set_title(plot_type + ' drizzle', fontsize=10)
    return axes


def get_canvas_wcs(target):
    npix=500
    coords = SkyCoord.from_name(target)
    wcs = astropy.wcs.WCS(naxis = 2)
    wcs.wcs.crpix = np.array([npix/2., npix/2.])
    wcs.wcs.crval = [coords.ra.value, coords.dec.value]
    wcs.wcs.ctype = ["RA--TAN", "DEC-TAN"]
    wcs.pixel_shape = (npix, npix)
    wcs.wcs.pc = np.array([[1,0],[0,1]])
    wcs.wcs.cdelt = [2.8888888888888894e-06, 2.8888888888888894e-06] #corresponds to a 10.4 mas platescale
    wcs.wcs.cunit = ["deg", "deg"]
    getLogger(__name__).debug(wcs)
    return wcs


def plot_icisip(fn, Ic, Is, Ip):
    '''
    Easy plotting function to compare Ic, Ip and Is
    :param fn: File path to save the finished plot to
    :param Ic: 2D array of Ic
    :param Is: 2D array of Ic
    :param Ip: 2D array of Ic
    :return:
    '''
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


def calculate_icis_binned_from_fits(fits_file, hdu_loc, x, y, savefile, exptime, save=True):
    '''
    Not tested
    :param fits_file: location of the fits file to perform SSD on
    :param hdu_loc: location of the image hdu within the its file - starts at 0 indexing
    :param x: number of x pixels
    :param y: number if y pixels
    :param save: if true will save the arrays in savefile
    :param savefile: location to save SSD products
    :param exptime: length of exposures
    :return: Ic, Is images
    '''
    hdu = fits.open(fits_file)
    data = hdu[hdu_loc].data
    # data should be a 1d array containing the binned intensity as a function of time, i.e. a lightcurve
    Ic_image = np.zeros((x, y))
    Is_image = np.zeros((x, y))
    maxval = x * y
    bar = ProgressBar(maxval=maxval).start()
    bari = 0
    for (x, y) in np.ndenumerate(data):
        mu = np.mean(data)
        var = np.var(data)
        try:
            IIc, IIs = np.asarray(muVar_to_IcIs(mu, var, exptime)) * exptime
        except:
            # print('\nmuVar_to_IcIs failed\n')
            IIc = mu / 2  # just create a reasonable seed
            IIs = mu - IIc
        Ic, Is, res = maxBinMRlogL(data[x, y], Ic_guess=IIc, Is_guess=IIs)
        Ic_image[x][y] = Ic
        Is_image[x][y] = Is
        bari += 1
        bar.update(bari)
    bar.finish()
    if save:
        np.save(savefile + 'Ic_binned', Ic_image)
        np.save(savefile + 'Is_binned', Is_image)
    return Ic_image, Is_image
    # assuming data is now an array with units of counts/bin in the shape (x_pix, y_pix)


def calculate_icis_binned(fn, save=True, savefile='', name_ext=''):
    """
    function to run binned SSD
    :param fn: file on which to run SSD
    :param save: if True will save the Ic and Is images using np.save()
    :param savefile: path where you would like the saved file to go - not used if save is False
    :return: Ic image, Is image
    """
    obs = Photontable(fn)
    exclude_flags = pixelflags.PROBLEM_FLAGS
    Ic_image = np.zeros((140, 146))
    Is_image = np.zeros((140, 146))
    if os.path.exists(savefile + 'Ic/' + fn[-13:-3] + name_ext + '.npy'):
        getLogger(__name__).info('Ic and Is already calculated for {}'.format(fn))
        return
    bar = ProgressBar(maxval=20439).start()
    bari = 0
    for (x, y), resID in np.ndenumerate(obs.beamImage):
        if obs.flagMask(exclude_flags, (x, y)) and any(exclude_flags):
            continue
        ts = obs.getPixelPhotonList(xCoord=x, yCoord=y)['Time']
        if len(ts) > 0:
            effExpTime = 0.02
            lightCurveIntensityCounts, lightCurveIntensity, lightCurveTimes = getLightCurve(ts/10**6)
            mu = np.mean(lightCurveIntensityCounts)
            var = np.var(lightCurveIntensityCounts)
            try:
                IIc, IIs = np.asarray(muVar_to_IcIs(mu, var, effExpTime)) * effExpTime
            except:
                # print('\nmuVar_to_IcIs failed\n')
                IIc = mu / 2  # just create a reasonable seed
                IIs = mu - IIc
            Ic, Is, res = maxBinMRlogL(lightCurveIntensityCounts, Ic_guess=IIc, Is_guess=IIs)
            Ic_image[x][y] = Ic
            Is_image[x][y] = Is
        else:
            Ic_image[x][y] = 0
            Is_image[x][y] = 0
        bari += 1
        bar.update(bari)
    bar.finish()
    if save:
        try:
            np.save(savefile + 'Ic/' + fn[-13:-3] + name_ext, Ic_image)
            np.save(savefile + 'Is/' + fn[-13:-3] + name_ext, Is_image)
        except FileNotFoundError:
            os.mkdir(savefile + 'Ic/')
            os.mkdir(savefile + 'Is/')
            np.save(savefile + 'Ic/' + fn[-13:-3] + name_ext, Ic_image)
            np.save(savefile + 'Is/' + fn[-13:-3] + name_ext, Is_image)
    return Ic_image, Is_image


def plot_intensity_histogram(data, object_name='object', N=400, span=[0, 300], axes=None, fit_poisson=True):
    """
    for a given 3D data array will plot the count rate histograms and perform a modified rician fit
    :param data: 3D data array
    :param object_name: str name of the object
    :param axes: axes on which to make the plot
    :param fit_poisson: if True will fit a poisson distribution to the histogram in addiiton to a MR
    :return: matplotlib Axes object
    """

    histS_o, binsS_o = lc.histogramLC(data, centers=True, N=N, span=span)
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


