"""
Author: Sarah Steiger           Date: 09/29/2020

Code to inject planet photons into MEC data
"""
import numpy as np
import matplotlib.pyplot as plt
from mkidcore.instruments import CONEX2PIXEL
import tables
from mkidpipeline.hdf.photontable import Photontable
import astropy
from astropy.io import fits
from astropy.coordinates import EarthLocation, SkyCoord
from astroplan import Observer
import os
from astropy.convolution import Gaussian2DKernel, convolve
import mkidpipeline.speckle.photonstats_utils as utils
from astropy.convolution import AiryDisk2DKernel
from multiprocessing import Pool
from functools import partial


class InjectPlanet:
    def __init__(self, h5_folder, target, sep, pa, cps, conex_ref=None, pixel_ref=None):
        self.h5_folder = h5_folder
        self.h5_files = []
        for i, fn in enumerate(os.listdir(self.h5_folder)):
            if fn.endswith('.h5'):
                self.h5_files.append(self.h5_folder + fn)
        self.target = target
        self.sep = sep
        self.cps = cps
        self.pa = pa
        self.photontables = [Photontable(h5) for h5 in self.h5_files]
        self.intt = self.photontables[0].getFromHeader('expTime')
        self.pix_sep = (sep * 1000) / self.photontables[0].metadata()['platescale']
        self.conex_ref = conex_ref if conex_ref else self.photontables[0].metadata()['dither_home']
        self.pixel_ref = pixel_ref if pixel_ref else self.photontables[0].metadata()['dither_ref']
        for i, pt in enumerate(self.photontables):
            self.photontables[i].file.close()
        self.start_times = np.array(get_start_times(self.h5_folder))
        self.pa_list = get_pa_list(self.start_times, intt=self.intt, target=self.target)
        self.delta_pas = np.array([i - self.pa_list[0] for i in self.pa_list])

    def run(self):
        """
        runs the planet injection code: puts fake ocmpanion photons into a sequence of h5 observation files
        :return:
        """
        for i, file in enumerate(self.h5_files):
            injected_photons = self.get_companion_photons(file)
            h5file = tables.open_file(file, mode="a", title="MKID Photon File")
            original_photons = h5file.root.Photons.PhotonTable.read()
            table = h5file.root.Photons.PhotonTable
            print('Removing exiting photons from table')
            table.remove_rows(0, table.nrows)
            print('combining existing and injected photons')
            total_photons = np.concatenate([original_photons, injected_photons])
            print('sorting photons on ResID then time')
            total_photons.sort(order=('ResID', 'Time'))
            table.append(total_photons)
            h5file.close()

    def get_companion_photons(self, h5_file):
        """

        :param h5_file: h5 file in which photons are to be injected
        :return: numpy array of photons
        """
        pt = Photontable(h5_file)
        intt = self.intt
        # get companion template where center of the PSF will have counts/s =max_counts
        companion = generate_diffrac_lim_planet(max_counts=self.cps)
        companion = companion.astype(int)

        # give the x and y pixel for where the companion is to be injected
        print('Processing h5: ' + str(h5_file))
        x_cen, y_cen = self.find_center_companion(pt)

        # get x and y coordinates for where companion is going to go
        delt_x = int(np.shape(companion)[0] / 2.0)
        delt_y = int(np.shape(companion)[1] / 2.0)
        xmin, xmax = x_cen - delt_x, x_cen + delt_x + 1
        ymin, ymax = y_cen - delt_y, y_cen + delt_y + 1

        companion_image = np.zeros((pt.nXPix, pt.nYPix))
        diff_x = 140 - xmax
        diff_y = 146 - ymax

        arr = np.array([xmin, diff_x, ymin, diff_y])
        if np.all([a > 0 for a in arr]):
            # companion entirely on array
            companion_image[xmin:xmax, ymin:ymax] = companion
        else:
            print('companion partially off array')
            arr = arr.astype(float)
            arr[arr < 0] = np.nan
            # get image bounds
            xmin_i = xmin if not np.isnan(arr[0]) else 0
            xmax_i = xmax if not np.isnan(arr[1]) else 140
            ymin_i = ymin if not np.isnan(arr[2]) else 0
            ymax_i = ymax if not np.isnan(arr[3]) else 146
            # get companion bounds
            xmin = 0 if not np.isnan(arr[0]) else -xmin
            xmax = np.shape(companion)[0] if not np.isnan(arr[1]) else diff_x
            ymin = 0 if not np.isnan(arr[2]) else -ymin
            ymax = np.shape(companion)[1] if not np.isnan(arr[3]) else diff_y
            companion_image[xmin_i:xmax_i, ymin_i:ymax_i] = companion[xmin:xmax, ymin:ymax]
        companion_image = companion_image.astype(int)
        num_photons = np.sum(companion_image * intt)
        # initialize photon list
        photons = np.zeros(num_photons, dtype=np.dtype([('ResID', np.uint32), ('Time', np.uint32),
                                                        ('Wavelength', np.float32), ('SpecWeight', np.float32),
                                                        ('NoiseWeight', np.float32)]))
        tally = 0
        for (x, y), resID in np.ndenumerate(pt.beamImage):
            if companion_image[x, y] != 0:
                counts = companion_image[x, y] * intt
                t_prev = 0
                for i in range(counts):
                    # create a photon entry for each count with poisson distributed arrival times
                    pixel_cps = counts/intt
                    t = t_prev + np.random.poisson(10**6/pixel_cps)
                    wvl = 1100
                    photons[tally + i] = (resID, t, wvl, 1.0, 1.0)
                    t_prev = t
                tally += counts

        # close h5 file
        pt.file.close()
        return photons

    def find_center_companion(self, pt):
        """

        :param pt: Photontable object
        :return: (x, y) location of the center of the companion
        """
        # calculate change in pixel location due to the dither
        fn = pt.file.filename
        idx = self.h5_files.index(fn)
        delta_pix_dith = dither_pixel_vector(pt, center=self.conex_ref)
        rotation_angle = np.deg2rad(self.delta_pas[idx])
        assigned_pa = np.deg2rad(self.pa)
        apply_angle = rotation_angle + assigned_pa
        # find reference pixel of the companion using the separation and position angle
        comp_ref_pix = (self.pixel_ref[0] + self.pix_sep*np.sin(apply_angle),
                        self.pixel_ref[1] + self.pix_sep*np.cos(apply_angle))
        x_cen, y_cen = comp_ref_pix + delta_pix_dith
        return int(x_cen), int(y_cen)


def dither_pixel_vector(pt, center):
    """
    adapted from drizzler.py, finds the pixel displacement from the start for a given dither postion
    :param pt: Photontable
    :param center: (x, y) location of reference center
    :return: (x, y) displacement relative to center
    """
    pos = pt.metadata()['dither_pos']
    position = np.asarray(pos)
    pix = np.asarray(CONEX2PIXEL(position[0], position[1])) - np.array(CONEX2PIXEL(*center))
    return pix


def get_start_times(folder):
    """
    generates a list of times from the names of the h5 files in a given folder
    :param folder: folder containing the h5 files
    :return: numpy array of start times
    """
    times = []
    for i, fn in enumerate(os.listdir(folder)):
        if fn.endswith('.h5'):
            times.append(int(fn[:-3]))
    return np.sort(np.array(times))


def get_pa_list(times, intt=1, target='', observatory='subaru'):
    """
    finds the lost of position angles for a given array of start times
    :param times: numpy array of start times
    :param intt: int timestep for calculating the PA
    :param target: str name of the target
    :param observatory: str name of the observatory where the observations occurred
    :return:
    """
    times = np.sort(times)
    pa_list = []
    coord = SkyCoord.from_name(target)
    ref = (times[1] - times[0]) / intt
    ref = int(ref)
    for i, time in enumerate(times):
        if i < (len(times) - 1):
            use_times = np.arange(times[i], times[i + 1], intt)
        else:
            use_times = np.arange(times[i], times[i] + ref, intt)
        if len(use_times) > ref:
            use_times = use_times[:ref]
        apo = Observer.at_site(observatory)
        parallactic_angles = apo.parallactic_angle(astropy.time.Time(val=use_times, format='unix'),
                                                   coord).value
        pa_list.append(parallactic_angles)
    pa_list = np.rad2deg(np.array(pa_list))
    return np.array(pa_list.flatten())


def corrsequence(Ttot, tau):
    """
    Generate a sequence of correlated Gaussian noise, correlation time
    tau.  Algorithm is recursive and from Markus Deserno.  The
    recursion is implemented as an explicit for loop but has a lower
    computational cost than converting uniform random variables to
    modified-Rician random variables.

    Arguments:
    Ttot: int, the total integration time in microseconds.
    tau: float, the correlation time in microseconds.

    Returns:
    t: a list of integers np.arange(0, Ttot)
    r: a correlated Gaussian random variable, zero mean and unit variance, array of length Ttot

    """

    t = np.arange(Ttot)
    g = np.random.normal(0, 1, Ttot)
    r = np.zeros(g.shape)
    f = np.exp(-1. / tau)
    sqrt1mf2 = np.sqrt(1 - f ** 2)
    r = utils.recursion(r, g, f, sqrt1mf2, g.shape[0])

    return t, r


def generate_diffrac_lim_planet(diffrac_lim=2.68, max_counts=300, type='Gaussian'):
    """
    generates a 2D cps image of a companon
    :param diffrac_lim: float diffraction limit
    :param max_counts: maximum counts for the center of the comapnion
    :param type: "Airy' or 'Gaussian'
    :return: 2D PSF image
    """
    if type == 'Airy':
        psf = AiryDisk2DKernel(radius=diffrac_lim).array
        norm = np.max(psf)/max_counts
        psf /= norm
    elif type == 'Gaussian':
        psf = Gaussian2DKernel(diffrac_lim).array
        norm = np.max(psf)/max_counts
        psf /= norm
    return psf


a = InjectPlanet(h5_folder='/mnt/data0/steiger/MEC/injected_companion_data/', target='Hip109427', sep=0.3, pa=90,
                 cps=300, conex_ref=[0.5, -0.6], pixel_ref=[62, 36])
a.run()
print('done!')
