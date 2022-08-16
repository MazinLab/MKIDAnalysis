#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from scipy import special, interpolate
import mkidanalysis.speckle.photonstats_utils as utils
from astropy.convolution import Gaussian2DKernel, convolve
from mkidpipeline.photontable import Photontable
import tables
from scipy.stats import gamma
import scipy


def p_I(I, k=None, theta=None, mu=None, I_peak=None):
    p = (((-np.log(I / I_peak) - mu) / theta) ** (k - 1) * np.exp((np.log(I / I_peak) + mu) / theta)) / (
                special.gamma(k) * theta * I)
    return p

def p_A(x, gam=None, bet=None, alph=None):
    pdf = gamma.pdf(x, alph, loc=gam, scale=1 / bet)
    return pdf


def gamma_icdf(median_strehl, Ip, bet=None, alph=None, interpmethod='cubic'):
    """
    :param sr: MEDIAN value of the strehl ratio
    :param Ip:
    :param gam:
    :param bet:
    :param alph:
    :param interpmethod:
    :return:
    """
    # compute mean and variance of gamma distribution
    sr1 = 0.1  # max(0, mu - 15 * sig)
    sr2 = 1.0  # min(mu + 15 * sig, 1.)
    sr = np.linspace(sr1, sr2, 1000)
    gam = -(median_strehl + (1 - median_strehl))
    p_I = (1. / (2 * np.sqrt(sr))) * (p_A(np.sqrt(sr), gam=gam, alph=alph, bet=bet) +
                                      p_A(-np.sqrt(sr), gam=gam, alph=alph, bet=bet))
    norm = scipy.integrate.simps(p_I)
    p_I /= norm
    # go from strehls to intensities
    I = (sr * Ip) / median_strehl
    dI = I[1] - I[0]
    I += dI / 2
    cdf = np.cumsum(p_I) * dI
    cdf /= cdf[-1]
    # The integral is defined with respect to the bin edges.
    I = np.asarray([0] + list(I + dI / 2))
    cdf = np.asarray([0] + list(cdf))
    # The interpolation scheme doesn't want duplicate values.  Pick
    # the unique ones, and then return a function to compute the
    # inverse of the CDF.
    i = np.unique(cdf, return_index=True)[1]
    return interpolate.interp1d(cdf[i], I[i], kind=interpmethod)


def MRicdf(Ic, Is, interpmethod='cubic'):
    """
    Compute an interpolation function to give the inverse CDF of the
    modified Rician with a given Ic and Is.

    Arguments:
    Ic: float, parameter for M-R
    Is: float > 0, parameter for M-R

    Optional argument:
    interpmethod: keyword passed as 'kind' to interpolate.interp1d

    Returns:
    interpolation function f for the inverse CDF of the M-R

    """

    if Is <= 0 or Ic < 0:
        raise ValueError("Cannot compute modified Rician CDF with Is<=0 or Ic<0.")

    # Compute mean and variance of modified Rician, compute CDF by
    # going 15 sigma to either side (but starting no lower than zero).
    # Use 1000 points, or about 30 points/sigma.

    mu = Ic + Is
    sig = np.sqrt(Is ** 2 + 2 * Ic * Is)
    I1 = max(0, mu - 15 * sig)
    I2 = mu + 15 * sig
    I = np.linspace(I1, I2, 1000)

    # Grid spacing.  Set I to be offset by dI/2 to give the
    # trapezoidal rule by direct summation.

    dI = I[1] - I[0]
    I += dI / 2

    # Modified Rician PDF, and CDF by direct summation of intensities
    # centered on the bins to be integrated.  Enforce normalization at
    # the end since our integration scheme is off by a part in 1e-6 or
    # something.

    # p_I = 1./Is*np.exp(-(Ic + I)/Is)*special.iv(0, 2*np.sqrt(I*Ic)/Is)
    p_I = 1. / Is * np.exp((2 * np.sqrt(I * Ic) - (Ic + I)) / Is) * special.ive(0, 2 * np.sqrt(I * Ic) / Is)

    cdf = np.cumsum(p_I) * dI
    cdf /= cdf[-1]

    # The integral is defined with respect to the bin edges.

    I = np.asarray([0] + list(I + dI / 2))
    cdf = np.asarray([0] + list(cdf))

    # The interpolation scheme doesn't want duplicate values.  Pick
    # the unique ones, and then return a function to compute the
    # inverse of the CDF.

    i = np.unique(cdf, return_index=True)[1]
    return interpolate.interp1d(cdf[i], I[i], kind=interpmethod)


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


def genphotonlist(Ic, Is, Ir, Ttot, tau, deadtime=0, interpmethod='cubic', taufac=500, return_IDs=False,
                  mean_strehl=None, gamma_distributed=True, beta=30, alpha=5, background_count_rate=0,
                  remove_background=True):
    """
    Generate a mock photonlist from input Ic, Is, and Ir with deadtime accurately handled. Ic and Is define the modified
    Rician fron which the speckle photons are sampled and Ir can either be taken to be constant or sampled from a Gamma
    distribution. Background photons can also optionally be placed into the photon list and either removed before
    returning (simulating a wavelength cut) or left in.
    :param Ic: Constant component of the modified Rician (MR), in units of 1/second
    :param Is: time variable component of the modified Rician (MR), in units of 1/second
    :param Ir: Companion intensity, in units of 1/second
    :param Ttot: total exposure time, in seconds
    :param tau: correlation time, in seconds
    :param deadtime: Detector dead time, in microseconds
    :param interpmethod: argument 'kind' to interpolate.interp1d
    :param taufac: float, discretize intensity with bin width tau/taufac.  Doing so speeds up the code immensely
    :param return_IDs: if True, returns an array giving the distribution (MR or constant) that produced each photon?
    :param mean_strehl: Average Strehl Ratio (SR) - used for calculating the Gamma CDF if gamma_distributed is True
    :param gamma_distributed: if True will have the companion intensity follow a Gamma distributed PDF
    :param beta: Gamma distribution parameter
    :param alpha: Gamma distribution parameter
    :param background_count_rate: background count rate to add to the photonlist, in unit of counts/second
    :param remove_background: if True will insert then remove the specified background - simulates performing a
    wavelength cut on MKID data
    :return:
    1D array of photon arrival times (in microseconds) and (if return_IDs)
    1D array of photon arrival times if planet wasnt there (no deadtime from planet photons),
    1D array, 0 if photon came from Ic/Is MR, 1 if photon came from Ir
    """
    # Generate a correlated Gaussian sequence, correlation time tau. Then transform this to a random variable
    # uniformly distributed between 0 and 1, and finally back to a modified Rician random variable.
    # This method ensures that: (1) the output is M-R distributed, and (2) it is exponentially correlated.  Finally,
    # return a list of photons determined by the probability of each unit of time giving a detected photon.

    # Number of microseconds per bin in which we discretize intensity
    N = max(int(tau * 1e6 / taufac), 1)
    #Add in MR distributed Speckle intensities
    if Is > 1e-8 * Ic:
        t, normal = corrsequence(int(Ttot * 1e6 / N), tau * 1e6 / N)
        uniform = 0.5 * (special.erf(normal / np.sqrt(2)) + 1)
        t *= N
        f = MRicdf(Ic, Is, interpmethod=interpmethod)
        I = f(uniform) / 1e6
    elif Is >= 0:
        N = max(N, 1000)
        t = np.arange(int(Ttot * 1e6))
        I = Ic / 1e6 * np.ones(t.shape)
    else:
        raise ValueError("Cannot generate a photon list with Is<0.")
    # Add in companion intensities (Gamma distributed or constant)
    if gamma_distributed and Ir > 1e-6:
        t2, normal2 = corrsequence(int(Ttot * 1e6 / N), tau * 1e6 / N)
        uniform = 0.5 * (special.erf(normal2 / np.sqrt(2)) + 1)
        t2 *= N
        f = gamma_icdf(mean_strehl, Ir, bet=beta, alph=alpha, interpmethod=interpmethod)
        I_comp = f(uniform) / 1e6
    else:
        t2 = np.arange(0, int(Ttot * 1e6), N)
        I_comp = np.ones(t2.shape) * Ir / 1e6
    t3 = np.arange(0, int(Ttot * 1e6), N)
    # Number of photons from each distribution in each time bin
    n_mr = np.random.poisson(I * N)
    n_comp = np.random.poisson(I_comp * N)
    n_back = np.random.poisson(np.ones(t3.shape) * background_count_rate / 1e6 * N)

    # Go ahead and make the list with repeated times
    tlist = t[(n_mr > 0)]
    tlist_r = t2[(n_comp > 0)]
    tlist_back = t3[(n_back > 0)]
    for i in range(1, max(np.amax(n_mr), np.amax(n_comp), np.amax(n_back)) + 1):
        tlist = np.concatenate((tlist, t[(n_mr > i)]))
        tlist_r = np.concatenate((tlist_r, t2[(n_comp > i)]))
        tlist_back = np.concatenate((tlist_back, t3[(n_back > i)]))

    tlist_tot = np.concatenate((tlist, tlist_r, tlist_back)) * 1.
    # Add a random number to give the exact arrival time within the bin
    tlist_tot += N * np.random.rand(len(tlist_tot))
    # Cython is much, much faster given that this has to be an explicit for loop; without Cython (even with numba)
    # this step would dominate the run time.  Returns indices of the times we keep.
    indx = np.argsort(tlist_tot)
    keep = utils.removedeadtime(tlist_tot[indx], deadtime)
    # plist tells us which distribution (MR or constant) actually produced a given photon; return this if desired.
    if return_IDs:
        indx2 = indx[(indx < len(tlist))]
        keep2 = utils.removedeadtime(tlist_tot[indx2], deadtime)
        plist1 = np.zeros(tlist.shape).astype(int)
        plist2 = np.ones(tlist_r.shape).astype(int)
        plist3 = np.full_like(tlist_back, 2).astype(int)
        plist_tot = np.concatenate((plist1, plist2, plist3))
        if remove_background:
            bkgd_exclude_keep = np.copy(keep)
            bkgd_exclude_keep[np.where(plist_tot[indx] == 2)] = 0
            return tlist_tot[indx][np.where(bkgd_exclude_keep)], tlist_tot[indx2][np.where(keep2)], \
                   plist_tot[indx][np.where(bkgd_exclude_keep)]
        else:
            return tlist_tot[indx][np.where(keep)], tlist_tot[indx2][np.where(keep2)], plist_tot[indx][np.where(keep)]
    return tlist_tot[indx][np.where(keep)]


def diffrac_lim_kernel(input_image, diffrac_lim=2.68, *argsconvolve, **kwargsconvolve):
    sigma = diffrac_lim / (2. * np.sqrt(2. * np.log(2.)))
    gaussian_2D_kernel = Gaussian2DKernel(sigma)
    convolved_kernel = convolve(input_image, gaussian_2D_kernel, *argsconvolve, **kwargsconvolve)
    return convolved_kernel


def genphotonlist2D(Ic, Is, Ir, Ttot, tau, out_directory, deadtime=0, interpmethod='cubic',
                    taufac=500, diffrac_lim=2.86):
    """
    Same functionality as genphotonlist except this function is intended to iterate over an entire 2D (Ic,Is) map
    It makes sure the intensities generated from corrsequence are correlated spatially (they will be correlated temporally)
    """

    # Generate a correlated Gaussian sequence, correlation time tau.
    # Then transform this to a random variable uniformly distributed
    # between 0 and 1, and finally back to a modified Rician random
    # variable.  This method ensures that: (1) the output is M-R
    # distributed, and (2) it is exponentially correlated.  Finally,
    # return a list of photons determined by the probability of each
    # unit of time giving a detected photon.

    # Number of microseconds per bin in which we discretize intensity
    N = max(int(tau * 1e6 / taufac), 1)

    t_base, N_base = corrsequence(int(Ttot * 1e6 / N), tau * 1e6 / N)
    x_size, y_size = Ic.shape
    t_size = len(t_base)
    intensity_cube = np.zeros([x_size, y_size, t_size])

    for x_pix in range(x_size):
        for y_pix in range(y_size):

            if Is[x_pix, y_pix] > 0.0 and Ic[x_pix, y_pix] > 0.0 and Is[x_pix, y_pix] > 1e-8 * Ic[x_pix, y_pix]:
                t, normal = corrsequence(int(Ttot * 1e6 / N), tau * 1e6 / N)
                intensity_cube[x_pix, y_pix, :] = normal

            elif Is[x_pix, y_pix] >= 0.0 and Ic[x_pix, y_pix] >= 0.0:
                # N = max(N, 1000)
                t = np.arange(int(Ttot * 1e6))
                intensity_cube[x_pix, y_pix, :] = Ic[x_pix, y_pix] / 1e6 * np.ones(t_size)

            else:
                intensity_cube[x_pix, y_pix, :] = np.full_like(intensity_cube[x_pix, y_pix, :], np.nan)

    intensity_cube_uncorr = np.copy(intensity_cube)

    for timestamp in range(t_size):
        intensity_cube[:, :, timestamp] = diffrac_lim_kernel(intensity_cube[:, :, timestamp], diffrac_lim=diffrac_lim,
                                                             nan_treatment='interpolate', preserve_nan=True)
    intensity_cube_corr = np.copy(intensity_cube)
    photon_table = tables.open_file(out_directory, mode='w')
    Header = photon_table.create_group(photon_table.root, 'header', 'header')
    header = photon_table.create_table(Header, 'header', Photontable.PhotontableHeader, title='Header')

    beamFlag = np.zeros([x_size, y_size])
    beamMap = np.zeros([x_size, y_size])

    BeamMap = photon_table.create_group(photon_table.root, 'BeamMap', 'BeamMap')
    tables.Array(BeamMap, 'Flag', obj=beamFlag)
    tables.Array(BeamMap, 'Map', obj=beamMap)

    Images = photon_table.create_group(photon_table.root, 'Images', 'Images')

    Photons = photon_table.create_group(photon_table.root, 'Photons', 'Photons')
    PhotonTable = photon_table.create_table(Photons, 'PhotonTable', Photontable.PhotonDescription, title='Photon Data')

    head = header.row

    head['beammap_file'] = ''
    head['data_path'] = ''
    head['energy_resolution'] = 0.1
    head['EXPTIME'] = 0.0
    head['flatcal'] = ''
    head['lincal'] = False
    head['pixcal'] = False
    head['speccal'] = ''
    head['UNIXSTART'] = 0
    head['target'] = 'CHARIS'
    head['timeMaskExists'] = False
    head['min_wavelength'] = 700.0
    head['max_wavelength'] = 700.0
    head['wavecal'] = ''
    head.append()

    photon = PhotonTable.row
    iteration = 0
    t *= N
    for x_pix in range(x_size):
        for y_pix in range(y_size):
            print(x_pix, y_pix)
            if Is[x_pix, y_pix] > 0.0 and Ic[x_pix, y_pix] > 0.0 and Is[x_pix, y_pix] > 1e-8 * Ic[x_pix, y_pix]:
                normal = intensity_cube[x_pix, y_pix, :]
                uniform = 0.5 * (special.erf(normal / np.sqrt(2)) + 1)
                f = MRicdf(Ic[x_pix, y_pix], Is[x_pix, y_pix], interpmethod=interpmethod)
                I = f(uniform) / 1e6
                intensity_cube[x_pix, y_pix, :] = I

            # Number of photons from each distribution in each time bin
            I = intensity_cube[x_pix, y_pix, :]
            n1 = np.random.poisson(I * N)
            n2 = np.random.poisson(np.ones(t.shape) * Ir / 1e6 * N)

            # Go ahead and make the list with repeated times

            tlist = t[np.where(n1 > 0)]
            tlist_r = t[np.where(n2 > 0)]

            for i in range(1, max(np.amax(n1), np.amax(n2)) + 1):
                tlist = np.concatenate((tlist, t[np.where(n1 > i)]))
                tlist_r = np.concatenate((tlist_r, t[np.where(n2 > i)]))

            tlist_tot = np.concatenate((tlist, tlist_r)) * 1.

            # Add a random number to give the exact arrival time within the bin

            tlist_tot += N * np.random.rand(len(tlist_tot))

            # Cython is much, much faster given that this has to be an
            # explicit for loop; without Cython (even with numba) this step
            # would dominate the run time.  Returns indices of the times we
            # keep.

            indx = np.argsort(tlist_tot)
            keep = utils.removedeadtime(tlist_tot[indx], deadtime)

            final_tlist = tlist_tot[indx][np.where(keep)]
            for Time in final_tlist:
                photon['resID'] = float(iteration)
                photon['wavelength'] = 700.0
                photon['weight'] = 1.0
                photon['time'] = Time
                photon.append()

            iteration += 1
    photon_table.flush()
    photon_table.close()
    intensity_cube_MR = np.copy(intensity_cube)

    return intensity_cube_uncorr, intensity_cube_corr, intensity_cube_MR


if __name__ == "__main__":
    # Demonstration: Ic=1000/s, Is=300/s, Ir=500/s, 5s integration,
    # decorrelation time 0.1s.  Returns list of ~9000 times.

    Ic, Is, Ir, Ttot, tau = [1000, 300, 500, 5, 0.1]
    t, p = genphotonlist(Ic, Is, Ir, Ttot, tau, deadtime=10, return_IDs=True)
    # Ic_array=np.zeros([20,20])+1000
    # Ic_array = np.load('/mnt/data0/isabel/sandbox/CHARIS/Ic.npy')[70:120, 70:120]
    # Is_array = np.load('/mnt/data0/isabel/sandbox/CHARIS/Is.npy')[70:120, 70:120]
    Ic_array = np.load('/mnt/data0/isabel/sandbox/CHARIS/Ic.npy')
    Is_array = np.load('/mnt/data0/isabel/sandbox/CHARIS/Is.npy')
    newIc=np.rotate(Ic_array, 30, 0,0)
    newIs=np.rotate(Is_array, 30, 0,0)
    Ic_array_crop = newIc[28:153, 28:153]
    Is_array_crop = newIs[28:153, 28:153]
    # Is_array = np.zeros([20, 20])+300
    intensity_cube_uncorr, intensity_cube_corr, intensity_cube_MR = genphotonlist2D(Ic_array_crop, Is_array_crop, 0, 10,
                                                                                    0.1,
                                                                                    '/mnt/data0/isabel/sandbox/CHARIS/testnewcode.h5',
                                                                                    deadtime=10, interpmethod='cubic',
                                                                                    taufac=500, diffrac_lim=2.86)
    uncorrsample = intensity_cube_uncorr[:, :, 1000]
    corrsample = intensity_cube_corr[:, :, 1000]
    MRcorrsample = intensity_cube_MR[:, :, 1000]
