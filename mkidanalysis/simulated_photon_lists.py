import numpy as np
from logging import getLogger
from scipy import special
import mkidanalysis.speckle.photonstats_utils as utils
import matplotlib.pyplot as plt
from mkidanalysis.speckle.generate_photons import corrsequence
from mkidanalysis.pdfs import mr_icdf, gamma_icdf
from progressbar import ProgressBar

class MockPhotonList:
    def __init__(self, Ic, Is, Ip, intt=30, tau=0.1, taufac=500, deadtime=10.e-6, mean_strehl=None, background_rate=0,
                 gamma_distributed=True, remove_background=False, beta=30, alpha=5, interpmethod='cubic'):
        """
        Generate a mock photonlist from input Ic, Is, and Ip with deadtime accurately handled. Ic and Is define the modified
        Rician fron which the speckle photons are sampled and Ip can either be taken to be constant or sampled from a Gamma
        distribution. Background photons can also optionally be placed into the photon list and either removed before
        returning (simulating a wavelength cut) or left in.
        :param Ic: Constant component of the modified Rician (MR), in units of 1/second
        :param Is: time variable component of the modified Rician (MR), in units of 1/second
        :param Ip: Companion intensity, in units of 1/second
        :param intt: total exposure time, in seconds
        :param tau: correlation time, in seconds
        :param deadtime: Detector dead time, in microseconds
        :param interpmethod: argument 'kind' to interpolate.interp1d
        :param taufac: float, discretize intensity with bin width tau/taufac.  Doing so speeds up the code immensely
        :param mean_strehl: Average Strehl Ratio (SR) - used for calculating the Gamma CDF if gamma_distributed is True
        :param gamma_distributed: if True will have the companion intensity follow a Gamma distributed PDF
        :param beta: Gamma distribution parameter
        :param alpha: Gamma distribution parameter
        :param background_rate: background count rate to add to the photonlist, in unit of counts/second
        :param remove_background: if True will insert then remove the specified background - simulates performing a
        wavelength cut on MKID data
        """
        self.Ic = Ic
        self.Is = Is
        self.Ip = Ip
        self.mean_strehl = mean_strehl
        self.intt = intt
        self.tau = tau
        self.deadtime = deadtime
        self.background_rate = background_rate
        self.remove_background = remove_background
        self.beta = beta
        self.alpha = alpha
        self.taufac = taufac
        self.interpmethod = interpmethod
        self.photon_list = None
        assert np.shape(self.Ic) == np.shape(self.Is) == np.shape(self.Ip)
        self.photon_flags = np.zeros_like(self.Ic, dtype=np.ndarray)
        self.dts = np.zeros_like(self.Ic, dtype=np.ndarray)
        self.photon_list = np.zeros_like(self.Ic, dtype=np.ndarray)
        if gamma_distributed and (alpha == 0 or beta == 0):
            raise ValueError('Requesting Gamma distributed arrival time distribution from on axis sources but at least'
                             'one Gamma parameter is 0')
        self.gamma_distributed = gamma_distributed
        bar = ProgressBar(maxval=len(self.Ic.flatten())).start()
        bari = 0
        for (x, y), val in np.ndenumerate(self.Ic):
            self.generate(self.Ic[x, y], self.Is[x, y], self.Ip[x, y], x, y)
            bari += 1
            bar.update(bari)
        bar.finish()

    def generate(self, Ic, Is, Ip, x, y):
        getLogger(__name__).info('Generating photon list')
        # Generate a correlated Gaussian sequence, correlation time tau. Then transform this to a random variable
        # uniformly distributed between 0 and 1, and finally back to a modified Rician random variable.
        # This method ensures that: (1) the output is M-R distributed, and (2) it is exponentially correlated.  Finally,
        # return a list of photons determined by the probability of each unit of time giving a detected photon.

        # Number of microseconds per bin in which we discretize intensity
        N = max(int(self.tau * 1e6 / self.taufac), 1)
        # Add in MR distributed Speckle intensities
        if Is > 1e-8 * Ic:
            t, normal = corrsequence(int(self.intt * 1e6 / N), self.tau * 1e6 / N)
            uniform = 0.5 * (special.erf(normal / np.sqrt(2)) + 1)
            t *= N
            f = mr_icdf(Ic, Is, interpmethod=self.interpmethod)
            I = f(uniform) / 1e6
        elif Is >= 0:
            N = max(N, 1000)
            t = np.arange(int(self.intt * 1e6))
            I = Ic / 1e6 * np.ones(t.shape)
        else:
            raise ValueError("Cannot generate a photon list with Is<0.")
        # Add in companion intensities (Gamma distributed or constant)
        if self.gamma_distributed and Ip > 1e-6:
            t2, normal2 = corrsequence(int(self.intt * 1e6 / N), self.tau * 1e6 / N)
            uniform = 0.5 * (special.erf(normal2 / np.sqrt(2)) + 1)
            t2 *= N
            f = gamma_icdf(self.mean_strehl, Ip, bet=self.beta, alph=self.alpha, interpmethod=self.interpmethod)
            I_comp = f(uniform) / 1e6
        else:
            t2 = np.arange(0, int(self.intt * 1e6), N)
            I_comp = np.ones(t2.shape) * Ip / 1e6
        t3 = np.arange(0, int(self.intt * 1e6), N)
        # Number of photons from each distribution in each time bin
        n_mr = np.random.poisson(I * N)
        n_comp = np.random.poisson(I_comp * N)
        n_back = np.random.poisson(np.ones(t3.shape) * self.background_rate / 1e6 * N)
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
        keep = utils.removedeadtime(tlist_tot[indx], self.deadtime)
        # plist tells us which distribution (MR or constant) actually produced a given photon; return this if desired.
        plist1 = np.zeros(tlist.shape).astype(int)
        plist2 = np.ones(tlist_r.shape).astype(int)
        plist3 = np.full_like(tlist_back, 2).astype(int)
        plist_tot = np.concatenate((plist1, plist2, plist3))
        if self.remove_background:
            bkgd_exclude_keep = np.copy(keep)
            bkgd_exclude_keep[np.where(plist_tot[indx] == 2)] = 0
            self.photon_list[x, y] = tlist_tot[indx][np.where(bkgd_exclude_keep)]
            self.photon_flags[x, y] = plist_tot[indx][np.where(bkgd_exclude_keep)]
        else:
            self.photon_list[x, y] = tlist_tot[indx][np.where(keep)]
            self.photon_flags[x, y] = plist_tot[indx]
        self.dts[x, y] = (self.photon_list[x, y][1:] - self.photon_list[x, y][:-1]) * 1e-6
