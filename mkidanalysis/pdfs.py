"""
Define some probability density functions (and cumulative density functions) for fitting histograms in millisecond
exposures
"""

import numpy as np
from scipy.special import factorial
from scipy.optimize.minpack import curve_fit
from scipy.stats import gamma
import scipy
from scipy import special
from scipy.stats import rv_continuous
import matplotlib.pyplot as plt
from scipy import interpolate


def p_I(I, k=None, theta=None, mu=None, I_peak=None):
    """
    Gamma intensity distribution
    :param I:
    :param k:
    :param theta:
    :param mu:
    :param I_peak:
    :return:
    """
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


def mr_icdf(Ic, Is, interpmethod='cubic'):
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
    p_I = modified_rician(I, Ic, Is)

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


def modified_rician(I, Ic, Is):
    '''
    MR pdf(I) = 1/Is * exp(-(I+Ic)/Is) * I0(2*sqrt(I*Ic)/Is)
    mean = Ic + Is
    variance = Is^2 + 2*Ic*Is
    '''
    mr = 1. / Is * np.exp((2 * np.sqrt(I * Ic) - (Ic + I)) / Is) * special.ive(0, 2 * np.sqrt(I * Ic) / Is)
    return mr

def poisson(I,mu):
    #poissonian pdf(I) = e^-mu * mu^I / I!
    pois = np.exp(-1.0*mu) * np.power(mu,I)/factorial(I)
    return pois

def gaussian(I,mu,sig):
    #gaussian pdf(I) = e^(-(x-mu)^2/(2(sig^2)*1/(sqrt(2*pi)*sig)
    gaus = np.exp(-1.0*np.power((I-mu),2)/(2.0*np.power(sig,2))) * 1/(sig*np.sqrt(2*np.pi))
    return gaus

def exponential(x,lam,tau,f0):
    expon = lam*np.exp(-x/tau)+f0
    return expon

def lorentzian(x,gamma,x0):
    loren = (1./(np.pi * gamma))*(gamma*gamma/(np.power((x-x0),2)+gamma*gamma))
    #loren = (1./(np.pi * gamma))*(gamma*gamma/(np.power((x),2)+gamma*gamma))
    return loren


def fit_lorentzian(x, y, guessGam, guessX0):
    '''
    Given a histogram of intensity values (x = I bin centers, y = N(I))
    and a guess for Gamma and x0, returns fit values for Gamma and x0.
    '''
    lor_guess = [guessGam,guessX0]
    lf = lambda fx, gam, x0: lorentzian(fx, gam, x0)
    params, cov = curve_fit(lf, x, y, p0=lor_guess, maxfev=2000)
    return params[0], params[1] #params = [gamma, x0]


def fit_double_lorentzian(x, y, guessGam1, guessX1, guessGam2, guessX2):
    dlor_guess = [guessGam1,guessX1,guessGam2,guessX2]
    dlf = lambda fx, gam1,x1,gam2,x2: lorentzian(fx,gam1,x1)+lorentzian(fx,gam2,x2)
    params,cov = curve_fit(dlf,x,y,p0=dlor_guess,maxfev=2000)
    return params[0], params[1], params[2], params[3] #params = [gamma1, x1, gamma2, x2]


def fit_mr(x, y, guessIc, guessIs):
    '''
    Given a histogram of intensity values (x = I bin centers, y = N(I))
    and a guess for Ic and Is, returns fit values for Ic and Is.
    '''
    mr_guess = [guessIc,guessIs]
    mrf = lambda fx, Ic, Is: modified_rician(fx, Ic, Is)
    params, cov = curve_fit(mrf, x, y, p0=mr_guess, maxfev=2000)
    return params[0], params[1] #params = [fitIc, fitIs]


def fit_poisson(x, y, guessLambda):
    '''
    Given a histogram of intensity values (x = I bin centers, y = N(I))
    and a guess for expectation value, returns fit values for lambda.
    '''
    p_guess = [guessLambda]
    pf = lambda fx, lam: poisson(fx, lam)
    params, cov = curve_fit(pf, x, y, p0=p_guess, maxfev=2000)
    return params[0] #params = [lambda]


def fit_gaussian(x, y, guessMu, guessSigma):
    '''
    Given a histogram of intensity values (x = I bin centers, y = N(I))
    and a guess for mu and sigma, returns fits for mu and sigma
    '''
    g_guess = [guessMu,guessSigma]
    gf = lambda fx, mu, sigma: gaussian(fx, mu, sigma)
    params, cov = curve_fit(gf, x, y, p0=g_guess, maxfev=2000)
    return params[0], params[1] #params = [mu, sigma]


def fit_exponential(x, y, guessLam, guessTau, guessf0):
    '''
    Given a histogram of intensity values (x = I bin centers, y = N(I))
    and a guess for mu and sigma, returns fits for mu and sigma
    '''
    e_guess = [guessLam,guessTau,guessf0]
    ef = lambda fx, lam, tau, f0: exponential(fx, lam,tau,f0)
    params, cov = curve_fit(ef, x, y, p0=e_guess, maxfev=2000)
    return params[0], params[1], params[2] #params = [lambda, tau, f0]

class mr_gen(rv_continuous):
    '''
    Modified Rician distribution for drawing random variates
    Define distribution with mr = mr_gen(). Class already knows (Ic, Is) are shape of PDF for rvs.
    Get random variates with randomSamples = mr.rvs(Ic=Ic, Is=Is, size=N)
    '''
    def _pdf(self, x, Ic, Is):
        return modified_rician(x, Ic, Is)
    def _stats(self, Ic, Is):
        return [Ic+Is, np.power(Is,2)+2*Ic*Is, np.nan, np.nan]

if __name__ == '__main__':
    x = np.arange(200)/100.
    mr = modified_rician(x, 0.5, 0.1)
    p = poisson(x,1.0)
    g = gaussian(x,1.0,0.3)
    c = np.convolve(mr,g,'same')
    plt.plot(x,mr,label="MR")
    plt.plot(x,p,label="Poisson")
    plt.plot(x,g,label="Gaussian")
    plt.plot(x,c/np.max(c),label="MR x G")
    plt.legend()
    plt.show()


