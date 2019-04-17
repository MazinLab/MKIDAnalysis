import numpy as np
from scipy import ndimage
from mkidcore.instruments import CONEX2PIXEL
from scipy.interpolate import griddata


def ditherp_2_pixel(positions):
    """ A function to convert the connex offset to pixel displacement"""
    positions = np.asarray(positions)
    pix = np.asarray(CONEX2PIXEL(positions[:, 0], positions[:, 1])) - np.array(CONEX2PIXEL(0, 0)).reshape(2, 1)
    return pix


def linear_func(x, slope, intercept):
    return x * slope + intercept


def upsample_image(image, npix):
    """
    Upsamples the array so that the rotation and shift can be done to subpixel precision
    each pixel will be converted in a square of nPix*nPix
    """
    upsampled = image.repeat(npix, axis=0).repeat(npix, axis=1)
    return upsampled


def median_stack(stack):
    return np.nanmedian(stack, axis=0)


def mean_stack(stack):
    return np.nanmean(stack, axis=0)


def negatives_to_nans(image):
    image = image.astype(float)
    image[image < 0] = np.nan
    return image


def dist(yc, xc, y1, x1):
    """
    Return the Euclidean distance between two points.
    """
    return np.sqrt((yc - y1) ** 2 + (xc - x1) ** 2)


def embed_image(image, framesize=1, pad_value=-1):
    """
    Gets a numpy array and -1-pads it. The frame size gives the dimension of the frame in units of the
    biggest dimension of the array (if image.shape gives (2,4), then 4 rows of -1s will be added before and
    after the array and 4 columns of -1s will be added before and after. The final size of the array will be (10,12))
    It is padding with -1 and not 0s to distinguish the added pixels from valid pixels that have no photons. Masked pixels
    (dead or hot) are nan
    It returns a numpy array
    """
    frame_pixsize = int(max(image.shape) * framesize)
    padded_array = np.pad(image, frame_pixsize, 'constant', constant_values=pad_value)
    return padded_array


def rotate_shift_image(image, degree, xshift, yshift):
    """
    Rotates the image counterclockwise and shifts it in x and y
    When shifting, the pixel that exit one side of the array get in from the other side. Make sure that
    the padding is large enough so that only -1s roll and not real pixels
    """
    ###makes sure that the shifts are integers
    xshift = int(round(xshift))
    yshift = int(round(yshift))
    rotated_image = ndimage.rotate(image, degree, order=0, cval=-1, reshape=False)
    rotated_image = negatives_to_nans(rotated_image)

    xshifted_image = np.roll(rotated_image, xshift, axis=1)
    rotated_shifted = np.roll(xshifted_image, yshift, axis=0)
    return rotated_shifted


def interpolate_image(image, method='linear'):
    '''
    2D interpolation to smooth over missing pixels using built-in scipy methods

    INPUTS:
        image - 2D input array of values
        method - method of interpolation. Options are scipy.interpolate.griddata methods:
                 'linear' (default), 'cubic', or 'nearest'

    OUTPUTS:
        the interpolated image with same shape as input array
    '''

    finalshape = np.shape(image)

    datapoints = np.where(
        np.logical_or(np.isnan(image), image == 0) == False)  # data points for interp are only pixels with counts
    data = image[datapoints]
    datapoints = np.array((datapoints[0], datapoints[1]),
                          dtype=np.int).transpose()  # griddata expects them in this order

    interppoints = np.where(image != np.nan)  # should include all points as interpolation points
    interppoints = np.array((interppoints[0], interppoints[1]), dtype=np.int).transpose()

    interpolated_frame = griddata(datapoints, data, interppoints, method)
    interpolated_frame = np.reshape(interpolated_frame, finalshape)  # reshape interpolated frame into original shape

    return interpolated_frame