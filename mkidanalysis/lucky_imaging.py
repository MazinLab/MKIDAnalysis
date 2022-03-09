import numpy as np
import matplotlib.pyplot as plt
import os
from photutils import CircularAnnulus, aperture_photometry
from mkidpipeline.photontable import Photontable
from mkidcore.instruments import CONEX2PIXEL


def get_file_sizes(path):
    sizes = []
    for i, fn in enumerate(os.listdir(path)):
        size = os.path.getsize(path + fn)
        sizes.append(size)
    return sizes


def flux_in_annulus(image, r_in, r_out, center=None):
    if center is None:
        center = (np.shape(image)[0] / 2, np.shape(image)[1] / 2)
    else:
        center = (center[0] + r_out, center[1] + r_out)
    annulus = CircularAnnulus(center, r_in=r_in, r_out=r_out)
    apers = [annulus]
    mask = np.zeros((np.shape(image)[0] + 2 * r_out, np.shape(image)[1] + 2 * r_out))
    copy = np.copy(mask)
    copy[r_out:-r_out, r_out:-r_out] = image
    image = copy
    annulus_shape = np.shape(annulus.to_mask().data)
    mask[int(center[0] - annulus_shape[0] / 2): int(center[0] + annulus_shape[0] / 2),
    int(center[1] - annulus_shape[1] / 2):int(center[1] + annulus_shape[1] / 2)] = annulus.to_mask().data
    dead_pixels = len(np.where(image[mask.astype(bool)] == 0)[0])
    total_pixels = len(image[mask.astype(bool)])
    live_pixels = total_pixels - dead_pixels
    photometry_table = aperture_photometry(image, apers)
    flux = photometry_table['aperture_sum_0'].data[0] / live_pixels
    return flux


def get_fluxes(frames, r_in, r_out, bin_width, center=None):
    flux_dict = {}
    for i, frame in enumerate(frames):
        flux = flux_in_annulus(frame, r_in, r_out, center=center)
        startt = bin_width * i
        stopt = bin_width * (i + 1)
        flux_dict[flux] = (startt, stopt)
    return flux_dict


def get_time_dict(pt, r_in, r_out, startt=0, duration=None, bin_width=0.02):
    x_con, y_con = pt.query_header('E_CONEXX'), pt.query_header('E_CONEXY')
    slopes = (pt.query_header('E_DPDCX'), pt.query_header('E_DPDCY'))
    ref_pix = (pt.query_header('E_PREFX'), pt.query_header('E_PREFY'))
    ref_con = (pt.query_header('E_CXREFX'), pt.query_header('E_CXREFY'))
    center = CONEX2PIXEL(x_con, y_con, slopes, ref_pix=ref_pix, ref_con=ref_con)
    t_cube = pt.get_fits(start=startt, duration=duration, bin_type='time', cube_type='time', bin_width=bin_width)[
        1].data
    flux_dict = get_fluxes(t_cube, r_in, r_out, bin_width, center=center)
    return flux_dict


def get_bin(cutoff, hist):
    sum = 0
    for i, bin in enumerate(hist):
        sum += bin
        if sum >= cutoff:
            return i


def get_lucky(pt, r_in, r_out, startt=0, duration=None, bin_width=0.02, percent_best=.20):
    flux_dict = get_time_dict(pt, r_in, r_out, startt=startt, duration=duration, bin_width=bin_width)
    vals = [a for a in flux_dict.keys()]
    hist, bin_edges = np.histogram(vals, bins=int(len(vals) / 4))
    cutoff = percent_best * np.sum(hist)
    max_bin = get_bin(cutoff, hist)
    max_flux = bin_edges[max_bin]
    times = []
    for i, (key, val) in enumerate(flux_dict.items()):
        if key <= max_flux:
            times.append(val)
        else:
            pass
    print(len(times))
    return times
