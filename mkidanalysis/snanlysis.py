from mkidpipeline.hdf.photontable import ObsFile
from regions import CirclePixelRegion, PixCoord
import numpy as np
import matplotlib.pyplot as plt

def getControlRegionMask(xs, xe, ys, ye, xydim=(140,146)):
    mask = np.zeros(xydim).astype(bool)
    mask[xs:xe, ys:ye] = True
    return mask

def getApertureMask(xPos, yPos, nPixPerLD, xydim=(140,146)): 
    rcdim = (xydim[1], xydim[0])
    center = PixCoord(xPos, yPos)
    apertureRegion = CirclePixelRegion(center, nPixPerLD/2.)
    exactApertureMask = apertureRegion.to_mask('center').to_image(rcdim)
    return exactApertureMask.T

def getDarkFrameCPS(darkh5, start=0, inttime=None):
    obsfile = ObsFile(darkh5)
    img = obsfile.getPixelCountImage(firstSec=start, integrationTime=inttime)
    img['image'] /= np.max(img['effIntTime'])
    return img['image']

def getControlRegionApertureList(xs, xe, ys, ye, nPixPerLD, xOffs=0, yOffs=0, xydim=(140,146)):
    assert(xOffs < nPixPerLD)
    assert(xs < xe)
    assert(ys < ye)
    xCenterRange = np.array([max(xs + nPixPerLD/2., xs + nPixPerLD/2. + xOffs), 
                    min(xe - nPixPerLD/2., xe - nPixPerLD/2. + xOffs)])
    yCenterRange = np.array([max(ys + nPixPerLD/2., ys + nPixPerLD/2. + yOffs), 
                    min(ye - nPixPerLD/2., ye - nPixPerLD/2. + yOffs)])
    
    xCenterList = np.arange(xCenterRange[0], xCenterRange[1], nPixPerLD)
    yCenterList = np.arange(yCenterRange[0], yCenterRange[1], nPixPerLD)
    coords = []
    apertureMasks = []

    for xc in xCenterList:
        for yc in yCenterList:
            apertureMasks.append(getApertureMask(xc, yc, nPixPerLD, xydim))
            coords.append((xc, yc))

    return apertureMasks, coords

def getLightCurvesFromApertureList(apMaskList, dataCube):
    # datacube is assumed to have shape (x, y, nFrames)
    lcList = []
    for mask in apMaskList:
        lcList.append(np.nansum(dataCube[mask.astype(bool)], axis=0))

    return lcList

    

    
    

