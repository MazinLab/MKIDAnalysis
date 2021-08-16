import numpy as np
from vip_hci.medsub import median_sub

class ADI():
    """
    Runs ADI on supplied FITS cube data using the algorithm defined by "Marois et al.,
    Angular differential Imaging: A Powerful High-contrast Imaging Technique.
    The Astrophysical Journal, 641(1), 556."

    The Vortex Image Processing package for High-Contrast Imaging has this algorithm defined
    under the median_sub function and will be used for this implmentation. There are also
    convenience functions used to help assist users with ADI tuning.

    in_cube: 3D numpy array with sequential frames necessary for ADI; Assumes first axis
    is the number of frames, other two are dims of actual frames. Also assumes that the
    the central PSF (center of rotation) is centered in each frame.

    angle_list: 0D numpy array with parallactic angle (in degrees) for each frame
    in the in_cube.

    obs_start: Unix epoch identifying the start of the observation (seconds).

    obs_end: Unix epoch identifying the end of the observation (seconds).

    mkid_obs **TO-DO**: MKIDObservingDataset object used to infer metadata necessary to run ADI
    and give output of convenience functions. Eliminates the need to pass obs start/end times.
    """


    def __init__(self, in_cube, angle_list, obs_start=None, obs_end=None, full_output=False, mkid_obs=None, **kwargs):
        self.in_cube = np.copy(in_cube)
        self.angle_list = angle_list
        self.kwargs = kwargs # kwargs must be valid for the median_sub function from VIP
        self.mkid_obs = mkid_obs
        self.obs_start = obs_start
        self.obs_end = obs_end
        self.full_output = full_output
        self.ref_psf = None # Will be set after every iteration of run()
        self.final_res = None # Will be set after every iteration of run()


    @property
    def fov_rotation(self):
        """
        Calcualates amount of FOV rotation based on given parallactic angle list.
        Useful if using MKIDObservingDataset since it can be used before running
        the MKID Pipeline to determine if there will be enough sky rotation for ADI,
        based on the observation metadata.

        Returns:
            Magnitude of FOV rotation in degrees
        """
        return np.abs(self.angle_list.max() - self.angle_list.min())

    def run(self):
        """
        Runs ADI algorithm with current metadata.

        Returns (if flagged): 
            out_cube: as-is (no-derotation), post-ADI cube
            derot_cube: derotated, post-ADI cube
        """
        # Calculate the global ref PSF to subtract
        # from each frame
        self.ref_psf = np.median(self.in_cube, axis=0)

        # Perform full median and localized annular subraction
        out_cube, derot_cube, self.final_res = median_sub(self.in_cube,
                                                          self.angle_list,
                                                          mode='annular',
                                                          full_output=True,
                                                          **self.kwargs)

        if self.full_output:
            return out_cube, derot_cube


    def __repr__(self):
        kwgs = ', '.join([f'{kwarg}={val}' for kwarg, val in self.kwargs.items()])
        return f'(obs_start={self.obs_start}, obs_end={self.obs_end}, full_output={self.full_output}, mkid_obs={self.full_output}, {kwgs})'