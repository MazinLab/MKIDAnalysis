from astropy.time import Time
from astroplan import Observer
import numpy as np
from vip_hci.medsub import median_sub

from datetime import timezone

from mkidpipeline.config import MKIDObservation


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

    mkid_obs **TO-DO**: MKIDObservingDataset object used to infer metadata necessary to run ADI
    and give output of convenience functions. Eliminates the need to pass obs start/end times.
    """


    def __init__(self, in_cube, angle_list, full_output=False):
        self.in_cube = in_cube # in_cube = None -> built by informational constructor
        self.angle_list = angle_list
        self.kwargs = None
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

    @staticmethod
    def _calc_para_angles(obs_start, obs_end, n_frames, site, target_coord):
        site = Observer.at_site(site)
        times = np.linspace(obs_start, obs_end, n_frames, endpoint=False)
        para_angle = site.parallactic_angle(Time(val=times, format='unix'), target_coord).value
        angle_arr = np.array(para_angle)
        return np.rad2deg(angle_arr)
    
    @staticmethod
    def _get_obs_times(mkid_obd):
        """
        Determines start/end times of an observation, based on the
        defined MKIDObservations in the data.yaml file
        """
        total_exp = 0
        start_times = []
        for mk_obs in mkid_obd.datadict.values():
            if isinstance(mk_obs, MKIDObservation):
                total_exp += mk_obs.duration

                # Converts date from datetime to UNIX epoch in UTC
                start_times.append(mk_obs.date.replace(tzinfo=timezone.utc).timestamp())

        obs_start = np.array(start_times, dtype=float).min()
        obs_end = obs_start + total_exp
        return obs_start, obs_end

    @classmethod
    def from_mkid_obd(cls, mkid_obd, frame_time=None, in_cube=None, site='subaru'):
        """
        Alternate Constructor using MKIDObservingData object. Useful when wanting to know
        information about ADI BEFORE running pipeline (FOV rotation and Parallactic Angles) or
        wanting to have the parallactic angles generated automatically.
        """
        obs_start, obs_end = cls._get_obs_times(mkid_obd)

        # Grab skycoord object from the first MKIDObservation object (All should have same with ADI)
        for mk_obs in mkid_obd.datadict.values():
            if isinstance(mk_obs, MKIDObservation):
                target_coord = mk_obs.skycoord
                break

        # Informational only (pre-pipeline)
        if not isinstance(in_cube, np.ndarray):
            n_frames = (obs_end - obs_start) / frame_time

        # Planning on run ADI (post-pipeline)
        else:
            n_frames = in_cube.shape[0]

        if n_frames % 2 != 0:
            print(f'Non-integer number of frames ({n_frames:.1f}) based on given frame time. Using {int(n_frames)} for number of frames.')

        angle_list = cls._calc_para_angles(obs_start, obs_end, int(n_frames), site, target_coord)

        return cls(in_cube, angle_list)


    def run(self, **kwargs):
        """
        Runs ADI algorithm with current metadata.

        Inputs:
            kwargs that are valid for the VIP median_sub func

        Returns (if flagged): 
            out_cube: as-is (no-derotation), post-ADI cube
            derot_cube: derotated, post-ADI cube
        """
        # Check to see if built by informational constructor or not
        if not isinstance(self.in_cube, np.ndarray):
            raise AttributeError('No data cube input, unable to run ADI!')
        else:
            cube = np.copy(self.in_cube)

        self.kwargs = kwargs
        # Calculate the global ref PSF to subtract
        # from each frame
        self.ref_psf = np.median(self.in_cube, axis=0)

        # Perform full median and localized annular subraction
        out_cube, derot_cube, self.final_res = median_sub(cube,
                                                          self.angle_list,
                                                          full_output=True,
                                                          **self.kwargs)

        if self.full_output:
            return out_cube, derot_cube


    #def __repr__(self):
    #    kwgs = ', '.join([f'{kwarg}={val}' for kwarg, val in self.kwargs.items()])
    #    return f'(full_output={self.full_output}, {kwgs})'