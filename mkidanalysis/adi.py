from datetime import timezone

from astropy.time import Time
from astroplan import Observer
import numpy as np
from vip_hci.medsub import median_sub

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

    def __init__(self,
                 obs_start,
                 obs_end,
                 target_coord,
                 invert_angles=False, #Need to figure out why this is necessary (rotation quirkiness in VIP)
                 frame_dur=None,
                 in_cube=None,
                 full_output=False, 
                 site='subaru'):

        self.in_cube = in_cube
        self.kwargs = None # Will be set after every iteration of run()
        self.obs_start = obs_start
        self.obs_end = obs_end
        self.target_coord = target_coord
        self.invert_angles = invert_angles
        self.frame_dur = frame_dur
        self.full_output = full_output
        self.ref_psf = None # Will be set after every iteration of run()
        self.final_res = None # Will be set after every iteration of run()

        # Informational only (pre-pipeline)
        if not isinstance(in_cube, np.ndarray):
            print('Number of frames in observation deduced from input frame duration...')
            self.n_frames = int((obs_end - obs_start) / frame_dur)
            if (obs_end - obs_start) / frame_dur % 2 != 0:
                print(f'Warning: fractional number of frames using frame_dur of {self.frame_dur}')
                print(f'Using {self.n_frames} frames instead of {((obs_end - obs_start) / frame_dur):.2f}')

        # Planning on run ADI (post-pipeline)
        else:
            print('Number of frames in observation deduced from input cube dims...')
            self.n_frames = in_cube.shape[0]

        self.angle_list = self._calc_para_angles(obs_start, obs_end, self.n_frames, site, target_coord)
        if self.invert_angles:
            self.angle_list *= -1

    @property
    def fov_rotation(self):
        """
        Returns magnitude of total FOV rotation (in degrees) for the observation
        """
        return np.abs(self.angle_list.max() - self.angle_list.min())

    @property
    def obs_duration(self):
        """
        Returns total observation duration (in seconds) based on given start/end times
        """
        return self.obs_end - self.obs_start

    def _calc_para_angles(self, obs_start, obs_end, n_frames, site, target_coord):
        """
        Returns NP array of calculated parallactic angles (in degrees) based on
        frame start times in the given cube.
        """
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
    def from_mkid_obd(cls, mkid_obd, frame_time=None, in_cube=None, full_output=False, site='subaru'):
        """
        Alternate Constructor using MKIDObservingDataset object. Pulls information from
        data.yaml file. TO-DO: Add more functionality to this to pull more information out
        of the MKIDObservingDataset object (e.g. more metadata) to reduce # of input params
        needed for instantiation.
        """
        obs_start, obs_end = cls._get_obs_times(mkid_obd)

        # Grab skycoord object from the first MKIDObservation object (All should be same with ADI)
        for mk_obs in mkid_obd.datadict.values():
            if isinstance(mk_obs, MKIDObservation):
                target_coord = mk_obs.skycoord
                break
        
        return cls(obs_start, obs_end, target_coord, frame_time, in_cube, full_output, site)


    def run(self, **kwargs):
        """
        Runs ADI algorithm

        Inputs:
            kwargs that are valid for the VIP median_sub function

        Returns (if flagged): 
            out_cube: as-is (no-derotation), post-ADI cube
            derot_cube: derotated, post-ADI cube
        """
        # Check to see if informational or not
        if not isinstance(self.in_cube, np.ndarray):
            raise AttributeError('No data cube input, unable to run ADI!')
        else:
            cube = np.copy(self.in_cube)

        self.kwargs = kwargs
        # Calculate the global ref PSF
        self.ref_psf = np.median(cube, axis=0)

        # Run ADI
        out_cube, derot_cube, self.final_res = median_sub(cube,
                                                          self.angle_list,
                                                          full_output=True,
                                                          **self.kwargs)

        if self.full_output:
            return out_cube, derot_cube


    #def __repr__(self):
    #    kwgs = ', '.join([f'{kwarg}={val}' for kwarg, val in self.kwargs.items()])
    #    obs_times = f'obs_start={self.obs_start}, obs_end={self.obs_end}'
    #    return f'({obs_times}, full_output={self.full_output}, {kwgs})'