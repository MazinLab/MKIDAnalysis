"""
TODO: Make sure that padding image works and expands to a square around your central star

TODO: Add a rate to the  validate_science_target() method (probably in deg/s) to further specify the best regions (in
 time) to use for ADI

TODO: Add image normalization function to take an cube and do some magic on it so that the images are all roughly the
 same brightness
"""
from datetime import timezone

from astropy.time import Time
from astroplan import Observer
from astropy.coordinates import SkyCoord
import numpy as np
from vip_hci.medsub import median_sub

from mkidpipeline.definitions import MKIDObservation
import mkidpipeline.definitions as mkd
import mkidpipeline.config as mkc

class ADI():
    """
    Runs ADI on supplied FITS cube data using the algorithm defined by "Marois et al.,
    Angular differential Imaging: A Powerful High-contrast Imaging Technique.
    The Astrophysical Journal, 641(1), 556."

    The Vortex Image Processing package for High-Contrast Imaging has this algorithm defined
    under the median_sub function and will be used for this implmentation. There are also
    convenience functions used to help assist users with ADI tuning.
    """

    def __init__(self,
                 target,
                 in_cube=None,
                 obs_start=None,
                 obs_end=None,
                 flip_frames_y=False,
                 frame_dur=None,
                 times=None,
                 site='subaru',
                 ref_keep:int=None):
        """
        Attrs:
            -in_cube: 3D NP cube with observation data. number of frames should be the first dimenstion.
            Can be None if using as informational only (haven't run pipeline).
            -kwargs: Keyword arguments (and values) used in the most recent execution of run()
            -obs_start/obs_end: Unix epoch of start and end times
            -flip_frames_y: The VIP algorithm assumes that the origin of the image is in the bottom left. The drizzler
            is set up so that it puts the origin in the top right. In the case derotation fails (i.e. the PAs are
            'backward') set flip_frames_y to True to orient the frames as they are expected to be input into VIP;s algorithm
            -target_coord: Astropy SkyCoord object created for the obs target
            -frame_dur: Duration, in seconds, of each frame in a potential data cube. Used for informational
            purposes only (E.g. Need to determine parallactic angles for some potential cube before running pipeline)
            -site: String of instrument location; For Observer object
            -ref_psf: NP array representing the reference PSF generated by median combining all frames in cube
            -final_res: NP array representing the final residual after running adi

            NOTE: frame_dur and in_cube cannot both be None. If passing values for both (no reason for this),
            in_cube will be preferred.
        """
        if not obs_start and not obs_end and (times is None):
            raise Exception("No times specified at all, ADI cannot run without being given times or a time range.")
        elif (times is not None) and not (obs_start or obs_end):
            obs_start = times[0]
            obs_end = times[-1]

        if in_cube is None:
            raise Exception("Cannot perform ADI without images to perform it on.")

        if isinstance(target, str):
            target = SkyCoord.from_name(target)

        for i in range(len(in_cube)):
            if flip_frames_y:
                in_cube[i] = np.flipud(in_cube[i])

        self.in_cube = in_cube
        self.kwargs = None # Will be set after every iteration of run()
        self.obs_start = obs_start
        self.obs_end = obs_end
        self.target_coord = target
        self.frame_dur = frame_dur
        self.site = site
        self.ref_psf = None # Will be set after every iteration of run()
        self.final_res = None # Will be set after every iteration of run()
        self.keep_frames_for_ref = ref_keep

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
    def from_mkid_obd(cls, mkid_obd, invert_angles=False, frame_dur=None, in_cube=None, site='subaru'):
        """
        Alternate Constructor using MKIDObservingDataset (mkid_obd) object. Pulls information from
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
        
        return cls(obs_start, obs_end, target_coord, invert_angles, frame_dur, in_cube, site)


    def run(self, **kwargs):
        """
        Runs ADI algorithm

        Inputs:
            kwargs that are valid for tuning the VIP median_sub function. See docs:
            https://vip.readthedocs.io/en/latest/vip_hci.medsub.html?highlight=adi#

        Returns: 
            out_cube: as-is (no-derotation), post-ADI cube
            derot_cube: derotated, post-ADI cube
            final_res: combined, residual image
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

        return {'unrot_cube': out_cube, 'derot_cube': derot_cube, 'final_res': self.final_res}


    #def __repr__(self):
    #    kwgs = ', '.join([f'{kwarg}={val}' for kwarg, val in self.kwargs.items()])
    #    obs_times = f'obs_start={self.obs_start}, obs_end={self.obs_end}'
    #    return f'({obs_times}, full_output={self.full_output}, {kwgs})'

def validate_science_target(startt, stopt, target, deltat=None):
    """
    Similar to the _calc_para_angles, used to report the duration of a given observation and how much rotation the FOV
    goes through during that duration. Also provides the corresponding timestamps for convenient plotting. If deltat is
    provided, it will use that as the time between timestamps/PAs, otherwise it will default to 1s intervals. The angles
    returned are in radians NOT DEGREES.
    """
    from astropy.coordinates import SkyCoord
    try:
        tc = SkyCoord.from_name(target)
    except:
        raise Exception('Sky Coord from name not found')
    site = Observer.at_site('Subaru')
    if deltat:
        times = np.arange(startt, stopt+deltat, deltat)
    else:
        times = np.arange(startt, stopt+1, 1)
    para_angle = np.array(site.parallactic_angle(Time(val=times, format='unix'), tc).value)
    # para_angle = np.rad2deg(np.array(site.parallactic_angle(Time(val=times, format='unix'), tc).value))
    # para_angle += 180

    return {'duration': (times[-1] - times[0]), 'angles': para_angle, 'time': times}


def expand_frame(ctr, img, fill_value=np.nan):
    """
    Helper function to take a frame (presumably from an ADI observation) where the center of the PSF is not centered in
    in the image and add space around the edges of the image to make sure that the PSF is centered. This is so that in
    the VIP ADI code, you do not have to hand off a keyword specifying the coordinate where the center of rotation is.
    Alternatively, in adi.run, you can pass {'cxy': [xpos,ypos]} as a kwarg and it will use that as the center to
    rotate around.
    :param ctr: List or tuple in form [x, y] where x and y are the position of the center of the coronagraph in the frame
    :param img: A frame that you wish to expand. Makes no assumptions other than x-axis is the horizontal and y-axis is
    the vertical.
    :param fill_value: The value which the padded space is filled in with. np.nan is default (And recommended).
    :return: The input frame, but with the PSF now in the center, with the expanded space filled in with fill_value.
    """
    img_y_dim, img_x_dim = img.shape
    top_to_y = img_y_dim - ctr[1]
    bottom_to_y = ctr[1]
    right_to_x = img_x_dim - ctr[0]
    left_to_x = ctr[0]
    y_expand = max(top_to_y, bottom_to_y)
    x_expand = max(right_to_x, left_to_x)
    diff = abs(x_expand-y_expand)
    if y_expand == bottom_to_y:
        pad_down = 1
        pad_up = ctr[1] + y_expand - img_y_dim
    else:
        pad_down = abs(ctr[1] - y_expand)
        pad_up = 1
    if x_expand == left_to_x:
        pad_left = 1
        pad_right = ctr[0] + x_expand - img_x_dim
    else:
        pad_left = abs(ctr[0] - x_expand)
        pad_right = 1
    if x_expand > y_expand:
        pad_up += diff
        pad_down += diff
    else:
        pad_left += diff
        pad_right += diff

    return np.pad(img, ((pad_down, pad_up), (pad_left, pad_right)), mode='constant', constant_values=fill_value)


def add_planet(imcube, sep, angles, cts):
    """
    Takes an image cube (typically an ADI sequence) and adds a planet at the given separation (in pixels) at the given
    angle from the list of angles input. Angles is assumed to be given in radians. The 'planet' will be a 3-by-3 square
    with the cts specified added to each pixel in the square (e.g. if cts=300, angle=0, sep=10, then there will be a
    'planet' centered at [ctr+10, ctr+0] with 2700 total counts.
    TODO: Make the planet size changeable and have the shape be changeable as well (gaussian PSF?)
    """
    img_cube = np.copy(imcube)
    new_pos = []
    ctr=np.array(img_cube.shape[1:])/2
    for i in angles:
        planet_pos = [ctr[0]+sep*np.cos(i), ctr[1]+sep*np.sin(i)]
        new_pos.append(planet_pos)

    for i,k in enumerate(new_pos):
        coord = [int(k[0]), int(k[1])]
        x=[j for j in range(int(coord[0])-2, int(coord[0])+3)]
        y=[j for j in range(int(coord[1])-2, int(coord[1])+3)]
        for xpos in x:
            for ypos in y:
                img_cube[i][ypos][xpos] += cts
    return new_pos, img_cube



    return {'duration': (times[-1] - times[0]), 'angles':para_angle, 'p_angles': wo_wrapping}