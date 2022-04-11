"""
TODO: Auto-trim sizes so that they are all the same?

TODO: Add a rate to the  validate_science_target() method (probably in deg/s) to further specify the best regions (in
 time-space) to use for ADI

TODO: Annular normalization to try to just normalize the speckle halo and nothing else
"""
import site
from datetime import timezone
import logging

from astropy.time import Time
from astroplan import Observer
from astropy.coordinates import SkyCoord
import numpy as np
from vip_hci.medsub import median_sub
from scipy.ndimage import rotate
from astropy.io import fits
import matplotlib.pyplot as plt

from mkidpipeline.definitions import MKIDObservation
import mkidpipeline.definitions as mkd
import mkidpipeline.config as mkc

log = logging.getLogger(__name__)

class ADI():
    """
    Runs ADI on supplied FITS cube data using the algorithm defined by "Marois et al.,
    Angular differential Imaging: A Powerful High-contrast Imaging Technique.
    The Astrophysical Journal, 641(1), 556."

    The Vortex Image Processing package for High-Contrast Imaging has this algorithm defined
    under the median_sub function and will be used for this implmentation. There are also
    convenience functions used to help assist users with ADI tuning.
    """

    def __init__(self, in_cube, target_name=None, flip_frames_y=False, times=None, site='subaru',
                 ref_keep:int=None, start_pa_zero=False, angles=None, times_from_obs=None):
        """
        Params:
            - in_cube: 3D NP cube with observation data or fits file for a temporal cube, time must be first dimension.

            - target_name: String with target name (resolvable by SIMBAD, e.g. "HR8799" and "hr 8799" will both work for
            HR 8799) for generating position angles. Will be used if no list of position angles are provided

            - flip_frames_y: The VIP algorithm assumes that the origin of the image is in the bottom left. The drizzler
            is set up so that it puts the origin in the top right. In the case derotation fails (i.e. the PAs are
            'backward') set flip_frames_y to True to orient the frames as they are expected to be input into VIP;s algorithm

            - times: a list of times to use to generate position angles in UNIX (second) format. Will be used if no list
            of position angles are provided

            - site: String of instrument location, defaults to 'subaru' (because of MEC instrument)

            - ref_keep: this will keep every ref_keepth frame. E.g. If you want to keep every 3rd frame, set ref_keep=3

            - start_pa_zero: If true, subtracts the first PA from all other values. This is not recommended for use and
            will be deprecated in a future version.

            - angles: A list of PA values to use for the ADI in degrees. If provided, will be prioritized over any other
            angle values. If not provided, the code will attempt to use time/target/site data to generate a list of angles

            - times_from_obs: Used specifically in the case where no list of times is provided and so the code tries to
            generate a list of times from the header data. This is expected to be a list of times in a 2 column format
            with the first column as the start times and second column as end times for the given dwell steps/stare
            steps. This is most necessary when the duration of frames is not equal to the duration of the dwell steps
            used (e.g. 15s dwell steps broken into 1.5s frames with a 3.0 second offset from the start of each dwell
            step). NOTE: Will be obsoleted in future updates as headers evolve and information can be grabbed so this
            can be automated
        """

        self.input = in_cube

        if isinstance(in_cube, fits.hdu.hdulist.HDUList):
            log.debug(f"Parsing fits file for ADI parameters")
            self.input_type = 'fits'
            self.data = np.copy(self.input[1].data)
        else:
            log.debug(f"Treating input cube as an array with temporal dimension")
            self.input_type = 'array'
            self.data = np.copy(self.input)

        self.target_name = target_name
        self.site = site
        self.obs_start = None
        self.obs_end = None

        self.target_coord = self._generate_target(target_name)
        self.times = self._generate_times(times, dwell_step_ts=times_from_obs)
        self.angles = self._generate_angles(angles)

        for i in range(len(self.data)):
            if flip_frames_y:
                self.data[i] = np.flipud(self.data[i])

        self.kwargs = None  # Will be set after every iteration of run()
        self.ref_psf = None  # Will be set after every iteration of run()
        self.final_res = None  # Will be set after every iteration of run()
        self.keep_frames_for_ref = ref_keep

        if start_pa_zero:
            self.angles -= self.angles[0]

    def _generate_target(self, target_name=None):
        if target_name is not None:
            log.info(f"Using target name: {target_name}")
            target = SkyCoord.from_name(target_name)
        else:
            if self.input_type == 'fits':
                log.info(f"Using fits header to grab target name")
                obj = self.input[0].header['OBJECT']
                log.info(f"Target name from fits header is: {obj}")
                target = SkyCoord.from_name(obj)
            else:
                log.warning("Insufficient data to generate sky coordinates. No target name provided. This may cause "
                            "unintended behavior and the code may not be able to generate its own position angles.")
                target = None
        return target

    def _generate_times(self, time_data=None, dwell_step_ts=None):
        if time_data is not None:
            times = np.array(time_data)
            self.obs_start = times[0]
            self.obs_end = times[1]
        elif time_data is None and self.input_type == "array":
            log.warning(f"No times were specified, if a list of position angles was not provided you may not have "
                        f"sufficient data to perform ADI")
            times = None
        elif time_data is None and self.input_type == "fits":
            if dwell_step_ts is None:
                log.warning(f"Insufficient data to generate time values from fits headers. No time values will be"
                            f" generated. If no position angles were provided, you may not have enough data to perform ADI")
                times = None
            else:
                log.info(f"Generating list of times from user-given dwell step times and fits header info.")

                dwell_time = int(np.average(dwell_step_ts[:, 1] - dwell_step_ts[:, 0]))
                self.obs_start = dwell_step_ts[0][0]
                self.obs_end = dwell_step_ts[-1][1]
                header = self.input[0].header
                t_delta = header['CDELT3']
                t_offset = header['CRVAL3']

                # frames_per_dwell = (dwell_time - t_offset) / t_delta
                # rts = [t_offset + i * t_delta for i in frames_per_dwell]
                rts = generate_relative_times(dwell_time, t_offset, t_delta)
                rts.append(dwell_time)
                rts = np.array(rts)
                relative_times = (rts[:-1] + rts[1:]) / 2

                tvals = []
                for i in dwell_step_ts:
                    tvals.append(i[0] + relative_times)
                times = np.array(tvals).flatten()

        return times

    def _generate_angles(self, angle_data=None):
        if angle_data is not None:
            angles = angle_data
        else:
            log.warning(f"No position angles were provided. Checking that there is sufficient information to generate them")
            if self.target_coord is None and self.times is None:
                raise Exception("Insufficient data to generate angle list. No target name/coordinate or times were provided")
            elif self.target_coord is not None and self.times is None:
                raise Exception("Insufficient data to generate angle list. No times were provided")
            elif self.target_coord is None and self.times is not None:
                raise Exception("Insufficient data to generate angle list. No target name/coordinate was provided")
            else:
                log.info(f"Generating angles for target {self.target_name} at {self.site}")
                site = Observer.at_site(self.site)
                angles = np.array(site.parallactic_angle(Time(val=self.times, format='unix'), self.target_coord).value)
        return angles

    @property
    def fov_rotation(self):
        """
        Returns magnitude of total FOV rotation (in degrees) for the observation
        """
        return np.abs(self.angles.max() - self.angles.min())

    @property
    def obs_duration(self):
        """
        Returns total observation duration (in seconds) based on given start/end times
        """
        return self.obs_end - self.obs_start
    
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

    def run(self, plot=False, **kwargs):
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
        if not isinstance(self.data, np.ndarray):
            raise AttributeError('No data cube input, unable to run ADI!')
        else:
            cube = np.copy(self.data)
            ref_cube = np.copy(self.data)

        self.kwargs = kwargs
        # Calculate the global ref PSF
        if self.keep_frames_for_ref:
            ref_cube = ref_cube[::self.keep_frames_for_ref]
        self.ref_psf = np.median(ref_cube, axis=0)

        # Run ADI
        out_cube, derot_cube, self.final_res = median_sub(cube,
                                                          self.angles,
                                                          full_output=True,
                                                          **self.kwargs)

        if plot:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self.ref_psf)
            ax[1].imshow(self.final_res)
            plt.show()

        return {'unrot_cube': out_cube, 'derot_cube': derot_cube, 'final_res': self.final_res, 'ref_psf': self.ref_psf}

    #def __repr__(self):
    #    kwgs = ', '.join([f'{kwarg}={val}' for kwarg, val in self.kwargs.items()])
    #    obs_times = f'obs_start={self.obs_start}, obs_end={self.obs_end}'
    #    return f'({obs_times}, full_output={self.full_output}, {kwgs})'


def validate_science_target(target, times=None, deltat=None, startt=None, stopt=None):
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
    if times is None:
        if deltat:
            times = np.arange(startt, stopt+deltat, deltat)
        else:
            times = np.arange(startt, stopt+1, 1)
    para_angle = np.array(site.parallactic_angle(Time(val=times, format='unix'), tc).value)

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
    'planet' centered at [ctr+10, ctr+10] with 2700 total counts.
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
                if img_cube[i][ypos][xpos] >= 10:
                    img_cube[i][ypos][xpos] += cts
    return new_pos, img_cube


def img_norm(image):
    return image / np.nansum(image)


def generate_relative_times(duration, start_offset, timestep):
    rts = np.arange(start_offset, duration + timestep, timestep)
    return (rts[:-1] + rts[1:]) / 2


if __name__ == "__main__":
    # NOTE: Some of the code to generate angles and times in this example can also be handled in the ADI class

    targ = 'Hip 36152'  # Name of the target

    # Frames to use (e.g. if you have a 100-frame temporal cube and want to use the middle 20 frames, st=40, end=60,
    # if you want to use the whole range, st=0, end=len(temporal cube))
    st = 0
    end = 70

    cfg_path = '/home/nswimmer/mkidwork/20220222/'  # path to config yaml files
    f = fits.open('/work/nswimmer/20220222/out/hip36152_dither0/hip36152_dither0_adi_out_drizzle.fits')  # name of the fits cube (or the file with the 3D, temporal cube)
    obs_id_string = 'hip36152_'  # see line 399, this will populate the 'times' array with values from the desired obs
    dwell_stept = 15  # duration (in s) of the dwell steps if using a dither

    target = f[0].header['OBJECT']
    # Read in the dither data (only necessary if you want to directly use the times from the metadata
    pipe_cfg = cfg_path + 'pipe.yaml'
    out_cfg = cfg_path + 'out.yaml'
    data_cfg = cfg_path + 'data.yaml'
    mkc.configure_pipeline(pipe_cfg)
    o = mkd.MKIDOutputCollection(out_cfg, datafile=data_cfg)
    d = o.dataset

    # Generate a list of times to use, accounts for if you use a time that is different than the dwell step time
    times = np.array([[i.start, i.stop] for i in d.all_observations if obs_id_string in i.name])
    rt = generate_relative_times(dwell_stept, f[0].header['CRVAL3'], f[0].header['CDELT3'])
    tvals = np.array([[i[0] + rt] for i in times])
    tvals = tvals.flatten()

    adi_cube = f[1].data

    # Generate a list of position angles for the target at each time.
    fpa = validate_science_target(target, tvals)['angles']
    # For reporting, generate a list of angles in degrees that ranges from 0 < theta < 360 (instead of -180 < theta < 180)
    fpa_deg = np.rad2deg(fpa)
    for i in range(len(fpa_deg)):
        if fpa_deg[i] < 0:
            fpa_deg[i] += 360

    adi = ADI(adi_cube[st:end], target, angles=np.rad2deg(fpa[st:end]), times=tvals[st:end], flip_frames_y=True)
    # out = adi.run(**{'mode':'annular'})
    out = adi.run(**{'mode': 'annular', 'radius_int': 5})
    # outs = adi.run()
    outs = adi.run(**{'radius_int': 5})

    plt.figure()
    plt.title(target + ' ADI - ($\Delta$PA=' + f'{abs(fpa_deg[end] - fpa_deg[st]):.2f}' + '$^{\circ}$)\n Annular Mode')
    plt.imshow(np.fliplr(out['final_res']))
    # plt.clim(-30,150)
    plt.colorbar()

    plt.figure()
    plt.title(
        target + ' ADI - ($\Delta$PA=' + f'{abs(fpa_deg[end] - fpa_deg[st]):.2f}' + '$^{\circ}$)\n Full Frame Mode')
    plt.imshow(np.fliplr(outs['final_res']))
    # plt.clim(-25,80)
    plt.colorbar()

    plt.figure()
    plt.title('1.5s frame - ' + target)
    plt.imshow(adi_cube[np.random.randint(st, end)])
    # plt.clim(0, 1000)
    plt.colorbar()

    plt.show()