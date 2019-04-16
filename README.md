# MKIDAnalysis
High contrast imaging analysis functions

You will need to enter the astroworld (clone of pipeline.yml, has all the astro things like photutils, astropy, etc)

conda env create -f astroworld.yml
conda activate astroworld

You might also want to install VIP (we are not currently using VIP though so this can wait)
Follow instructions here:  https://vip.readthedocs.io/en/latest/
Use the GIT repository:  git clone https://github.com/vortex-exoplanet/VIP.git

Also install a few dependencies:
pip install git+git://github.com/ericmandel/pyds9.git#egg=pyds9
pip install pyfits
conda install opencv

python setup.py install

Using Isabel’s conda environment ‘astroworld’, created specifically to use photutils but now it has basically every astro package one might need for high/regular contrast astronomy
/home/isabel/.conda/envs/astroworld

import vip_hci as vip

It didn’t work, got error ‘cannot import name '_validate_lengths'’  (seems like a numpy issue)
Googling showed that we need to upgrade scikit-image because that numpy function was deprecated but older scikit-image tries to use it

pip install --upgrade scikit-image

Now we can import VIP
