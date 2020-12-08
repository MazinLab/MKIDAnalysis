import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='mkid analysis repository',
    version='0.0.1',
    author='Mazin Lab',
    author_email='',
    description='Repository for the analysis of MKID data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MazinLab/MKIDAnalysis.git',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ]
)