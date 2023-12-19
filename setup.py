from setuptools import setup, find_packages
import codecs
import os.path

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIREMENTS_EXTRA =["scikit-image", "fsspec"]
REQUIREMENTS_GOOGLE = ["gcsfs", "google-cloud-storage", "earthengine-api"]
REQUIREMENTS_TORCH = ["torch", "torchvision"]
REQUIREMENTS_PLANETARY_COMPUTER = ["pystac-client", "planetary-computer"]
REQUIREMENTS_PROBAV = ["h5py", "requests", "tqdm", "lxml"]
REQUIREMENTS_SCIHUB = ["sentinelsat"]
REQUIREMENTS_EMIT = ["netcdf4"]


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires


# Optional Packages
# See https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/
EXTRAS = {
    "all": REQUIREMENTS_EXTRA + REQUIREMENTS_GOOGLE + REQUIREMENTS_TORCH +
           REQUIREMENTS_PLANETARY_COMPUTER + REQUIREMENTS_PROBAV+ REQUIREMENTS_SCIHUB + REQUIREMENTS_EMIT,
    "google": REQUIREMENTS_EXTRA + REQUIREMENTS_GOOGLE,
    "torch": REQUIREMENTS_TORCH,
    "microsoftplanetary": REQUIREMENTS_EXTRA + REQUIREMENTS_PLANETARY_COMPUTER,
    "sentinel2": REQUIREMENTS_EXTRA,
    "probav": REQUIREMENTS_PROBAV,
    "scihub": REQUIREMENTS_SCIHUB,
    "emit": REQUIREMENTS_EMIT,
    "tests": ["pytest"],
    "docs": [ ],
}

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(name="georeader-spaceml",
      version=get_version("georeader/__init__.py"),
      author="Gonzalo Mateo-Garcia",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(".", exclude=["tests"]),
       package_data={
        "georeader" : ["SolarIrradiance_Thuillier.csv"]
       },
      description="Lightweight reader for raster files",
      install_requires=parse_requirements_file("requirements.txt"),
      url="https://github.com/spaceml-org/georeader",
      classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: GIS",
        "Development Status :: 5 - Production/Stable", 
    ],
      extras_require=EXTRAS,
      keywords=["raster reading", "rasterio"],
)
