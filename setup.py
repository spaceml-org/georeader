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


setup(name="georeader",
      version=get_version("georeader/__init__.py"),
      author="Gonzalo Mateo-Garcia",
      long_description=long_description,
      packages=find_packages(".", exclude=["tests"]),
      description="Lightweight reader for raster files",
      install_requires=parse_requirements_file("requirements.txt"),
      extras_require=EXTRAS,
      keywords=["raster reading", "rasterio"],
)
