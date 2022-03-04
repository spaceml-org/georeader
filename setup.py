from setuptools import setup, find_packages


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires


# Optional Packages
# See https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/
EXTRAS = {
    "all": ["geopandas", "h5py", "zarr"],
    "tests": ["pytest"],
    "docs": [ ],
}

setup(name="georeader",
      version="0.0.1",
      author="Gonzalo Mateo-Garcia",
      packages=find_packages(".", exclude=["tests"]),
      description="Lightweight thead and process save readers for big rasters",
      install_requires=parse_requirements_file("requirements.txt"),
      extras_require=EXTRAS,
      keywords=["raster reading", "rasterio"],
)