import os
import re
import shutil
import logging
from datetime import datetime
from typing import Tuple, List, Optional
import numpy as np
import json
from georeader.readers.download_utils import download_product as download_product_base

try:
    import requests
    from lxml import html
    from requests.auth import HTTPBasicAuth
except ImportError:
    raise ImportError("Please install requests and lxml with 'pip install requests lxml'")


# Here you have the free data main
# https://www.vito-eodata.be/PDF/datapool/Free_Data/


RESOLUTIONS_LINKS = {
    "333M": "https://www.vito-eodata.be/PDF/datapool/Free_Data/PROBA-V_300m/L2A_-_300_m_C1/%s/%s/%s/",
    "333M_TOC": "https://www.vito-eodata.be/PDF/datapool/Free_Data/PROBA-V_300m/S1_TOC_-_300_m_C1/%s/%s/%s/",
    "333M_TOA": "https://www.vito-eodata.be/PDF/datapool/Free_Data/PROBA-V_300m/S1_TOA_-_300_m_C1/%s/%s/%s/",
    "100M": "https://www.vito-eodata.be/PDF/datapool/Free_Data/PROBA-V_100m/L2A_-_100_m_C1/%s/%s/%s/",
    "100M_TOA": "https://www.vito-eodata.be/PDF/datapool/Free_Data/PROBA-V_100m/S1_TOA_100_m_C1/%s/%s/%s/",
    "100M_TOC": "https://www.vito-eodata.be/PDF/datapool/Free_Data/PROBA-V_100m/S1_TOC_100_m_C1/%s/%s/%s/",
    "1KM": "https://www.vito-eodata.be/PDF/datapool/Free_Data/PROBA-V_1km/L2A_-_1_km_C1/%s/%s/%s/"
}


def get_auth():
    home_dir = os.path.join(os.path.expanduser('~'),".georeader")
    json_file = os.path.join(home_dir, "auth_probav.json")
    if not os.path.exists(json_file):
        os.makedirs(home_dir, exist_ok=True)
        with open(json_file, "w") as fh:
            json.dump({"user": "SET-USER", "password": "SET-PASSWORD"}, fh)

        raise FileNotFoundError(f"In order to download Proba-V images add user and password to file : {json_file}")

    with open(json_file, "r") as fh:
        data = json.load(fh)
    
    if data["user"] == "SET-USER":
        raise FileNotFoundError(f"In order to download Proba-V images add user and password to file : {json_file}")

    return HTTPBasicAuth(data["user"], data["password"])


def download_product(link_down:str, filename:Optional[str]=None,
                     display_progress_bar:bool=True) -> str:
    """
    Download a product from the probav website

    Args:
        link_down: link to the product
        filename: filename to save the product
        display_progress_bar: display tqdm progress bar

    Example:
        >>> link_down = "https://www.vito-eodata.be/PDF/datapool/Free_Data/PROBA-V_100m/S1_TOA_100_m_C1/2019/2/9/PV_S1_TOA-20190209_100M_V101/PROBAV_S1_TOA_X07Y05_20190209_100M_V101.HDF5"
        >>> filename = download_product(link_down)
    """
    auth_probav = get_auth()
    return download_product_base(link_down, filename=filename, auth=auth_probav,
                                  display_progress_bar=display_progress_bar)


def fetch_products_date_region(date: datetime, bounding_box: Tuple[float, float, float, float],
                               product_name: str = "333M") -> List[str]:
    assert product_name in RESOLUTIONS_LINKS, f"{product_name} not found in {RESOLUTIONS_LINKS.keys()}"

    resolution_link = RESOLUTIONS_LINKS[product_name] % (date.year, date.month, date.day)
    params = "?coord=%s,%s,%s,%s" %  tuple(bounding_box)#tuple([round(i,2) for i in bounding_box])
    resolution_link += params
    auth_probav = get_auth()

    r = requests.get(resolution_link, auth=auth_probav)

    if r.status_code != 200:
        raise FileNotFoundError("page %s cant be retrieved status %d" % (resolution_link, r.status_code))

    tree = html.fromstring(r.content)
    links = [a_elm.attrib["href"] for a_elm in tree.xpath("//a") if
             "href" in a_elm.attrib and "PROBA-V" in a_elm.attrib["href"]]

    logging.info("Found %d links for given date and bounding box" % len(links))

    links_all = []
    for l in links:
        r = requests.get(l, auth=auth_probav)
        tree = html.fromstring(r.content)
        if r.status_code != 200:
            print(f"ERROR retrieving link {l} continue")
            continue

        lbase = os.path.dirname(l)
        links_hdf5_files = [os.path.join(lbase, a_elm.attrib["href"]) for a_elm in tree.xpath("//a") if
                            "href" in a_elm.attrib and a_elm.attrib["href"].endswith(".HDF5")]
        links_all.extend(links_hdf5_files)

    return links_all


def download_L2A_date_region(date, bounding_box, dir_out, resolution="333M", only_xml=False):
    """
    https://www.vito-eodata.be/PDF//image/Data_pool_manual.pdf

    :param date:
    :param bounding_box: list  XLL,YLL,XUR,YUR
    :param dir_out:
    :param resolution:
    :param only_xml:

    :return:
    """

    resolution_link = RESOLUTIONS_LINKS[resolution] % (date.year, date.month, date.day)
    params = "?coord=%s,%s,%s,%s"%tuple(bounding_box)
    resolution_link+=params
    auth_probav = get_auth()

    r = requests.get(resolution_link,
                     auth=auth_probav)
    if r.status_code != 200:
        raise FileNotFoundError("page %s cant be retrieved status %d" % (resolution_link, r.status_code))

    tree = html.fromstring(r.content)
    links = [a_elm.attrib["href"] for a_elm in tree.xpath("//a") if
             "href" in a_elm.attrib and "PROBA-V" in a_elm.attrib["href"]]

    files_down = []
    logging.info("Found %d links for given date and bounding box"%len(links))
    for link_down in links:
        link_down = link_down.replace(params, "")
        product_link_name = link_down.split("/")[-2]
        matches = re.match(r"PV_(LEFT|RIGHT|CENTER)_L2A-(\d{4})(\d{2})(\d{2})(\d{6})_(\d..?M)_(V\d0\d)",
                           product_link_name)
        if matches is None:
            logging.warning("link %s does not follow the expected naming pattern. skip download"%product_link_name)
            continue
        camera_string, year, month, day, id_lookup, resolution, version = matches.groups()
        camera_int = CAMERAS.index(camera_string) + 1

        filename = download_L2A_product(year, month, day, id_lookup, camera_int, resolution,
                                        version, dir_out, only_xml=only_xml)

        if filename is not None:
            files_down.append(filename)

    return files_down


CAMERAS = ["LEFT", "CENTER", "RIGHT"]


def is_downloadable(url, auth=None):
    """
    Does the url contain a downloadable resource
    """
    h = requests.head(url, allow_redirects=True,auth=auth)
    header = h.headers
    content_type = header.get('content-type')
    if not 'xml' in content_type.lower():
        if 'text' in content_type.lower():
            return False
        if 'html' in content_type.lower():
            return False
    return True


def download_L2A_product(year, month, day, id_lookup, camera_int, resolution, version, dir_out, only_xml=False):
    """

    :param year:
    :param month:
    :param day:
    :param id_lookup:
    :param camera_int:
    :param resolution:
    :param version:
    :param dir_out
    :param only_xml:


    :return:
    """
    product_name = "PROBAV_L2A_%s%s%s_%s_%s_%s_%s" % (str(year), str(month), str(day), str(id_lookup),
                                                           str(camera_int), resolution, version)

    if only_xml:
        product_name += ".xml"
    else:
        product_name += ".HDF5"

    filename = os.path.join(dir_out, product_name)
    if os.path.exists(filename):
        logging.warning("..... file %s exists will NOT overwrite!!" % filename)
        return filename

    auth_probav = get_auth()
    resolution_link = RESOLUTIONS_LINKS[resolution] % (year, month, day)
    product_link_name = "PV_%s_L2A-%s%s%s%s_%s_%s" % (CAMERAS[int(camera_int) - 1],
                                                      str(year), str(month), str(day), str(id_lookup),
                                                      resolution, version)
    link_down = resolution_link + product_link_name + '/' + product_name

    logging.info("Downloading file %s from link %s" % (product_name,link_down))

    if not is_downloadable(link_down,auth_probav):
        raise FileNotFoundError("......Failed %s download \n link download %s is not a file"%(product_name,link_down))


    r_link = requests.get(link_down, stream=True, auth=auth_probav)

    if r_link.status_code == 200:
        with open(filename, 'wb') as f:
            r_link.raw.decode_content = True
            shutil.copyfileobj(r_link.raw, f)
    else:
        raise FileNotFoundError(
            "......Failed %s download status: %d\n link download %s" % (product_name, r_link.status_code, link_down))

    return filename


def download_L2A_product_from_name(product_name, dir_out):
    """

    :param product_name: name of the product to download
    :param dir_out:

    # Download same products 1km
    pv100mprods = [probav_image_operational.ProbaVImageOperational(f) for f in glob("/media/disk/databases/PROBAV_CLOUDS/CLOUDSV3/100m_UCLcce/*.HDF5")]
    for pv100m in pv100mprods:
        download_pv_product.download_L2A_product_from_name(pv100m.name.replace("100M","1KM"),"/media/disk/databases/PROBAV_CLOUDS/CLOUDSV3/1km_UCLcce/")

    :return:
    """
    if exists_product_name(product_name, dir_out):
        return

    fields = extract_L2_file_naming_content(product_name)
    return download_L2A_product(fields["year"], fields["month"], fields["day"], fields["id_lookup"], fields["camera"],
                                fields["resolution"], fields["version"], dir_out)


def download_L2A_xml_from_name(product_name, dir_out):
    """

    :param product_name: name of the product to download
    :param dir_out:

    # Download same products 1km
    pv100mprods = [probav_image_operational.ProbaVImageOperational(f) for f in glob("/media/disk/databases/PROBAV_CLOUDS/CLOUDSV3/100m_UCLcce/*.HDF5")]
    for pv100m in pv100mprods:
        download_pv_product.download_L2A_product_from_name(pv100m.name.replace("100M","1KM"),"/media/disk/databases/PROBAV_CLOUDS/CLOUDSV3/1km_UCLcce/")

    :return:
    """
    if exists_product_name(product_name, dir_out):
        return

    fields = extract_L2_file_naming_content(product_name)
    return download_L2A_product(fields["year"], fields["month"], fields["day"], fields["id_lookup"], fields["camera"],
                                fields["resolution"], fields["version"], dir_out, only_xml=True)


def exists_product_name(product_name, dir_out):
    filename = os.path.join(dir_out, product_name)
    if os.path.exists(filename):
        logging.warning("..... file %s exists will NOT overwrite!!" % filename)

    return os.path.exists(filename)


def extract_L2_file_naming_content(product_name):
    matches = re.match(r"PROBAV_L2A_(\d{4})(\d{2})(\d{2})_(\d{6})_(\d)_(\d..?M)_(V\d0\d)", product_name)
    if matches is None:
        raise ValueError("..... file %s does not follow L2A file naming" % product_name)

    year, month, day, id_lookup, camera, resolution, version = matches.groups()
    fields = {
        "year":year,
        "month":month,
        "day":day,
        "id_lookup": id_lookup,
        "camera":camera,
        "resolution":resolution,
        "version":version
    }

    return fields


def read_L2A_xml(product_name, dir_out):

    def idRowMatch(content, tag):
        id_ = np.argwhere([True if row.find(tag) != -1 else False for row in content])[0][0]
        return id_

    def getValueFromAttr(string):
        string = string.strip()
        value = (string.split(">")[1]).split("</")[0]
        value = value.strip().split(" ")
        if len(value) == 1:
            value = value[0]

        return value

    xml_file = os.path.join(dir_out, product_name+".xml")
    if os.path.exists(xml_file):
        download_L2A_xml_from_name(product_name, dir_out)

    with open(xml_file, "r") as file:
        content = [row for row in file]

    metadata = {}
    metadata["bbox"]      = getValueFromAttr(content[idRowMatch(content, "BoundingBox") + 1])
    metadata["polygon"]   = getValueFromAttr(content[idRowMatch(content, "gml:posList")])
    metadata["land_pct"]  = getValueFromAttr(content[idRowMatch(content, "LandPercentage") + 1])
    metadata["miss_pct"]  = getValueFromAttr(content[idRowMatch(content, "MissingDataPercentage") + 1])
    metadata["cloud_pct"] = getValueFromAttr(content[idRowMatch(content, "cloudCoverPercentage")])
    metadata["snow_pct"]  = getValueFromAttr(content[idRowMatch(content, "snowCoverPercentage")])
    metadata["water_pct"] = str(round(100 - float(metadata["land_pct"]) - float(metadata["miss_pct"]), 3))
    return metadata