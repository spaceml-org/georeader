import os
import shutil
from typing import Optional, Any

try:
    import requests
except ImportError:
    raise ImportError("Please install requests with 'pip install requests'")

def download_product(link_down:str, filename:Optional[str]=None, auth:Any=None, 
                     display_progress_bar:bool=True, verify:bool=True, 
                     headers:Optional[Any]=None) -> str:
    """
    Download a product from a link

    Args:
        link_down (str): Link to download the product
        filename (Optional[str], optional): Filename to save the product. Defaults to None.
        auth (Any, optional): Authentication to download the product. Defaults to None.
        display_progress_bar (bool, optional): Display a progress bar. Defaults to True.
        verify (bool, optional): Verify the SSL certificate. Defaults to True.
        headers (Optional[Any], optional): Headers to download the product. Defaults to None.

    Returns:
        str: Filename of the downloaded product
    
    Raises:
        requests.exceptions.HTTPError: If the download fails.
    
    Example:
        >>> # Download a Proba-V image
        >>> from georeader.readers.download_utils import download_product
        >>> from georeader.readers import download_pv_product
        >>> auth = download_pv_product.get_auth()
        >>> link_down = "https://www.vito-eodata.be/PDF/datapool/Free_Data/PROBA-V_100m/S1_TOA_100_m_C1/2019/2/9/PV_S1_TOA-20190209_100M_V101/PROBAV_S1_TOA_X07Y05_20190209_100M_V101.HDF5"
        >>> filename = download_product(link_down, auth=auth)
    """
    from tqdm import tqdm

    if filename is None:
        filename = os.path.basename(link_down)

    if os.path.exists(filename):
        print(f"File {filename} exists. It won't be downloaded again")
        return filename

    filename_tmp = filename+".tmp"

    with requests.get(link_down, stream=True, auth=auth, verify=verify, headers=headers, allow_redirects=True) as r_link:
        total_size_in_bytes = int(r_link.headers.get('content-length', 0))
        r_link.raise_for_status()
        block_size = 8192  # 1 Kibibyte
        with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, disable=not display_progress_bar) as progress_bar:
            with open(filename_tmp, 'wb') as f:
                for chunk in r_link.iter_content(chunk_size=block_size):
                    if display_progress_bar:
                        progress_bar.update(len(chunk))
                    f.write(chunk)

    shutil.move(filename_tmp, filename)

    return filename