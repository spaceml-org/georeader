from matplotlib import colors
import matplotlib.patches as mpatches
import numpy as np
from georeader.abstract_reader import GeoData
from typing import Optional, List, Union
import matplotlib
import rasterio.plot as rasterioplt


def show(data:GeoData, **kwargs) -> matplotlib.axes.Axes:
    return rasterioplt.show(data.values, transform=data.transform, **kwargs)


def plot_segmentation_mask(mask:Union[GeoData, np.array], color_array:np.array,
                           interpretation_array:Optional[List[str]]=None,
                           legend:bool=True, ax:Optional[matplotlib.axes.Axes]=None) -> matplotlib.axes.Axes:
    """
    Args:
        mask: (H, W) np.array with values from 0 to len(color_array)-1
        color_array: colors for values 0,...,len(color_array)-1 of mask
        interpretation_array: interpretation for classes 0, ..., len(color_array)-1
        legend: plot the legend
        ax: matplotlib.Axes to plot

    """
    cmap_categorical = colors.ListedColormap(color_array)

    norm_categorical = colors.Normalize(vmin=-.5,
                                        vmax=color_array.shape[0] - .5)

    color_array = np.array(color_array)
    if interpretation_array is not None:
        assert len(interpretation_array) == color_array.shape[
            0], f"Different numbers of colors and interpretation {len(interpretation_array)} {color_array.shape[0]}"

    if hasattr(mask, "values"):
        mask_values = mask.values.squeeze()
        transform = mask.transform
    else:
        mask_values = mask
        transform = None

    assert mask_values.ndim == 2, f"Expected 2 D array found {mask_values.shape}"

    ax = rasterioplt.show(mask_values, transform=transform, ax=ax,
                          cmap=cmap_categorical, norm=norm_categorical, interpolation='nearest')

    if legend:
        patches = []
        for c, interp in zip(color_array, interpretation_array):
            patches.append(mpatches.Patch(color=c, label=interp))

        ax.legend(handles=patches,
                  loc='upper right')
    return ax
