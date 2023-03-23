from matplotlib import colors
import matplotlib.patches as mpatches
import numpy as np
from georeader.abstract_reader import GeoData
from typing import Optional, List, Union
import matplotlib.axes
import rasterio.plot as rasterioplt

COLORS_CLOUDSEN12 = np.array([[139, 64, 0], # clear
                              [220, 220, 220], # Thick cloud
                              [180, 180, 180], # Thin cloud
                              [60, 60, 60]], # cloud shadow
                             dtype=np.float32) / 255

INTERPRETATION_CLOUDSEN12 = ["clear", "Thick cloud", "Thin cloud", "Cloud shadow"]

INTERPRETATION_DYNAMIC_WORLD = ["water", "trees", "grass", "flooded_vegetation",  "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]
COLORS_DYNAMIC_WORLD_HEX = ["0000FF","007700", "88B053","42f5e0", "E49635", "DFC35A", "C4281B","A59B8F","B39FE1"]
COLORS_DYNAMIC_WORLD = np.array([tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) for c in COLORS_DYNAMIC_WORLD_HEX])/255

COLORS_DYNAMIC_WORLD_CLOUDS_HEX = ["0000FF","007700", "88B053","42f5e0", "E49635", "DFC35A", "C4281B","A59B8F","B39FE1", "000000"]
COLORS_DYNAMIC_WORLD_CLOUDS = np.array([tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) for c in COLORS_DYNAMIC_WORLD_CLOUDS_HEX])/255
INTERPRETATION_DYNAMIC_WORLD_CLOUDS = INTERPRETATION_DYNAMIC_WORLD + ["cloud"]


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


def plot_cloudSEN12mask(mask:Union[np.array, GeoData], legend: bool = True,
                        ax:Optional[matplotlib.axes.Axes]=None) -> matplotlib.axes.Axes:
    """
    See https://cloudsen12.github.io/

    Args:
        mask: (H, W)
        legend: plot the legend
        ax: matplotlib.Axes to plot

    Returns:
        matplotlib.Axes
    """

    return plot_segmentation_mask(mask=mask, color_array=COLORS_CLOUDSEN12,
                                  interpretation_array=INTERPRETATION_CLOUDSEN12, legend=legend, ax=ax)


def plot_DynamicWorldmask(mask: Union[np.array, GeoData], legend: bool = True,
                          ax: Optional[matplotlib.axes.Axes] = None) -> matplotlib.axes.Axes:
    """
    See https://github.com/google/dynamicworld

    Args:
        mask: (H, W)
        legend: plot the legend
        ax: matplotlib.Axes to plot

    Returns:
        matplotlib.Axes
    """

    return plot_segmentation_mask(mask=mask, color_array=COLORS_DYNAMIC_WORLD,
                                  interpretation_array=INTERPRETATION_DYNAMIC_WORLD,
                                  legend=legend, ax=ax)
