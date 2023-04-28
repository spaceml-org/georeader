from matplotlib import colors
import matplotlib.patches as mpatches
import numpy as np
from georeader.abstract_reader import GeoData
from typing import Optional, List, Union
import matplotlib.axes
import matplotlib.image
import rasterio.plot as rasterioplt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def colorbar_next_to(im:matplotlib.image.AxesImage, ax:plt.Axes):
    """
    Add a colorbar next to the plot.

    Args:
        im (matplotlib.image.AxesImage): 
        ax (plt.Axes):
    
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.gcf().colorbar(im, cax=cax, orientation='vertical')


def show(data:GeoData, add_colorbar_next_to:bool=False, **kwargs) -> matplotlib.image.AxesImage:
    """Wrapper around rasterio.plot.show that adds a colorbar next to the plot.

    Args:
        data (GeoData): GeoData object to plot with imshow
        add_colorbar_next_to (bool, optional): Defaults to False. Add a colorbar next to the plot

    Returns:
        matplotlib.image.AxesImage: image object
    """
    im = rasterioplt.show(data.values, transform=data.transform, **kwargs)
    if add_colorbar_next_to:
        if "ax" in kwargs:
            ax = kwargs["ax"]
        else:
            ax = plt.gca()
        colorbar_next_to(im, ax)
    return im


def plot_segmentation_mask(mask:Union[GeoData, np.array], color_array:np.array,
                           interpretation_array:Optional[List[str]]=None,
                           legend:bool=True, ax:Optional[matplotlib.axes.Axes]=None) -> matplotlib.image.AxesImage:
    """
    Plots a discrete segmentation mask with a legend.

    Args:
        mask: (H, W) np.array with values from 0 to len(color_array)-1
        color_array: colors for values 0,...,len(color_array)-1 of mask
        interpretation_array: interpretation for classes 0, ..., len(color_array)-1
        legend: plot the legend
        ax: matplotlib.Axes to plot
    
    Returns:
        matplotlib.image.AxesImage

    """
    cmap_categorical = colors.ListedColormap(color_array)

    if ax is None:
        ax = plt.gca()

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

    im = rasterioplt.show(mask_values, transform=transform, ax=ax,
                          cmap=cmap_categorical, norm=norm_categorical, interpolation='nearest')

    if legend:
        patches = []
        for c, interp in zip(color_array, interpretation_array):
            patches.append(mpatches.Patch(color=c, label=interp))

        ax.legend(handles=patches,
                  loc='upper right')
    return im

