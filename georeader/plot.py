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
import rasterio.warp
import warnings
import rasterio

rasterio.Affine

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


def show(data:GeoData, add_colorbar_next_to:bool=False,
         add_scalebar:bool=False,
         kwargs_scalebar:Optional[dict]=None,
         mask:Union[bool,np.array]= False, bounds_in_latlng:bool=True,
           **kwargs) -> plt.Axes:
    """
    Wrapper around rasterio.plot.show for GeoData objects. It adds options to add a colorbar next to the plot
    and a scalebar showing the geographic scale.

    Args:
        data (GeoData): GeoData object to plot with imshow
        add_colorbar_next_to (bool, optional): Defaults to False. Add a colorbar next to the plot
        add_scalebar (bool, optional): Defaults to False. Add a scalebar to the plot
        kwargs_scalebar (Optional[dict], optional): Defaults to None. Keyword arguments for the scalebar. 
        See https://github.com/ppinard/matplotlib-scalebar. (install with pip install matplotlib-scalebar)
        mask (Union[bool,np.array], optional): Defaults to False. Mask to apply to the data. 
            If True, the fill_value_default of the GeoData is used.
        bounds_in_latlng (bool, optional): Defaults to True. If True, the x and y ticks are shown in latlng.
        **kwargs: Keyword arguments for imshow

    Returns:
        plt.Axes: image object
    """
    if "ax" in kwargs:
        ax = kwargs.pop("ax")
    else:
        ax = plt.gca()
    
    if isinstance(mask, bool):
        if mask:
            mask = data.values == data.fill_value_default
            np_data = np.ma.masked_array(data.values, mask=mask)
        else:
            mask = None
            np_data = data.values
    else:
        np_data = np.ma.masked_array(data.values, mask=mask)

    if len(np_data.shape) == 3:
        if np_data.shape[0] == 1:
            np_data = np_data[0]
        else:
            np_data = np_data.transpose(1, 2, 0)

            if mask is not None:
                mask = np.any(mask, axis=0)
                # Convert np_data to RGBA using mask as alpha channel.
                np_data = np.concatenate([np_data, ~mask[..., None]], axis=-1)

    xmin, ymin, xmax, ymax = data.bounds
    # kwargs['extent'] = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    # xmin, ymin, xmax, ymax
    kwargs['extent'] = (xmin, xmax, ymin, ymax)

    if not data.transform.is_rectilinear:
        warnings.warn("The transform is not rectilinear. The x and y ticks and the scale bar are not going to be correct."
                      " To discard this warning use: warnings.filterwarnings('ignore', message='The transform is not rectilinear.')")
    
    title = None
    if "title" in kwargs:
        title = kwargs.pop("title")
    
    ax.imshow(np_data, **kwargs)

    if title is not None:
        ax.set_title(title)
    
    if add_colorbar_next_to:
        im = ax.images[-1]
        colorbar_next_to(im, ax)
    
    if add_scalebar:
        try:
             from matplotlib_scalebar.scalebar import ScaleBar
        except ImportError as e:
            raise ImportError("Install matplotlib-scalebar to use scalebar"
                              "pip install matplotlib-scalebar"
                              f"{e}")
        
        if kwargs_scalebar is None:
            kwargs_scalebar = {"dx":1}
        if "dx" not in kwargs_scalebar:
            kwargs_scalebar["dx"] = 1
        ax.add_artist(ScaleBar(**kwargs_scalebar))
    
    if bounds_in_latlng:
        xmin, ymin, xmax, ymax = rasterio.warp.transform_bounds(data.crs, "epsg:4326", *data.bounds)
        warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator")
        ax.set_xticklabels([f"{x:.2f}" for x in np.linspace(xmin, xmax, len(ax.get_xticks()))])
        ax.set_yticklabels([f"{x:.2f}" for x in np.linspace(ymin, ymax, len(ax.get_yticks()))])

        # ax.set_xticks(np.linspace(xmin, xmax, len(ax.get_xticks())))
        # ax.set_yticks(np.linspace(ymin, ymax, len(ax.get_yticks())))
    
    return ax


def plot_segmentation_mask(mask:Union[GeoData, np.array], color_array:np.array,
                           interpretation_array:Optional[List[str]]=None,
                           legend:bool=True, ax:Optional[plt.Axes]=None,
                           add_scalebar:bool=False,
                           kwargs_scalebar:Optional[dict]=None) -> plt.Axes:
    """
    Plots a discrete segmentation mask with a legend.

    Args:
        mask: (H, W) np.array with values from 0 to len(color_array)-1
        color_array: colors for values 0,...,len(color_array)-1 of mask
        interpretation_array: interpretation for classes 0, ..., len(color_array)-1
        legend: plot the legend
        ax: plt.Axes to plot
        add_scalebar (bool, optional): Defaults to False. Add a scalebar to the plot
        kwargs_scalebar (Optional[dict], optional): Defaults to None. Keyword arguments for the scalebar. 
        See https://github.com/ppinard/matplotlib-scalebar. (install with pip install matplotlib-scalebar)
    
    Returns:
        plt.Axes

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

    rasterioplt.show(mask_values, transform=transform, ax=ax,
                     cmap=cmap_categorical, norm=norm_categorical, interpolation='nearest')

    if legend:
        patches = []
        for c, interp in zip(color_array, interpretation_array):
            patches.append(mpatches.Patch(color=c, label=interp))

        ax.legend(handles=patches,
                  loc='upper right')
    
    if add_scalebar:
        assert transform is not None, "Cannot show scalebar without transform"
        try:
             from matplotlib_scalebar.scalebar import ScaleBar
        except ImportError as e:
            raise ImportError("Install matplotlib-scalebar to use scalebar"
                              "pip install matplotlib-scalebar"
                              f"{e}")
        
        if kwargs_scalebar is None:
            kwargs_scalebar = {"dx":1}
        if "dx" not in kwargs_scalebar:
            kwargs_scalebar["dx"] = 1
        
        ax.add_artist(ScaleBar(**kwargs_scalebar))

    return ax

