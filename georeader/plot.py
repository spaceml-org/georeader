from matplotlib import colors
import matplotlib.patches as mpatches
import numpy as np
from georeader.abstract_reader import GeoData
from typing import Optional, List, Union, Any
import matplotlib.axes
import matplotlib.image
import rasterio.plot as rasterioplt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio.warp
import warnings
import rasterio
try:
    # This only works with shapely>=2.0
    from shapely import Geometry
except ImportError:
    from shapely.geometry.base import BaseGeometry as Geometry
import geopandas as gpd
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.cm


def colorbar_next_to(im:matplotlib.cm.ScalarMappable, ax:plt.Axes, 
                     location:str='right', 
                     pad:float=0.05, 
                     orientation:str='vertical'):
    """
    Add a colorbar next to the plot. 
    
    This function divides the axes and adds a colorbar next to the plot.

    Args:
        im (matplotlib.image.AxesImage): The mappable object (i.e., `.AxesImage`,
            `.ContourSet`, etc.) described by this colorbar.This is the return value from `imshow`.
        ax (plt.Axes): Axes to plot the colorbar
        location (str, optional): Defaults to 'right'. Location of the colorbar. 
            Options are: 'left', 'right', 'top', 'bottom'
        pad (float, optional): Defaults to 0.05. Padding between the plot and the colorbar.
        orientation (str, optional): Defaults to 'vertical'. Orientation of the colorbar. 
            Options are: 'vertical', 'horizontal'.
    
    Example:
        >>> import matplotlib.pyplot as plt
        >>> from georeader import plot
        >>> import numpy as np
        >>> gt = GeoTensor(values=np.random.rand(100,100), transform=rasterio.Affine(1,0,0,0,-1,0), crs="EPSG:4326")
        >>> ax = plt.gca()
        >>> im = ax.imshow(gt.values)
        >>> plot.colorbar_next_to(im, ax)
    
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size='5%', pad=pad)
    plt.gcf().colorbar(im, cax=cax, orientation=orientation)


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
        if ax is None:
            ax = plt.gca()
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
                assert len(mask.shape) in (2, 3), f"mask must be 2D or 3D found shape: {mask.shape}"
                if len(mask.shape) == 3:
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
        from matplotlib.ticker import FuncFormatter

        xmin, ymin, xmax, ymax = data.bounds

        @FuncFormatter
        def x_formatter(x, pos):
            # transform x,ymin to latlng
            longs, lats = rasterio.warp.transform(data.crs, "epsg:4326", [x], [ymin])
            return f"{longs[0]:.2f}"
        

        @FuncFormatter
        def y_formatter(y, pos):
            # transform xmin,y to latlng
            longs, lats = rasterio.warp.transform(data.crs, "epsg:4326", [xmin], [y])
            return f"{lats[0]:.2f}"

        ax.xaxis.set_major_formatter(x_formatter)
        ax.yaxis.set_major_formatter(y_formatter)

    
    return ax

def add_shape_to_plot(shape:Union[gpd.GeoDataFrame, List[Geometry], Geometry], ax:Optional[plt.Axes]=None,
                      crs_plot:Optional[Any]=None,
                      crs_shape:Optional[Any]=None,
                      polygon_no_fill:bool=False,
                      kwargs_geopandas_plot:Optional[Any]=None,
                      title:Optional[str]=None) -> plt.Axes:
    """
    Adds a shape to a plot. It uses geopandas.plot.

    Args:
        shape (Union[gpd.GeoDataFrame, List[Geometry], Geometry]): geodata to plot
        ax (Optional[plt.Axes], optional): Defaults to None. Axes to plot the shape
        crs_plot (Optional[Any], optional): Defaults to None. crs to plot the shape. If None, the crs of the shape is used.
        crs_shape (Optional[Any], optional): Defaults to None. crs of the shape. If None, the crs of the plot is used.
        polygon_no_fill: If True, the polygons are plotted without fill.
        kwargs_geopandas_plot (Optional[Any], optional): Defaults to None. Keyword arguments for geopandas.plot
        title (Optional[str], optional): Defaults to None. Title of the plot.

    Returns:
        plt.Axes: 
    """
    if not isinstance(shape, gpd.GeoDataFrame):
        if isinstance(shape, Geometry):
            shape = [shape]
        shape = gpd.GeoDataFrame(geometry=shape,crs=crs_shape if crs_shape is not None else crs_plot)

    if crs_plot is not None:
        shape = shape.to_crs(crs_plot)
    
    # if color is not None:
    #     if not isinstance(color, str):
    #         assert len(color) == shape.shape[0], "The length of color array must be the same as the number of shapes"
        
    #     color = pd.Series(color, index=shape.index)
    
    if ax is None:
        ax = plt.gca()

    if kwargs_geopandas_plot is None:
        kwargs_geopandas_plot = {}

    if polygon_no_fill:
        shape.boundary.plot(ax=ax, **kwargs_geopandas_plot)
    else:
        shape.plot(ax=ax, **kwargs_geopandas_plot)
    
    if title is not None:
        ax.set_title(title)

    # if legend and color is not None:
    #     color_unique = color.unique()
    #     legend_elements = [Patch(facecolor=color_unique,  label=c) for c in color_unique]
    #     ax.legend(handles=legend_elements)
    
    return ax
    
    

def plot_segmentation_mask(mask:GeoData, color_array:np.array,
                           interpretation_array:Optional[List[str]]=None,
                           legend:bool=True, ax:Optional[plt.Axes]=None,
                           add_scalebar:bool=False,
                           kwargs_scalebar:Optional[dict]=None,
                           bounds_in_latlng:bool=True) -> plt.Axes:
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
        bounds_in_latlng (bool, optional): Defaults to True. If True, the x and y ticks are shown in latlng.
    
    Returns:
        plt.Axes

    """
    cmap_categorical = colors.ListedColormap(color_array)
    color_array = np.array(color_array)
    norm_categorical = colors.Normalize(vmin=-.5,
                                        vmax=color_array.shape[0] - .5)

    
    if interpretation_array is not None:
        assert len(interpretation_array) == color_array.shape[
            0], f"Different numbers of colors and interpretation {len(interpretation_array)} {color_array.shape[0]}"


    ax = show(mask, ax=ax,
              cmap=cmap_categorical, norm=norm_categorical, 
              interpolation='nearest', add_scalebar=add_scalebar,
              kwargs_scalebar=kwargs_scalebar, bounds_in_latlng=bounds_in_latlng)

    if legend:
        patches = []
        for c, interp in zip(color_array, interpretation_array):
            patches.append(mpatches.Patch(color=c, label=interp))

        ax.legend(handles=patches,
                  loc='upper right')
    
    return ax

