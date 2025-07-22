import pandas as pd
from typing import Optional, Iterable, Tuple
import requests
import geopandas as gpd
import numpy as np
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt

#
# This file contains util functions used in the lab notebooks
# The functions are copied from the provided lab notebooks and are NOT my own work
# Changes have been done in the functions and have been marked with "CHANGE EHRENHOEFER"
#

# source: Lab01
#
#
def dataspace_dataframe_from_attributes(
    collection: str = "SENTINEL-2",
    aoi: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    attributes: Optional[Iterable[Tuple[str, str, float]]] = None,
    max_returned_items: int = 20
):
    """
    Get a dataframe of items from the Copernicus DataSpace API based on the given attributes.
    The request is build based on the OData standard as documented at
    https://documentation.dataspace.copernicus.eu/APIs/OData.html

    Parameters
    ----------
    collection : str
        The collection to search for. Default is "SENTINEL-2".
    aoi : str, optional
        The area of interest in WKT format. Default is None.
    start_date : str, optional
        The start date in the format "YYYY-MM-DD". Default is None.
    end_date : str, optional
        The end date in the format "YYYY-MM-DD". Default is None.
    attributes : Iterable[Tuple[str, str, float]], optional
        The attributes to filter by. Default is None which means no filtering and is equivalent to an empty list.
        Each tuple should be in the format (key, comparison, value).
        The comparison should be one of "lt", "le", "eq", "ge", "gt".
        Currently only attributes of type double and that are comparable are supported.
    max_returned_items : int, optional
        The maximum number of items to return. Default is 20. Must be in [0, 1000].
    """
    if attributes is None:
        attributes = []
    request_str = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter="
    request_str += f"Collection/Name eq '{collection}'"
    if aoi is not None:
        request_str += f" and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}')"
    if start_date is not None:
        request_str += f" and ContentDate/Start gt {start_date}T00:00:00.000Z"
    if end_date is not None:
        request_str += f" and ContentDate/Start lt {end_date}T00:00:00.000Z"
    for k, comp, v in attributes:
        assert comp in ["lt", "le", "eq", "ge", "gt"]
        # CHANGE EHRENHOEFER - start
        # also made this work for string params
        if isinstance(v, str):
            request_str += f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq '{k}' and att/OData.CSC.StringAttribute/Value {comp} '{v}')"
        else:
            request_str += f" and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq '{k}' and att/OData.CSC.DoubleAttribute/Value {comp} {v:.2f})"
        # CHANGE EHRENHOEFER - end
    # get all attributes
    request_str += "&$expand=Attributes"
    # get top n items
    assert 0 <= max_returned_items <= 1000, f"Copernicus API only allows returned items in [0, 1000], but {max_returned_items} is outside this range."
    request_str += f"&$top={max_returned_items}"
    json_result = requests.get(request_str).json()
    json_vals = json_result['value']
    return pd.DataFrame.from_dict(json_result['value'])


# source: Lab01
def copernicus_df_to_gdf(copernicus_df: pd.DataFrame, strftime: Optional[str] = None) -> gpd.GeoDataFrame:
    if strftime is not None and strftime.lower() in ['iso', 'iso 8601', 'default']:
        strftime = '%Y-%m-%dT%H:%M:%S'
    df = copernicus_df.copy()
    # split footprint into geometry (wkt text) and crs
    # assign crs
    crss = df['Footprint'].apply(lambda x: "EPSG:"+x.split(';')[0].split("'")[1].split('=')[1])
    assert len(crss.unique()) == 1, "Multiple CRS values in data frame, geopandas can't handle that."
    # assign geometry
    df['geometry'] = gpd.GeoSeries.from_wkt(df['Footprint'].apply(lambda x: x.split(';')[1].split("'")[0]), crs=crss.iloc[0])
    gdf = gpd.GeoDataFrame(df)
    attribute_names = set(sl['Name'] for l in gdf['Attributes'] for sl in l)
    ignored_date_conversions = []
    successful_date_conversions = []
    for attr in attribute_names:
        gdf[attr] = gdf['Attributes'].apply(lambda x: [i for i in x if i['Name'] == attr][0]['Value'] if len([i for i in x if i['Name'] == attr]) > 0 else np.nan)
    for attr in gdf.columns:
        if 'Date' in attr:
            # convert to DateTime
            try:
                gdf[attr] = pd.to_datetime(gdf[attr], yearfirst=True, format='mixed')
                # convert to numeric or string to make it json-serializable
                if strftime is None:
                    gdf[attr] = gdf[attr].values.astype(np.int64) // 10 ** 9
                else:
                    gdf[attr] = gdf[attr].dt.strftime(strftime)
            except Exception as e:
                ignored_date_conversions += [attr]
                continue
            successful_date_conversions += [attr]
    print(f"Ignored date conversions: {ignored_date_conversions}")
    print(f"Successful date conversions: {successful_date_conversions}")
    return gdf


# source: Lab01
class S2TileReader:
    def __init__(self, directory: Path):
        """
        Initialize the reader with a directory containing the SAFE file of a Sentinel-2 product.

        Parameters
        ----------
        directory : Path
            The directory containing the SAFE file of a Sentinel-2 product.
        """
        assert directory.is_dir(), f"{directory} is not a directory"
        self.image_files = list(directory.glob(f"**/IMG_DATA/*.jp2"))
        if len(self.image_files) == 0:
            self.image_files = list(directory.glob(f"**/IMG_DATA/R60m/*.jp2"))
            self.image_files.extend(list(directory.glob(f"**/IMG_DATA/R20m/*.jp2")))
            self.image_files.extend(list(directory.glob(f"**/IMG_DATA/R10m/*.jp2")))
        self.band2file_mapping = self._bands()
        self.bands = sorted(self.band2file_mapping.keys())
        print(f"{len(self.band2file_mapping)} images found in {directory}")

    def _bands(self):
        """
        Extract the band names from the image files and create a mapping from band name to file path.

        Example:
        {
            "B01": Path("path/to/B01.jp2"),
            "B02": Path("path/to/B02.jp2"),
            ...
        }
        or if the product has multiple resolutions:
        {
            "B01_60m": Path("path/to/R60m/B01.jp2"),
            "B02_10m": Path("path/to/R10m/B02.jp2"),
            ...
        }
        """
        return {"_".join(x.stem.split("_")[2:]):x for x in self.image_files}

    def read_band(self, band: str):
        """
        Read the data of a specific band. The data is returned as a numpy array. If the band is a single channel,
        the array will have shape (height, width). If the band is a multi-channel band, the array will have shape
        (height, width, channels).

        Parameters
        ----------
        band : str
            The name of the band to read. Must be one of the bands in the product.
            Use the `bands` attribute to see the available bands.
        """
        assert band in self.bands, f"Band {band} invalid. Please select one of {self.bands}"
        img_path = self.band2file_mapping[band]
        with rasterio.open(self.band2file_mapping[band]) as f:
            data = f.read()
        if data.shape[0] == 1:
            return data.squeeze(0)
        elif data.shape[0] == 3:
            return np.transpose(data, (1, 2, 0))


# source: Lab01
def quant_norm_data(
    data: np.ndarray, lower_quant: float = 0.01, upper_quant: float = 0.99
) -> np.ndarray:
    """
    Normalize the data by quantiles `lower_quant/upper_quant`.
    The quantiles are calculated globally/*across all channels*.
    """
    masked_data = np.ma.masked_equal(data, 0)
    lq, uq = np.quantile(masked_data.compressed(), (lower_quant, upper_quant))
    data = np.clip(masked_data, a_min=lq, a_max=uq)
    data = (data - lq) / (uq - lq)
    return data


# source: Lab01
def vis(data: np.ndarray, quant_norm: bool = False):
    """
    Visualize an array by calling `imshow` with `cmap="gray"` for 1 channel inputs and no cmap for 3 channel inputs.
    Expected shape is either (H, W) or (H, W, 3) for 1 and 3 channel inputs respectively. Assumes RGB order for 3
    channel inputs.

    Parameters
    ----------
    data : np.ndarray
        The data to visualize. Expected shape is either (H, W) or (H, W, 3) for 1 and 3 channel inputs respectively.
        Assumes RGB order for 3 channel inputs.
    quant_norm : bool
        Whether to quantile normalize the data. Default is False.
    """
    if quant_norm:
        data = quant_norm_data(data)
    if data.ndim == 2:
        plt.imshow(data, cmap="gray")
    elif data.ndim == 3:
        plt.imshow(data)
    else:
        raise ValueError(f"Expected data to have 2 or 3 dimensions, but got {data.ndim} dimensions.")
    plt.axis("off")
    plt.show()


# my own method
def normalize_rgb(rgb):
    rgb_norm = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3): # loop over R G B Channels 
        band = rgb[:, :, i]
        rgb_norm[:, :, i] = (band - band.min()) / (band.max() - band.min() + 1e-6)
    return rgb_norm