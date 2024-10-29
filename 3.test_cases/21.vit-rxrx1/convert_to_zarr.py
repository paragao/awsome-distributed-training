import argparse
import os

import dask
import dask.bag
import toolz as t
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd

import zarr
import tensorflow as tf
from skimage.io import imread

DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)
DEFAULT_COMPRESSION = {"cname": "zstd", "clevel": 3, "shuffle": 2}

DEFAULT_BASE_PATH = '/fsxl/awsankur/rxrx1/rxrx1/'
DEFAULT_METADATA_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'metadata')
DEFAULT_IMAGES_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'images')

def zarrify(x, dest, chunk=512, compression=DEFAULT_COMPRESSION):
    compressor = None
    if compression:
        compressor = zarr.Blosc(**compression)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    z = zarr.open(
        dest,
        mode="w",
        shape=x.shape,
        chunks=(chunk, chunk, None),
        dtype="<u2",
        compressor=compressor)
    z[:] = x
    return z

ZARR_DEST = "{dataset}/{experiment}/Plate{plate}/{well}_s{site}.zarr"



def load_image(image_path):
    with tf.io.gfile.GFile(image_path, 'rb') as f:
        return imread(f, format='png')


def load_images_as_tensor(image_paths, dtype=np.uint8):
    n_channels = len(image_paths)

    data = np.ndarray(shape=(512, 512, n_channels), dtype=dtype)

    for ix, img_path in enumerate(image_paths):
        data[:, :, ix] = load_image(img_path)

    return data

def image_path(dataset,
               experiment,
               plate,
               address,
               site,
               channel,
               base_path=DEFAULT_IMAGES_BASE_PATH):
    """
    Returns the path of a channel image.

    Parameters
    ----------
    dataset : str
        what subset of the data: train, test
    experiment : str
        experiment name
    plate : int
        plate number
    address : str
        plate address
    site : int
        site number
    channel : int
        channel number
    base_path : str
        the base path of the raw images

    Returns
    -------
    str the path of image
    """
    return os.path.join(base_path, dataset, experiment, "Plate{}".format(plate),
                        "{}_s{}_w{}.png".format(address, site, channel))




def load_site(dataset,
              experiment,
              plate,
              well,
              site,
              channels=DEFAULT_CHANNELS,
              base_path=DEFAULT_IMAGES_BASE_PATH):
    """
    Returns the image data of a site

    Parameters
    ----------
    dataset : str
        what subset of the data: train, test
    experiment : str
        experiment name
    plate : int
        plate number
    address : str
        plate address
    site : int
        site number
    channels : list of int
        channels to include
    base_path : str
        the base path of the raw images

    Returns
    -------
    np.ndarray the image data of the site
    """
    channel_paths = [
        image_path(
            dataset, experiment, plate, well, site, c, base_path=base_path)
        for c in channels
    ]
    return load_images_as_tensor(channel_paths)



@t.curry
def convert_to_zarr(src_base_path, dest_base_path, site_info):
    dest = os.path.join(dest_base_path, ZARR_DEST.format(**site_info))
    site_data = load_site(base_path=src_base_path, **site_info)
    zarrify(site_data, dest)

def convert_all(raw_images, dest_path, metadata):
    #metadata_df = rio.combine_metadata(metadata, include_controls=False)

    metadata_df = pd.read_csv(metadata+'metadata.csv')
    #import pdb;pdb.set_trace()
    sites = metadata_df[['dataset', 'experiment', 'plate', 'well', 'site']].to_dict(orient='records')
    bag = dask.bag.from_sequence(sites)
    bag.map(convert_to_zarr(raw_images, dest_path)).compute()


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Converts the raw PNGs into zarr files")
    parser.add_argument("--raw-images", type=str, help="Path of the raw images", required=True)
    parser.add_argument("--dest-path", type=str, help="Path of the zarr files to write", required=True)
    parser.add_argument("--metadata", type=str, help="Path where the metadata files live", required=True)

    args = parser.parse_args()

    from dask.diagnostics import ProgressBar
    ProgressBar().register()

    convert_all(**vars(args))



if __name__ == '__main__':
    cli()
