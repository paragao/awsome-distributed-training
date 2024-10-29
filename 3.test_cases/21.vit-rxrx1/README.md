# Train Vision Transformers on RxRx1 Dataset

## Download RxRx1 dataset

Download the [rxrx1 dataset](https://www.rxrx.ai/rxrx1#Download). It has 125,510 8-bit PNG 512x512x6 images. 

```
# 45GB - takes <1hr to download
wget https://storage.googleapis.com/rxrx/rxrx1/rxrx1-images.zip

unzip rxrx1-images.zip
```

## Convert images to zarr format

Reading individual image files can become an IO bottleneck during training. This script packs each site image into a single zarr. So, instead of having to load 6 separate channel pngs for a singe image all of those channels will be saved together in a single zarr file. More about zarr files [here](https://zarr.readthedocs.io/en/stable/)

```
git clone https://github.com/recursionpharma/rxrx1-utils
cd rxrx1-utils

scikit-image
dask
zarr
pandas
tensorflow

 python3 -m rxrx.preprocess.images2zarr --raw-images /fsxl/awsankur/rxrx1/rxrx1/images/ --dest-path /fsxl/awsankur/rxrx1/zarr --metadata /fsxl/awsankur/rxrx1/rxrx1
```
