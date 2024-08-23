import hashlib
import pathlib
import os

# import imutils
import cv2
import numpy as np
from urllib.request import urlopen


# CACHE_FOLDER = "/home/martijn/Documents/work/2023-08_synthesis/vu_high_res/iiif_cache/work/"
CACHE_FOLDER = "/scratch/iiif_cache/work/"


def fetch_image(url: str, use_cache: bool = True) -> np.ndarray:
    """
    Fetch image from url, but before requesting from the server
    see if a cached version of it exists in the `CACHE_FOLDER`
    """
    # print("url inside fetch_image", url)
    hashed = hashlib.md5(url.encode("utf-8")).hexdigest() + ".jpg"
    file_name = os.path.join(CACHE_FOLDER, hashed)
    if use_cache and pathlib.Path(file_name).is_file():
        # it is there, fetch from cache folder
        with open(file_name, "rb") as fh:
            print(f"reading cached file {file_name}")
            raw = bytearray(fh.read())
    else:
        # not there / or no desire to cache
        # retrieve from remote + store in cache (iff use_cache=True)
        # can throw e.g. urllib.error.HTTPError :: 503
        # we could try a few more times (each time with more time between)
        resp = urlopen(url)

        raw = bytearray(resp.read())
        if use_cache:
            print(f"storing file {file_name} in cache")
            with open(file_name, "wb") as fh:
                fh.write(raw)
    flag = cv2.IMREAD_COLOR
    image = cv2.imdecode(np.asarray(raw, dtype="uint8"), flag)
    return image
