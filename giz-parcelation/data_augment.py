# Assume that we have the geojson, satelite image and geojson files in the same CRS (EPSG:4396)

import rasterio
import rasterio.features
from shapely.geometry import box

import random
import numpy as np
import math
from tqdm import tqdm

import click

from PIL import Image, ImageDraw

from utils import load_geojson, pixel_bounds_to_geo_bounds, geo_to_pixel
from logger import logger


SIZE_RAW = [1536, 1792, 2048, 2560]
SIZE_TAR = 640

SQUARE_2 = 1.414306640625


@click.command()
@click.option("--image_path", "-i", type=str, help="Path to the image geotiff file")
@click.option("--geojson", "-j", type=str, help="Path to the geojson file")
@click.option("--outpath", "-o", type=str, default="./", help="Path to the destination")
@click.option("--n_samples", "-n", type=int, default=100, help="Number of samples to generate")
@click.option("--debug", "-d", type=bool, default=False, help="Enable debug")
def main(image_path, geojson, outpath, n_samples, debug):
    # Set logger
    logger.set_mode(debug=debug)
    logger.set_logdir(outpath)

    # Load geojson file
    geoms = load_geojson(geojson)

    # Open image geotiff file
    logger.info_print("Loading image...")
    with rasterio.open(image_path) as dataset:
        width = dataset.width
        height = dataset.height
        transform = dataset.transform
        img = dataset.read()[:3, ...].transpose((1,2,0)) # [H, W, C]
        logger.debug_print(f"Loaded {dataset.meta}")
        logger.debug_print(img.shape)

    logger.info_print("Starting generating...")
    # Main generating module
    for k in tqdm(range(n_samples), desc="Generating dataset"):
        # Random crop(discard which does not have parcels inside)
        cent_size = random.choice(SIZE_RAW)
        crop_size = int(cent_size * SQUARE_2)

        found_parcel = False
        while not found_parcel:
            # There must be a parcel inside the inner square (crop_size/2)
            i = random.randint(0, img.shape[0] - crop_size)
            j = random.randint(0, img.shape[1] - crop_size)
        
            x1, y1, x2, y2 = i + crop_size/4, j + crop_size / 4, i + 3*crop_size/4, j + 3*crop_size/4
            inner_bounds = pixel_bounds_to_geo_bounds((x1, y1), (x2, y2), transform)
            inner_bounds = box(*inner_bounds)
            inner_parcels = [geom for geom in geoms if geom.within(inner_bounds)]

            x1, y1, x2, y2 = i, j, i + crop_size, j + crop_size
            outer_bounds = pixel_bounds_to_geo_bounds((x1, y1), (x2, y2), transform)
            outer_bounds = box(*outer_bounds)
            outer_parcels = [geom for geom in geoms if geom.within(outer_bounds)]

            logger.debug_print(outer_parcels)
            found_parcel = len(inner_parcels) > 0

        img_crop = Image.fromarray(img[x1:x2, y1:y2])
        polygons = [geo_to_pixel(list(parcel.exterior.coords), transform, offset=(x1, y1)) for parcel in outer_parcels]

        # YOLO/PIL use x as horizontal and y as vertical (W, H)
        polygons = [[(point[1], point[0]) for point in polygon] for polygon in polygons]
        logger.debug_print(polygons)
        if debug:
            # PIL DRAW uses 
            draw = ImageDraw.Draw(img_crop)
            for polygon in polygons:
                draw.polygon(polygon, width=10, outline="blue")
            logger.write_pil_image(f"debug/{k:03d}.png", img_crop)


        # Add rotation
        rot_angle = random.choice(range(0, 360, 20))
        logger.debug_print(rot_angle)
        # To match Image.rotate convention
        angle_rad = - rot_angle * (math.pi / 180.0)
        rotation_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
                                    [math.sin(angle_rad), math.cos(angle_rad)]])

        img_rot = img_crop.rotate(rot_angle)
        polygons_rot = [
            [rotation_matrix @ np.array((point[0] - crop_size/2, point[1] - crop_size/2)) for point in polygon] 
            for polygon in polygons
        ]
        polygons_rot = [
            [tuple([point[0] + crop_size/2, point[1] + crop_size/2]) for point in polygon]
            for polygon in polygons_rot
        ]

        if debug:
            # PIL DRAW uses x as horizontal and y as vertical
            draw = ImageDraw.Draw(img_rot)
            for polygon in polygons_rot:
                draw.polygon(polygon, width=5, outline="red")
            logger.write_pil_image(f"debug/{k:03d}_rot.png", img_rot)

        # Take the center
        left   = (crop_size - cent_size) / 2
        right  = (crop_size + cent_size) / 2
        top    = (crop_size - cent_size) / 2
        bottom = (crop_size + cent_size) / 2

        img_out = img_rot.crop((left, top, right, bottom))
        img_out = img_out.resize((SIZE_TAR, SIZE_TAR))

        # Shift and filter polygons
        polygons_out = [
            [((point[0] - left)/cent_size, (point[1] - top)/cent_size) for point in polygon]
            for polygon in polygons_rot
        ]
        polygons_out = [
            polygon for polygon in polygons_out if (np.array(polygon) > 0.).all() and (np.array(polygon) < 1.).all()
        ]
        if debug:
            logger.write_pil_image(f"debug/{k:03d}_out.png", img_out)
        
        # Save image 
        logger.write_pil_image(f"images/{k:04d}.png", img_out)

        # Save mask 
        mask = Image.new("L", (SIZE_TAR, SIZE_TAR), 0)
        draw = ImageDraw.Draw(mask)
        for i, polygon in enumerate(polygons_out):
            points = [(point[0] * SIZE_TAR, point[1] * SIZE_TAR) for point in polygon]
            draw.polygon(points, fill=i, outline=None)
        logger.write_pil_image(f"masks/{k:04d}.png", mask)

        # Save label
        logger.write_yolo_label(f"labels/{k:04d}.txt", [0]*len(polygons_out), polygons_out)

        

if __name__ == "__main__":
    main()
