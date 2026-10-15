import geojson
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon
from shapely.ops import transform as shapely_transform

from logger import logger
import rasterio
from rasterio.coords import BoundingBox


def load_geojson(file_path):
    geoms = []
    # Load geometries from the GeoJSONL file
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            try:
                geom = geojson.loads(line)
                if (not geom['geometry']['coordinates']):
                    continue
                else:
                    polygon = shape(geom['geometry'])
                    if isinstance(polygon, Polygon):
                        geoms.append(polygon)  # Convert to Shapely geometry for easy filtering
            except Exception as e:
                print(f"Error on line {i}: {e}")
                continue
    return geoms

def pixel_to_geo(pixels, transform):
    """
    Tranform geometric coordinates to pixel space. Note that the x and y axis
    in pixel space need to be exchanged.
    
    Args:
    - pixels: List of 2D points in pixel space
    - transform: transform of the geotiff file, e.g. dataset.transform

    Returns:
    - geoms: List of 2D points in geometrical space. 

    """
    pixels = [(pixel[1], pixel[0]) for pixel in pixels]
    geoms = [transform * pixel for pixel in pixels]

    return geoms

def geo_to_pixel(geoms, transform, offset=(0, 0)):
    """
    Tranform geometric coordinates to pixel space. Note that the x and y axis
    in pixel space need to be exchanged.
    
    Args:
    - geoms: List of 2D points(Tuple) in geometrical space. 
    - transform: transform of the geotiff file, e.g. dataset.transform

    Returns:
    - pixels: List of 2D points(Tuple) in pixel space

    """
    pixels = [~transform * geom for geom in geoms]
    pixels = [(pixel[1] - offset[0], pixel[0] - offset[1]) for pixel in pixels]

    return pixels

def pixel_bounds_to_geo_bounds(top_left, bottom_right, transform):
    """
    Transform pixel to geometric bounds. 

    Args:
    - top_left, bottom_right: tuple (row, column)
    - transform: transform of the geotiff file, e.g. dataset.transform

    Returns:
    - bounds: rasterio.coords.BoundingBox
    """
    geoms = pixel_to_geo([top_left, bottom_right], transform)

    return BoundingBox(
        left=geoms[0][0],
        top=geoms[0][1],
        right=geoms[1][0],
        bottom=geoms[1][1]
    )


if __name__ == "__main__":
    mask_path = "/cluster/scratch/jingyuwang/data/mask.tif"
    with rasterio.open(mask_path) as dataset:
        logger.debug_print(dataset.meta)
        bounds = dataset.bounds
        logger.debug_print(dataset.bounds)
        height, width = dataset.height, dataset.width
        transform = dataset.transform
    logger.debug_print(pixel_bounds_to_geo_bounds((0, 0), (height, width), transform))
    logger.debug_print(geo_to_pixel([(bounds.left, bounds.top), (bounds.right, bounds.bottom)], transform))
