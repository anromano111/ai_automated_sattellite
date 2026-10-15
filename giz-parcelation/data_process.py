import rasterio
import rasterio.features
import click
import os
from utils import load_geojson
from logger import logger
from shapely.geometry import shape, box, mapping
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_geom

logger.set_mode(debug=True)

@click.command()
@click.option("--geotiff", type=str, help="Path to the geotiff file")
@click.option("--geojson", type=str, help="Path to the geojson file")
@click.option("--outpath", type=str, default="./", help="Path to the destination")
def main(geotiff, geojson, outpath):

    # Load geojson file
    geoms = load_geojson(geojson)
    logger.debug_print(len(geoms))

    # Open geotiff file
    with rasterio.open(geotiff) as dataset:
        # Transform dataset bounds to EPSG:4326
        dataset_bounds = box(*dataset.bounds)  # Create Shapely box from original bounds
        transformed_bounds = transform_geom(
            dataset.crs, 'EPSG:4326', mapping(dataset_bounds), precision=6
        )
        transformed_bounds = shape(transformed_bounds)  # Convert to Shapely geometry for `within` filtering

        # Get raster's transform and dimensions, reprojected to EPSG:4326
        out_transform, out_width, out_height = calculate_default_transform(
            dataset.crs, 'EPSG:4326', dataset.width, dataset.height, *dataset.bounds
        )
        out_shape = (out_height, out_width)

        logger.info_print("Finished loading and transforming raster data!")
        
        # Filter geometries that are fully within the transformed bounds
        within_geoms = [geom for geom in geoms if geom.within(transformed_bounds)]
        logger.debug_print(len(within_geoms))
        
        logger.info_print("Finished filtering geometries!")
        
        # Rasterize the transformed geometries onto the transformed raster's grid
        mask = rasterio.features.rasterize(
            [(shape(geom), i + 1) for i, geom in enumerate(within_geoms)],  # Burn value '1' for each geometry
            out_shape=out_shape,
            transform=out_transform,  # Use the transformed raster transform
            fill=0,  # Pixels outside the polygons will be set to 0
            all_touched=True,  # Burn all pixels touched by polygons
            dtype='uint16',  # Output data type
        )
        logger.debug_print(mask.shape)
        
        logger.info_print("Finished rasterizing geometries!")

        # Update metadata for the new rasterized output
        mask_meta = dataset.meta.copy()
        mask_meta.update({
            "count": 1,
            "crs": 'EPSG:4326',  # Set the CRS to EPSG:4326 for the new raster
            "transform": out_transform,
            "width": out_width,
            "height": out_height,
            "dtype": 'uint16',
            "compress": 'lzw' # Added compression
        })
        
        # Save the mask to a new file
        with rasterio.open(os.path.join(outpath, 'mask.tif'), 'w', **mask_meta) as out_dataset:
            out_dataset.write(mask, 1)  # Write the rasterized array to the first band

        # Update metadata for the new file
        new_meta = dataset.meta.copy()
        new_meta.update({
            "crs": 'EPSG:4326',
            "transform": out_transform,
            "width": out_width,
            "height": out_height
        })
        # Save the original geotiff file in EPSG:4326
        with rasterio.open(os.path.join(outpath, 'img.tif'), "w", **new_meta) as dst:
            for i in range(1, dataset.count + 1):  # Loop through all bands
                reproject(
                    source=rasterio.band(dataset, i),
                    destination=rasterio.band(dst, i),
                    src_transform=dataset.transform,
                    src_crs=dataset.crs,
                    dst_transform=out_transform,
                    dst_crs="EPSG:4396",
                    resampling=Resampling.nearest
                )

        print("Finished saving rasterization!")



if __name__ == "__main__":
    main()
