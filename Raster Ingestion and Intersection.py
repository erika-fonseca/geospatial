# Databricks notebook source
# MAGIC %md # Raster Ingestion and Intersection
# MAGIC 
# MAGIC This notebook converts Raster data to H3, allowing you to spatially aggregate, spatially join, and visualize data in an efficient manner.
# MAGIC 
# MAGIC <img src="https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/GUID-6754AF39-CDE9-4F9D-8C3A-D59D93059BDD-web.png" width=250px> 
# MAGIC →
# MAGIC <img src="https://www.databricks.com/wp-content/uploads/2019/11/Processing-Geospatial-Data-at-Scale-With-Databricks-02.jpg" width=250px>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup
# MAGIC 
# MAGIC For demo purposes we are installing **rasterio** and **databricks-mosaic** in this notebook, but for production workloads keep in mind that accoding to the documentation: 
# MAGIC 
# MAGIC *"Using notebook-scoped libraries might result in more traffic to the driver node as it works to keep the environment consistent across executor nodes."* 
# MAGIC <br>Source: <a href="https://docs.databricks.com/libraries/notebooks-python-libraries.html">Notebook-scoped Python libraries</a>
# MAGIC 
# MAGIC Guide on how to install libraries on a cluster <a href="https://docs.databricks.com/libraries/cluster-libraries.html">here</a>.

# COMMAND ----------

# MAGIC %pip install rasterio databricks-mosaic

# COMMAND ----------

import json
from pyspark.sql.functions import col, lit, explode
from pyspark.sql.types import ArrayType, StringType, DoubleType, StructType, StructField, LongType, IntegerType
import pyspark.sql.functions as F
from matplotlib import pyplot
import pandas as pd
import numpy as np

import rasterio
import rasterio.features
import rasterio.warp
from rasterio.plot import show
from rasterio.io import MemoryFile

from rasterio.io import MemoryFile
import rasterio.mask

import mosaic as mos
from mosaic import enable_mosaic

h3_resolution = 4

# COMMAND ----------

enable_mosaic(spark, dbutils)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Loading Raster Data (1x)
# MAGIC 
# MAGIC #### TO DO:
# MAGIC * Upload your file to a location accessible on your Databricks workspace
# MAGIC * Update the DATA_DIR variable on the cell bellow with the location of your data

# COMMAND ----------

# Path to directory of geotiff images 
DATA_DIR = "/FileStore/geospatial"
DATA_DIR_FUSE = "/dbfs" + DATA_DIR
FILE = "mesh_20211014_030000_20211014_090000_3577-1.tif"

# COMMAND ----------

# This is how your data looks like being loaded with rasterio
dataset = rasterio.open(DATA_DIR_FUSE+"/"+FILE)
show(dataset)

# COMMAND ----------

# Rasterio reads raster data into numpy arrays so plotting a single band as two dimensional data can be accomplished directly with pyplot.
fig, ax = pyplot.subplots(1, figsize=(12, 12))
show((dataset, 1), cmap='Greys_r', interpolation='none', ax=ax)
pyplot.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data ingestion as binary file

# COMMAND ----------

df_bin = (spark.read
          .format("binaryFile")
          .option("pathGlobFilter", "*.tif")
          .load(DATA_DIR)
          .cache() # Caching while developing, TODO: Remove in prod
         )

display(df_bin)

# COMMAND ----------

# MAGIC %md
# MAGIC Extracting the CRS (coordinate reference system) associated with the dataset identifying where the raster is located in geographic space. 

# COMMAND ----------

@udf(returnType=StringType()) 
def get_crs(content):
  # Read the in-memory tiff file
  with MemoryFile(bytes(content)) as memfile:
    with memfile.open() as data:
      # Use rasterio with the data object
      return str(data.crs)

df_bin.withColumn("crs", get_crs("content")).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract masks from images
# MAGIC An image mask identifies the regions of the image where there is valid data to be processed.

# COMMAND ----------

@udf(returnType=ArrayType(StringType())) 
def get_mask_shapes(content):
  geometries = []
  
  # Read the in-memory tiff file
  with MemoryFile(bytes(content)) as memfile:
    with memfile.open() as data:

      # Read the dataset's valid data mask as a ndarray.
      mask = data.dataset_mask()

      # Extract feature shapes and values from the array.
      for geom, val in rasterio.features.shapes(
              mask, transform=data.transform):

        if val > 0: # Only append shapes that have a positive maks value
          
          # Transform shapes from the dataset's own coordinate
          # reference system to CRS84 (EPSG:4326).
          geom = rasterio.warp.transform_geom(
              data.crs, 'EPSG:4326', geom, precision=6)

          geometries.append(json.dumps(geom))
          
  return geometries

# COMMAND ----------

df_masks = (df_bin
            .withColumn("mask_json_shapes", get_mask_shapes("content"))
            .withColumn("mask_json", explode("mask_json_shapes"))
            # Convert geoJSON to WKB
            .withColumn("mask_wkb", mos.st_aswkb(mos.st_geomfromgeojson("mask_json")))
            .drop("content", "mask_json_shapes", "mask_json")
            .cache() # Caching while developing, TODO: Remove in prod
           )
df_masks.display()

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC df_masks "mask_wkb" "geometry"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract chips

# COMMAND ----------

df_chips = (df_masks
            # Tessellate with Mosaic
            .withColumn("chips", mos.grid_tessellateexplode("mask_wkb", lit(h3_resolution)))
            .select("path", "modificationTime", "chips.*")
            .withColumn("chip_geojson", mos.st_asgeojson("wkb"))
            .cache() # Caching while developing, TODO: Remove in prod
           )
df_chips.display()

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC df_chips "wkb" "geometry" 10000

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pixels to chips

# COMMAND ----------

schema = ArrayType(
  StructType([
    StructField("values", ArrayType(DoubleType())),
    StructField("nonzero_pixel_count", IntegerType()),
  ]))

@udf(returnType=schema)
def get_shapes_avg(content, chips):
  chip_values = []
  
  # Read the in-memory tiff file
  with MemoryFile(bytes(content)) as memfile:
    with memfile.open() as data:
      
      for chip in chips:
        chip_geojson = json.loads(chip)                                        # Chip in GeoJSON format
        geom = rasterio.warp.transform_geom('EPSG:4326', data.crs, chip_geojson, precision=6)  # Project chips to the image CRS
        out_image, out_transform = rasterio.mask.mask(data, [geom], crop=True, filled=False)   # Crop the image on a shape containing the chip
        
        val = np.average(out_image, axis=(1,2)).tolist() # Aggregated by band
        nonzeroes = np.count_nonzero(out_image.mask)     # Cound pixels within the chip shape
        
        chip_values.append({
          "values": val,                     # Aggregated pixel values by band
          "nonzero_pixel_count": nonzeroes   # Number of pixels within the mask shape
        })
        
  return chip_values

df_chipped = (df_chips
              .groupBy("path", "modificationTime")
              .agg(F.collect_list(F.struct("chip_geojson", "index_id", "wkb", "is_core")).alias("chips"))  # Collecting the list of chips
              .join(df_bin, ["path", "modificationTime"])                                                  # Join with the original data files
              .withColumn("chip_values", get_shapes_avg(col("content"), F.expr("chips.chip_geojson")))     # Execute UDF to aggregate pixels for each chip
              .withColumn("zipped_chip_values", F.arrays_zip("chips", "chip_values"))
              .withColumn("zipped_chip_value", F.explode("zipped_chip_values"))                            # Explode result array in multiple rows
              .select(                                                                                     # Select only relevant columns
                col("path"), 
                col("modificationTime"), 
                F.expr("zipped_chip_value.chips.*"),
                F.expr("zipped_chip_value.chip_values.*"),
                F.expr("zipped_chip_value.chip_values.values[0]").alias("value_band_0")
              )
              .cache()  # TODO: Remove in production
             )
df_chipped.display()

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC df_chipped "wkb" "geometry" 10000

# COMMAND ----------

# MAGIC %md
# MAGIC In the TIFF file the shape is the rectangle that contains image (we can get this from <a href="https://rasterio.readthedocs.io/en/latest/quickstart.html#dataset-georeferencing">bounds</a>). We need to cut the image with the same technique, but on top of that we need find all the pixels that fall within each cell and run an aggregation for those pixels (min/max/average/median/etc.).
# MAGIC In order to do that we need to use functions like <a href="https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html">masking</a> or <a hrefe="https://rasterio.readthedocs.io/en/latest/topics/features.html#burning-shapes-into-a-raster">rasterize</a> to get the portion of the image that corresponds to each grid cell.
# MAGIC This will generate an aggregated information based on the grid cells. We can store that in a table and visualise it, join it with tesselated vectors etc.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading tabular data

# COMMAND ----------

# Path to directory of the csv for impact assessment
EXP_FILE = "exposure_hail.csv"
exposed_hail = spark.read.format("csv").option("header","true").load(DATA_DIR+"/"+EXP_FILE)

# COMMAND ----------

display(exposed_hail)

# COMMAND ----------

exposed_hail.count()

# COMMAND ----------

exposed_hail_tf = (
  exposed_hail
    .drop("ID")
    .withColumn("latitude",col("latitude").cast(DoubleType()))
    .withColumn("longitude",col("longitude").cast(DoubleType()))                                              
    .withColumn("geom", mos.st_astext(mos.st_point(col("longitude"), col("latitude")))) # First we need to creating a new Mosaic Point geometry, and afterwards translate a geometry into its Well-known Text (WKT) representation
)

# COMMAND ----------

# MAGIC %md
# MAGIC We can use Mosaic functionality to identify how to best index our data based on the data inside the specific dataframe. </br>
# MAGIC Selecting an appropriate indexing resolution can have a considerable impact on the performance. </br>

# COMMAND ----------

hailsWithIndex = (exposed_hail_tf
  .withColumn("index_id", mos.grid_pointascellid(col("geom"), lit(h3_resolution)))
)

# COMMAND ----------

display(hailsWithIndex)

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC hailsWithIndex "geom" "geometry" 100000

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Join with the other dataset using index_id

# COMMAND ----------

withHailZone = (
  hailsWithIndex.join(
    df_chipped,
    on="index_id",
    how="right")
    .where(
      # If the borough is a core chip (the chip is fully contained within the geometry), then we do not need
      # to perform any intersection, because any point matching the same index will certainly be contained in
      # the borough. Otherwise we need to perform an st_contains operation on the chip geometry.
      col("is_core") | mos.st_contains(col("wkb"), col("geom")))
    .groupBy(["index_id", "wkb", "value_band_0", "nonzero_pixel_count", "is_core"])
    .agg(F.count("geom").alias("point_count"))
    .cache() # TODO: Remove in production
#     drop("count")
)

display(withHailZone)

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC withHailZone "wkb" "geometry" 100000

# COMMAND ----------

# MAGIC %md
# MAGIC Intersection between the points and raster data.

# COMMAND ----------

withHailZoneIntersection = (
  hailsWithIndex.join(
    df_chipped,
    on="index_id",
    how="inner")
    .where(
      # If the borough is a core chip (the chip is fully contained within the geometry), then we do not need
      # to perform any intersection, because any point matching the same index will certainly be contained in
      # the borough. Otherwise we need to perform an st_contains operation on the chip geometry.
      col("is_core") | mos.st_contains(col("wkb"), col("geom")))
    .groupBy(["index_id", "wkb", "value_band_0", "nonzero_pixel_count", "is_core"])
    .agg(F.count("geom").alias("point_count"))
    .cache() # TODO: Remove in production
#     drop("count")
)

display(withHailZoneIntersection)

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC withHailZoneIntersection "wkb" "geometry" 100000
