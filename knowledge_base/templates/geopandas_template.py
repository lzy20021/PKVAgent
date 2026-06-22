import os
import geopandas as gpd

# Read input
gdf = gpd.read_file('{input_file}')
print(f'Read {len(gdf)} features from {input_file}')

# Optional CRS normalization for metric operations
if gdf.crs is not None and gdf.crs.is_geographic:
    gdf = gdf.to_crs('EPSG:3857')

# TODO: algorithm operation
# gdf = ...

os.makedirs(os.path.dirname('{output_file}') or '.', exist_ok=True)
gdf.to_file('{output_file}')
show_layer('{output_file}', '{layer_name}')
print('Done')
