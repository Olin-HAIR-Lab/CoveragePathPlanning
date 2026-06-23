import geopandas as gpd
import pandas as pd
from pathlib import Path

dirs = [
    "../scripts/region_previews_farm12/",
    "../scripts/region_previews_farm03/"
]

for path_dir in dirs:
    for path in Path(path_dir).glob("*.gpkg"):
        print(path.name)
        gdf = gpd.read_file(path,layer='points')

        if "IsoTime" not in gdf.columns:
            print(f"Skipping {path.name}: no isoTime column")
            continue

        if "date" in gdf.columns:
            print(f"Skipping {path.name}; already has date")
            continue

        gdf["date"] = pd.to_datetime(gdf["IsoTime"], errors="coerce").dt.date

        out_path = path.with_name(path.name)
        gdf.to_file(out_path, driver="GPKG", layer='points')
        poly_layer = gpd.read_file(path, layer='polygon')
        poly_layer.to_file(out_path, driver="GPKG", layer='polygon')

        print(f"Saved {out_path}")