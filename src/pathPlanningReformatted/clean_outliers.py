import sys
import yaml
import geopandas as gpd
import numpy as np

def main(path):
    with open(path,'r') as f:
        yaml_data = yaml.safe_load(f)
    
    for datapath in yaml_data['data_paths']:
        gdf = gpd.read_file(datapath,layer='points')

        mean = gdf["Moisture"].mean()
        std = gdf["Moisture"].std()

        gdf["z_score"] = (gdf["Moisture"] - mean) / std

        try:
            dates = gdf["date"]
        except KeyError:
            continue

        dates_set = dates.unique()
        #print(dates_set)
        date_count = np.zeros(dates_set.size)
        for i,date in enumerate(dates_set):
            date_count[i] = len(gdf[gdf["date"] == date])
        date_proportion = date_count / sum(date_count)
        #print(date_proportion)
        print(f"Original points: {len(gdf)}")

        # Most of the points in one of the subregions are from the same date
        # We just throw out the rest 
        idx = np.argmax(date_proportion)
        target_date = dates_set[idx]
        gdf_filtered_date = gdf[gdf['date'] == target_date]
        print(f"Removed {len(gdf) - len(gdf_filtered_date)} points based on their date")

        gdf_filtered = gdf_filtered_date[(gdf_filtered_date["z_score"] >= -3.5) &
                        (gdf_filtered_date["z_score"] <= 3.5)]

        print(f"Removed {len(gdf_filtered_date) - len(gdf_filtered)} points ({100*(len(gdf_filtered_date) - len(gdf_filtered))/len(gdf):.2f}%) based on their Z score")
        print(f"Remaining: {len(gdf_filtered)} ({100*len(gdf_filtered)/len(gdf):.1f}%) of original date")

        print(f"Range: {max(gdf['Moisture']) - min(gdf['Moisture']):.2f} original --> {max(gdf_filtered['Moisture']) - min(gdf_filtered['Moisture']):.2f} after filtering")
        print(f"Std: {np.std(gdf['Moisture']):.2f} original --> {np.std(gdf_filtered['Moisture']):.2f} after filtering")

        # Save results
        gdf.to_file(f"{datapath[:-5]}_filtered.gpkg", driver="GPKG", layer="points")
        poly_layer = gpd.read_file(datapath,layer="polygon")
        poly_layer.to_file(f"{datapath[:-5]}_filtered.gpkg", driver="GPKG", layer="polygon")

if __name__ == "__main__":
    path = sys.argv[1]
    main(path)