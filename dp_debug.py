import pandas as pd
import matplotlib.pyplot as plt

filtered_vehicle_df = pd.read_csv('./datasets/a2-datasets/filtered_vehicle.csv')
accident_df = pd.read_csv('./datasets/a2-datasets/accident.csv')
road_cond_df = pd.read_csv('./datasets/a2-datasets/road_surface_cond.csv')
atmosph_df = pd.read_csv('./datasets/a2-datasets/atmospheric_cond.csv')

filtered_vehicle_cols = [ "ACCIDENT_NO"
                        , "VEHICLE_COLOUR_1"
                        ]
accident_cols = [ "ACCIDENT_NO"
                , "LIGHT_CONDITION"
                , "SEVERITY"
                ]
road_cond_cols = [ "ACCIDENT_NO"
                , "SURFACE_COND"
                , "SURFACE_COND_DESC"
                , "SURFACE_COND_SEQ"
                ]
atmosph_cols = [ "ACCIDENT_NO"
            , "ATMOSPH_COND"
            , "ATMOSPH_COND_SEQ"
            , "ATMOSPH_COND_DESC"
            ]
filtered_vehicle_df = filtered_vehicle_df[filtered_vehicle_cols]
accident_df = accident_df[accident_cols]
road_cond_df = road_cond_df[road_cond_cols]
atmosph_df = atmosph_df[atmosph_cols]

##distinct_light_condition = accident_df.drop_duplicates(subset=['LIGHT_CONDITION']).sort_values(by = "ACCIDENT_TIME")
df = pd.merge(filtered_vehicle_df, accident_df, how='left', on='ACCIDENT_NO')
grouped_df = df.groupby(["LIGHT_CONDITION","VEHICLE_COLOUR_1"]).size().reset_index(name='ACCIDENTS')
print(grouped_df)