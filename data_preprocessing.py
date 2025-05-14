#neural networks
#random forests
import pandas as pd
import matplotlib.pyplot as plt

filtered_vehicle_df = pd.read_csv('./datasets/a2-datasets/filtered_vehicle.csv')
accident_df = pd.read_csv('./datasets/a2-datasets/accident.csv')
road_cond_df = pd.read_csv('./datasets/a2-datasets/road_surface_cond.csv')
atmosph_df = pd.read_csv('./datasets/a2-datasets/atmospheric_cond.csv')

filtered_vehicle_cols = [ "ACCIDENT_NO"
                         , "VEHICLE_BODY_STYLE"
                         , "VEHICLE_MAKE"
                         , "VEHICLE_MODEL"
                         , "VEHICLE_POWER"
                         , "VEHICLE_TYPE"
                         , "VEHICLE_WEIGHT"
                         , "LAMPS"
                         , "LEVEL_OF_DAMAGE"
                         , "VEHICLE_COLOUR_1"
                        ]
accident_cols = [ "ACCIDENT_NO"
                , "ACCIDENT_DATE"
                , "ACCIDENT_TIME"
                , "LIGHT_CONDITION"
                , "NO_PERSONS_KILLED"
                , "NO_PERSONS"
                , "SEVERITY"
                , "SPEED_ZONE"
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
##nan_value = filtered_vehicle_df.loc[:, filtered_vehicle_df.isna().any()]
filtered_vehicle_df.size

print(filtered_vehicle_df.head())
print(accident_df.head())
print(road_cond_df.head())
print(atmosph_df.head())
##accident_df.head()
##person_df.head()
##vehicle_df.head()
