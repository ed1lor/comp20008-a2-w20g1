#neural networks
#random forests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
#import seaborn as sns

final_data_df = pd.read_csv('./datasets/a2-datasets/final_data_file.csv', index_col = False)
colormap_df = pd.read_csv('./datasets/a2-datasets/color_mapping.csv', index_col = False)
# result = pd.merge(filtered_vehicle_df, accident_df, how='left', on='ACCIDENT_NO')
result = pd.merge(final_data_df, colormap_df, how = 'left', on = 'VEHICLE_COLOUR_1')

df = result[["SEVERITY","LIGHT_CONDITION","HEX", "COLOR"]]
group = ["HEX", "COLOR"]
features = ["SEVERITY","LIGHT_CONDITION"]
grouped_df = df.groupby(group)[features].mean().reset_index()
#normalize the features
normalized_features = MinMaxScaler().fit_transform(grouped_df[features])

sum_of_squared_errors = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state = 0, n_init=10)
    kmeans.fit_predict(normalized_features)
    sum_of_squared_errors.append(kmeans.inertia_)

points = np.column_stack((k_range, sum_of_squared_errors))

# Line from first to last point
line_start, line_end = points[0], points[-1]
##### formula from genAI in order to find the distance between point to line using vector projection (perpendicular distance between point and line made with start and end cluster points)
# Compute distances
def perpendicular_distance(point, start, end):
    return np.abs(np.cross(end-start, start-point)) / np.linalg.norm(end-start)

distances = [perpendicular_distance(p, line_start, line_end) for p in points]
elbow_k = k_range[np.argmax(distances)]

print(f"Optimal number of clusters (elbow point): k = {elbow_k}")
###### End of AI assistance
# Plot Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(k_range, sum_of_squared_errors, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.savefig('elbow_graph.png')
plt.show()
plt.close()

k_result = elbow_k

clusters = KMeans(n_clusters = k_result, random_state = 0, n_init=10)
clustering = clusters.fit_predict(normalized_features)

#get the crash count for the grouped columns
crash_count_df = df.groupby(group).size().reset_index(name='ACCIDENT_COUNT')

#create cluster column for the dataset
grouped_df['CLUSTERS'] = clustering
#join the crash count with the clusters to see frequency of crash on the clusters
merged_df = pd.merge(crash_count_df, grouped_df, on=group)
#merged_df = grouped_df

plt.rcParams["figure.figsize"] = (13,8)

colormap = {0: 'red', 1: 'green', 2: 'blue', 3: 'black'}
#print(merged_df)
#iterate through the 3 clusters
for cluster_id in range(k_result):
    plt.scatter(merged_df.loc[merged_df['CLUSTERS'] == cluster_id, 'COLOR'], merged_df.loc[merged_df['CLUSTERS'] == cluster_id, 'ACCIDENT_COUNT'],
                label=f"Cluster number: {cluster_id+1}", c=colormap[cluster_id])

#Basic scatter plot
plt.xlabel('Colors')
plt.ylabel('Crashes under similar Severity and Lighting')
plt.title('Color vs Severity, and Lighting')
plt.savefig('cluster.png')
plt.show()

# Export crash count per cluster
for cluster_id in range(k_result):
    cluster_data = merged_df[merged_df['CLUSTERS'] == cluster_id]
    result = cluster_data.sort_values(by='ACCIDENT_COUNT', ascending=False)

    # Save CSV
    filename = f'cluster_result{cluster_id}.csv'
    result.to_csv(filename, index=False)


# #normalize the features
# normalized_features = MinMaxScaler().fit_transform(df_grouped[features])

# sum_of_squared_errors = []
# k_range = range(1, 11)
# for k in k_range:
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(normalized_features)
#     sum_of_squared_errors.append(kmeans.inertia_)

# df_grouped = filtered_vehicle.groupby(grouping)[features].mean().reset_index()




