"""Libraries """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
import pytz
import networkx as nx


"""Function 1: Spherical to Cartesian Function"""
def spherical_to_cartesian(latitude, longitude, altitude):
    # Convert degrees to radians
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)
    # Radius of the Earth
    R = 6371  # Assuming the Earth's radius in kilometers
    # Convert spherical coordinates to Cartesian coordinates
    x = (R + altitude) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (R + altitude) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (R + altitude) * np.sin(lat_rad)
    return x, y, z


"""Function 2: Compute Euclidean distance Function"""
def euclidean_distance_3d(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)



"""Read the Starlink Satellites informations: Latitude and Longitude"""
file_path = 'D:\Ahmad\France\Paris-Saclay\Courses\Semester 2\Joint Project\Joint Project\Constellations_ inforamtion\Starlink.xlsx'
df = pd.read_excel(file_path)

# Calculate Cartesian coordinates for all satellites
df['Satellite-number'] = df['Satellite_number']
df['X'], df['Y'], df['Z'] = spherical_to_cartesian(df['Latitude'], df['Longitude'], 550)


# Function to find neighboring satellites (previous and next) for each satellite on the same orbit
def find_neighbors(df):
    neighbors = []

    for orbit in range(72):  # 72 orbits numbered from 0 to 71
        orbit_data = df[(df['Satellite-number'] > orbit * 22) & (df['Satellite-number'] <= (orbit + 1) * 22)]  # Satellites on the current orbit

        for i in range(len(orbit_data)):
            # Get indices of neighboring satellites within the same orbit
            prev_index = (i - 1) % len(orbit_data)
            next_index = (i + 1) % len(orbit_data)

            # Get Cartesian coordinates for current and neighboring satellites
            x_current, y_current, z_current = orbit_data.iloc[i]['X'], orbit_data.iloc[i]['Y'], orbit_data.iloc[i]['Z']
            x_prev, y_prev, z_prev = orbit_data.iloc[prev_index]['X'], orbit_data.iloc[prev_index]['Y'], orbit_data.iloc[prev_index]['Z']
            x_next, y_next, z_next = orbit_data.iloc[next_index]['X'], orbit_data.iloc[next_index]['Y'], orbit_data.iloc[next_index]['Z']

            # Compute distances between current satellite and neighboring satellites within the same orbit
            distance_to_prev = euclidean_distance_3d(x_current, y_current, z_current, x_prev, y_prev, z_prev)
            distance_to_next = euclidean_distance_3d(x_current, y_current, z_current, x_next, y_next, z_next)

            # Add neighboring satellites and their distances to the list
            neighbors.append({'Satellite': int(orbit_data.iloc[i]['Satellite-number']),
                              'Prev_Satellite': int(orbit_data.iloc[prev_index]['Satellite-number']),
                              'Next_Satellite': int(orbit_data.iloc[next_index]['Satellite-number']),
                              'Distance_to_prev': distance_to_prev,
                              'Distance_to_next': distance_to_next,
                              'Orbit': orbit})

    return pd.DataFrame(neighbors)


# Initialize DataFrame to store neighboring satellites and distances
neighbor_distances = find_neighbors(df)

# Display DataFrame containing neighboring satellites and distances
print(neighbor_distances.to_string(index=False))



"""Build the grid"""
def build_3d_spherical_fully_connected_grid(df):
    connections = []
    
    # Iterate over neighboring satellites
    for index, row in df.iterrows():
        # Add connection to previous satellite
        if not pd.isnull(row['Prev_Satellite']):
            connections.append({'Satellite1': int(row['Satellite']),
                                'Satellite2': int(row['Prev_Satellite']),
                                'Distance': row['Distance_to_prev'],
                                'Orbit': int(row['Orbit'])})
        # Add connection to next satellite
        if not pd.isnull(row['Next_Satellite']):
            connections.append({'Satellite1': int(row['Satellite']),
                                'Satellite2': int(row['Next_Satellite']),
                                'Distance': row['Distance_to_next'],
                                'Orbit': int(row['Orbit'])})
    
    return pd.DataFrame(connections)

# Build the 3D spherical fully connected grid
fully_connected_grid = build_3d_spherical_fully_connected_grid(neighbor_distances)
fully_connected_grid.to_csv('connections_data.csv', index=False)

# Plot the grid
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot nodes
ax.scatter(df['X'], df['Y'], df['Z'], color='b', marker='o', s=5)

# Plot edges
for index, row in fully_connected_grid.iterrows():
    x_values = [df.loc[df['Satellite-number'] == row['Satellite1'], 'X'].values[0], 
                df.loc[df['Satellite-number'] == row['Satellite2'], 'X'].values[0]]
    y_values = [df.loc[df['Satellite-number'] == row['Satellite1'], 'Y'].values[0], 
                df.loc[df['Satellite-number'] == row['Satellite2'], 'Y'].values[0]]
    z_values = [df.loc[df['Satellite-number'] == row['Satellite1'], 'Z'].values[0], 
                df.loc[df['Satellite-number'] == row['Satellite2'], 'Z'].values[0]]
    ax.plot(x_values, y_values, z_values, color='k', linewidth=0.4)  # black lines (color='k'), smaller thickness (linewidth=0.5)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

"""Find the closest satellite to Los Angeles"""
LA_latitude = 34.0522
LA_longitude = -118.2437

# Convert Los Angeles coordinates to Cartesian
LA_x, LA_y, LA_z = spherical_to_cartesian(LA_latitude, LA_longitude, 550)

# Calculate distances between Los Angeles and each satellite
distances_to_LA = np.sqrt((df['X'] - LA_x) ** 2 + (df['Y'] - LA_y) ** 2 + (df['Z'] - LA_z) ** 2)

# Find the index of the satellite with the minimum distance to Los Angeles
closest_satellite_index = distances_to_LA.idxmin()

# Get information about the closest satellite
closest_satellite_info = df.loc[closest_satellite_index]

print("Closest Satellite to Los Angeles:")
print("Satellite Number:", int(closest_satellite_info['Satellite-number']))
print("Distance to Los Angeles:", distances_to_LA.min(), "kilometers")


"""Apply Dijkstra and compute the distance from the satellite 1 to the closest satellite to Los Angeles"""
G = nx.Graph()

# Calculate Euclidean distances and add edges to the graph
for i in range(len(df)):
    for j in range(i + 1, len(df)):  # Start from i + 1 to avoid duplicate edges
        distance = euclidean_distance_3d(df.at[i, 'X'], df.at[i, 'Y'], df.at[i, 'Z'],
                                         df.at[j, 'X'], df.at[j, 'Y'], df.at[j, 'Z'])
        # If the distance is within a certain threshold, consider them neighbors
        # Adjust this threshold according to your needs
        if distance < 2000:  # Assuming 2000 km as the threshold
            G.add_edge(df.at[i, 'Satellite-number'], df.at[j, 'Satellite-number'], weight=distance)

# Apply Dijkstra's Algorithm
source_node = 1  # Satellite 1 (Paris)
target_node = 1178
shortest_path = nx.dijkstra_path(G, source=source_node, target=target_node, weight='weight')
shortest_path_distance = nx.dijkstra_path_length(G, source=source_node, target=target_node, weight='weight')

print("Shortest path:", shortest_path)
print("Shortest path distance (in kilometers):", shortest_path_distance)

# Print the latitude and longitude of each satellite in the shortest path
print("Satellite coordinates in the shortest path:")
for satellite_number in shortest_path:
    latitude = df.at[satellite_number - 1, 'Latitude']  # Adjust for 0-based indexing
    longitude = df.at[satellite_number - 1, 'Longitude']  # Adjust for 0-based indexing
    print(f"Satellite {satellite_number}: Latitude: {latitude}, Longitude: {longitude}")



# Speed of light in km/s
SPEED_OF_LIGHT = 299792.458  # in km/s

"""Calculate the RTT"""
def compute_rtt(graph, source_node, target_node):
    # Find the shortest path
    shortest_path = nx.dijkstra_path(graph, source=source_node, target=target_node, weight='weight')
    
    # Initialize RTT
    rtt = 0
    
    # Calculate RTT
    for i in range(len(shortest_path) - 1):
        satellite1 = shortest_path[i]
        satellite2 = shortest_path[i + 1]
        distance = graph[satellite1][satellite2]['weight']
        # Add one-way time for each edge
        one_way_time = distance / SPEED_OF_LIGHT
        rtt += one_way_time
    
    # Double the total one-way time to get RTT
    rtt *= 2
    
    return rtt

# Coordinates for Los Angeles
lat_los_angeles = 34.0522    # Latitude in degrees
lon_los_angeles = -118.2437  # Longitude in degrees
alt_los_angeles = 0          # Altitude in km

# Coordinates for Satellite 1178
lat_satellite1178 = 33.485162  # Latitude in degrees
lon_satellite1178 = -118.144703 # Longitude in degrees
alt_satellite1178 = 550        # Altitude in km

# Compute distance between Satellite 1178 and Los Angeles
distance_satellite1178_los_angeles = euclidean_distance_3d(lat_satellite1178, lon_satellite1178, alt_satellite1178,
                                                        lat_los_angeles, lon_los_angeles, alt_los_angeles)

# Compute one-way time
one_way_time = distance_satellite1178_los_angeles / SPEED_OF_LIGHT

# Compute RTT
rtt_satellite1178_los_angeles = 2 * one_way_time

print("Round-Trip Time (RTT) between Satellite 1178 and Los Angeles:", rtt_satellite1178_los_angeles, "seconds")


# Coordinates for Paris
lat_paris = 48.8566  # Latitude in degrees
lon_paris = 2.3522   # Longitude in degrees
alt_paris = 0        # Altitude in km

# Coordinates for Satellite 1
lat_satellite1 = 47.367918  # Latitude in degrees
lon_satellite1 = 1.896666    # Longitude in degrees
alt_satellite1 = 550         # Altitude in km

# Compute distance between Paris and Satellite 1
distance_paris_satellite1 = euclidean_distance_3d(lat_paris, lon_paris, alt_paris,
                                               lat_satellite1, lon_satellite1, alt_satellite1)

# Compute one-way time
one_way_time = distance_paris_satellite1 / SPEED_OF_LIGHT

# Compute RTT
rtt_paris_satellite1 = 2 * one_way_time 

print("Round-Trip Time (RTT) between Paris and Satellite 1:", rtt_paris_satellite1, "seconds")
# Compute RTT between satellite 1 and satellite 1178
rtt = compute_rtt(G, source_node, target_node) + rtt_paris_satellite1 + rtt_satellite1178_los_angeles 
print("Round-Trip Time (RTT):", rtt, "seconds")


"""Time zone: France"""

# list of time to periapsis in seconds for each satellite
time_to_periapsis_seconds = df['Time_to_periapsis'].tolist()

# Current UTC time
current_utc_time = datetime.utcnow()

# Convert time to periapsis to datetime objects and adjust for the current UTC time
satellite_time_of_periapsis_utc = [current_utc_time + timedelta(seconds=time) for time in time_to_periapsis_seconds]

# Convert UTC time to France time for each satellite
france_tz = pytz.timezone('Europe/Paris')
satellite_time_of_periapsis_france = [utc_time.astimezone(france_tz) for utc_time in satellite_time_of_periapsis_utc]

# Print the time of periapsis passage in France for each satellite
for i, france_time in enumerate(satellite_time_of_periapsis_france):
   print("Satellite", i+1, "Time of periapsis passage in France:", france_time.strftime("%Y-%m-%d %H:%M:%S %Z%z"))



