NOTES: 
Modification between codes:
- Path of the excel file
- def find_neighbors(df) function range and the nb of satellites in an orbit must be modified 
exp: For Starlink : for orbit in range(72):  # 72 orbits numbered from 0 to 71
        orbit_data = df[(df['Satellite-number'] > orbit * 22) & (df['Satellite-number'] <= (orbit + 1) * 22)]
For Project kuiper: for orbit in range(36):  # 72 orbits numbered from 0 to 71
        orbit_data = df[(df['Satellite-number'] > orbit *36) & (df['Satellite-number'] <= (orbit + 1) * 36)]  
- To find closest satellite for a Country Latitude and Longitude for the country should be modified
- source_node and target_node should be modified for each constellation
- To have a full view of the output you can hide some outputs like changing to time zone