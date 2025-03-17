% Define satellite coordinates along with their respective numbers
satellites_data = {
    '1', [47.367918, 1.896666];
    '1285', [49.589776, -5.308779];
    '1286', [52.983444, -31.058085];
    '1175', [52.910178, -53.42047];
    '1176', [50.924695, -79.90941];
    '1177', [43.792809, -101.787387];
    '1178', [33.485162, -118.144703]
};

% Define specific cities and their coordinates
specific_cities = {
    'Paris', [48.8566, 2.3522]; % Paris, France
    'London', [51.5074, -0.1278]; % London, United Kingdom
    'New York', [40.7128, -74.0060]; % New York City, USA
    'Los Angeles', [34.0522, -118.2437]; % Los Angeles, USA
};

% Shortest path between satellites
shortest_path = [1, 1285, 1286, 1175, 1176, 1177, 1178];

% Create a map plot
figure;
worldmap('World');
load coastlines;

% Display Earth map as the background
geoshow(coastlat, coastlon, 'DisplayType', 'polygon', 'FaceColor', [0.8 0.8 1]); % Light blue color for the ocean

% Plot the shortest path between satellites
for i = 1:length(shortest_path) - 1
    start_index = find(strcmp(satellites_data(:,1), num2str(shortest_path(i))));
    end_index = find(strcmp(satellites_data(:,1), num2str(shortest_path(i+1))));
    start_coord = satellites_data{start_index, 2};
    end_coord = satellites_data{end_index, 2};
    geoshow([start_coord(1), end_coord(1)], [start_coord(2), end_coord(2)], 'DisplayType', 'line', 'Color', 'b', 'LineWidth', 2);
end

% Plot the satellites
for i = 1:size(satellites_data, 1)
    plotm(satellites_data{i, 2}(1), satellites_data{i, 2}(2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
    textm(satellites_data{i, 2}(1), satellites_data{i, 2}(2), satellites_data{i, 1}, ...
        'FontSize', 8, 'Color', 'black', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
end

% Plot specific cities
for i = 1:size(specific_cities, 1)
    plotm(specific_cities{i, 2}(1), specific_cities{i, 2}(2), 's', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k');
    textm(specific_cities{i, 2}(1), specific_cities{i, 2}(2), specific_cities{i, 1}, ...
        'FontSize', 8, 'Color', 'black', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
end

% Add title
title('Satellite Coordinates with Specific Cities and Shortest Path');
