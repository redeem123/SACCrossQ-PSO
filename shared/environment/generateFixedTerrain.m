function [terrainGrid, terrainX, terrainY] = generateFixedTerrain(mapSize, scenarioIdx)
    % Load terrain from ChrismasTerrain2.tif file
    % Get the project root directory (go up from shared/environment to project root)
    currentDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(fileparts(currentDir));
    terrainFilePath = fullfile(projectRoot, 'data', 'ChrismasTerrain2.tif');

    H = imread(terrainFilePath); % Get elevation data
    H(H < 0) = 0; % Ensure non-negative values
    
    % Get dimensions of loaded terrain
    MAPSIZE_X = size(H, 2); % x index: columns of H
    MAPSIZE_Y = size(H, 1); % y index: rows of H
    
    % Create coordinate grids based on mapSize
    x_terrain = linspace(0, mapSize(1), MAPSIZE_X);
    y_terrain = linspace(0, mapSize(2), MAPSIZE_Y);
    [X_terrain, Y_terrain] = meshgrid(x_terrain, y_terrain);
    
    % Create output grids with standard resolution (100x100)
    x_new = linspace(0, mapSize(1), 100);
    y_new = linspace(0, mapSize(2), 100);
    [terrainX, terrainY] = meshgrid(x_new, y_new);
    
    % Convert terrain data to double and scale for UAV altitudes
    H_double = double(H);
    
    % Scale terrain heights to fit within 0 to 40 (max height 50, leaving clearance)
    maxHeightOrig = max(H_double(:));
    if maxHeightOrig > 0
        H_scaled = H_double * (40 / maxHeightOrig); % Scale to max height of 40
    else
        H_scaled = H_double;
    end
    
    % Interpolate terrain data to standard 100x100 grid
    terrainGrid = interp2(X_terrain, Y_terrain, H_scaled, terrainX, terrainY, 'linear', 0);
    
    % Ensure non-negative terrain
    terrainGrid = max(terrainGrid, 0);
end

