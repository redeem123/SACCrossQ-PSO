function height = getTerrainHeight(x, y, terrainGrid, terrainX, terrainY)
    % Get terrain height at given x,y coordinates using linear interpolation
    % Inputs:
    %   x, y - Coordinates to query (scalars or vectors)
    %   terrainGrid - 2D grid of terrain heights (Z)
    %   terrainX, terrainY - Coordinate grids (from meshgrid)
    % Output:
    %   height - Interpolated terrain height at (x,y) (same size as x,y)

    % Use linear interpolation for smooth terrain handling
    % interp2 is optimized for vector inputs when x and y are arrays.
    % Extrapolation value set to 1e9 to penalize out-of-bounds queries
    height = interp2(terrainX, terrainY, terrainGrid, x, y, 'linear', 1e9);
end