function height = getTerrainHeight(x, y, terrainGrid, terrainX, terrainY)
    % Get terrain height at given x,y coordinates
    % Inputs:
    %   x, y - Coordinates to query (scalars or vectors)
    %   terrainGrid - 2D grid of terrain heights
    %   terrainX, terrainY - Coordinate grids
    % Output:
    %   height - Terrain height at (x,y)

    % Match the shared utility semantics so callers remain correct even when
    % this older RLAMPSO utility shadows the shared path implementation.
    height = interp2(terrainX, terrainY, terrainGrid, x, y, 'linear', 1e9);
end
