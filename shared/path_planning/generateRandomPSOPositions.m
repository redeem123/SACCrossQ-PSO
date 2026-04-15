function positions = generateRandomPSOPositions(popSize, numWaypoints, mapSize, terrainGrid, terrainX, terrainY)
%GENERATERANDOMPSOPOSITIONS Terrain-aware random position initialization for PSO.
%
%   positions = generateRandomPSOPositions(popSize, numWaypoints, mapSize,
%       terrainGrid, terrainX, terrainY)
%
%   Returns a [popSize x (numWaypoints*3)] matrix of random waypoints.
%   Each waypoint [x, y, z] has:
%     x in [0, mapSize(1)]
%     y in [0, mapSize(2)]
%     z = terrain(x,y) + 10 + rand()*20
%
%   Uses O(1) terrain index lookup on regular grids.

    dims = numWaypoints * 3;
    positions = zeros(popSize, dims);

    % Precompute O(1) terrain index mapping
    xMin = terrainX(1, 1);
    xStep = terrainX(1, 2) - xMin;
    yMin = terrainY(1, 1);
    yStep = terrainY(2, 1) - yMin;
    [numRows, numCols] = size(terrainGrid);

    for i = 1:popSize
        for j = 1:numWaypoints
            idx = (j-1)*3 + 1;
            x = rand() * mapSize(1);
            y = rand() * mapSize(2);

            % O(1) terrain height lookup
            xIdx = max(1, min(numCols, round((x - xMin) / xStep) + 1));
            yIdx = max(1, min(numRows, round((y - yMin) / yStep) + 1));
            terrainHeight = terrainGrid(yIdx, xIdx);
            z = terrainHeight + 10 + rand() * 20;

            positions(i, idx:idx+2) = [x, y, z];
        end
    end
end
