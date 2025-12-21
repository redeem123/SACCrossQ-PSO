function obstacles = generateFixedObstacles(numObstacles, mapSize, terrainGrid, terrainX, terrainY)
    % Generate fixed obstacles at predetermined positions
    obstacles = zeros(numObstacles, 4);  % [x, y, z, radius]
    
    % Define fixed obstacle positions and properties
    obstaclePositions = [
        30, 30, 25, 2.5;  % [x, y, z, radius]
        50, 50, 20, 2.0;
        70, 30, 30, 3.0;
        40, 70, 25, 2.2;
        25, 60, 22, 1.8;
        65, 65, 32, 2.7;
        85, 45, 28, 1.9;
        55, 85, 23, 2.3;
        45, 25, 26, 2.1;
        75, 55, 31, 2.8;
        35, 45, 21, 1.7;
        60, 25, 27, 2.4;
        80, 70, 29, 2.6;
        20, 75, 24, 2.0;
        90, 85, 30, 2.2
    ];
    
    % Use fixed positions for the specified number of obstacles
    for i = 1:min(numObstacles, size(obstaclePositions, 1))
        x = obstaclePositions(i, 1);
        y = obstaclePositions(i, 2);
        z = obstaclePositions(i, 3);
        radius = obstaclePositions(i, 4);
        
        % Get terrain height at this position to ensure it's above terrain
        [~, xIndex] = min(abs(terrainX(1,:) - x));
        [~, yIndex] = min(abs(terrainY(:,1) - y));
        terrainHeight = terrainGrid(yIndex, xIndex);
        
        % Ensure z is at least radius + 5 units above terrain
        z = max(z, terrainHeight + radius + 5);
        
        obstacles(i,:) = [x, y, z, radius];
    end
end

