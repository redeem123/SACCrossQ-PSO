function [metrics, currentPosition, finalPath] = updateSimulationState(metrics, currentTime, localPlanDuration, nextPosition, currentPosition, finalPath, obstacles, terrainGrid, terrainX, terrainY)
    % Update simulation state and metrics
    
    % Update UAV position
    currentPosition = nextPosition;
    finalPath = [finalPath; currentPosition];
    
    % Calculate minimum distance to obstacles
    minObstacleDistance = inf;
    for j = 1:size(obstacles, 1)
        obstaclePos = obstacles(j,1:3);
        obstacleRadius = obstacles(j,4);
        distance = norm(currentPosition - obstaclePos) - obstacleRadius;
        minObstacleDistance = min(minObstacleDistance, distance);
    end
    metrics.obstacleDistances = [metrics.obstacleDistances; minObstacleDistance];
    
    % Calculate height above terrain
    [~, xIndex] = min(abs(terrainX(1,:) - currentPosition(1)));
    [~, yIndex] = min(abs(terrainY(:,1) - currentPosition(2)));
    if xIndex > 0 && yIndex > 0 && xIndex <= size(terrainX, 2) && yIndex <= size(terrainY, 1)
        terrainHeight = terrainGrid(yIndex, xIndex);
        terrainClearance = currentPosition(3) - terrainHeight;
        metrics.terrainClearances = [metrics.terrainClearances; terrainClearance];
    else
        metrics.terrainClearances = [metrics.terrainClearances; 0];
    end
end

