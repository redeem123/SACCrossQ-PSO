function collisionFree = isCollisionFree(point1, point2, dangerZones, terrainGrid, terrainX, terrainY)
    % Check if a path segment is collision-free with danger zones
    % Danger zones are infinite-height cylinders from ground to infinity

    % Use the shared implementation
    collisionFree = isCollisionFreeDangerZones(point1, point2, dangerZones, terrainGrid, terrainX, terrainY);
end
    
