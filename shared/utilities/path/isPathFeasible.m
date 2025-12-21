function isFeasible = isPathFeasible(position, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
    % Check if a particle represents a feasible (collision-free) path

    % Extract waypoints from position vector
    waypoints = zeros(numWaypoints, 3);
    for i = 1:numWaypoints
        idx = (i-1)*3 + 1;
        waypoints(i,:) = position(idx:idx+2);
    end

    % Create full path with start and goal
    fullPath = [startPoint; waypoints; goalPoint];

    % Check if entire path is collision-free with danger zones
    for i = 1:size(fullPath, 1)-1
        if ~isCollisionFreeDangerZones(fullPath(i,:), fullPath(i+1,:), dangerZones, terrainGrid, terrainX, terrainY)
            isFeasible = false;
            return;
        end
    end

    isFeasible = true;
end

