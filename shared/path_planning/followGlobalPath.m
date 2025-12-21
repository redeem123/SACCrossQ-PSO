function nextPosition = followGlobalPath(currentPosition, globalPath, uavSpeed, timeStep)
    % Direct path following without local planning
    
    if isempty(globalPath) || size(globalPath, 1) < 2
        % If no path or insufficient waypoints, stay at current position
        nextPosition = currentPosition;
        return;
    end
    
    % Find closest point on global path
    distances = vecnorm(globalPath - currentPosition, 2, 2);
    [~, closestIdx] = min(distances);
    
    % Look ahead to next waypoint
    if closestIdx < size(globalPath, 1)
        targetPoint = globalPath(closestIdx + 1, :);
    else
        targetPoint = globalPath(end, :);
    end
    
    % Calculate direction vector
    direction = targetPoint - currentPosition;
    distance = norm(direction);
    
    % Calculate movement for this time step
    maxDistance = uavSpeed * timeStep;
    
    if distance <= maxDistance
        % Can reach target point in this time step
        nextPosition = targetPoint;
    else
        % Move towards target point
        direction = direction / distance; % Normalize
        nextPosition = currentPosition + direction * maxDistance;
    end
end

