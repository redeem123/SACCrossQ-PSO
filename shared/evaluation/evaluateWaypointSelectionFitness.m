function fitness = evaluateWaypointSelectionFitness(binarySelection, candidateWaypoints, startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY)
    % Evaluate fitness of a binary waypoint selection for UAV path planning
    
    % Extract selected waypoints
    selectedIndices = find(binarySelection == 1);
    
    if isempty(selectedIndices)
        fitness = 10000; % Very bad fitness for empty selection
        return;
    end
    
    selectedWaypoints = candidateWaypoints(selectedIndices, :);
    
    % Sort waypoints by distance from start to create ordered path
    distFromStart = vecnorm(selectedWaypoints - startPoint, 2, 2);
    [~, sortOrder] = sort(distFromStart);
    orderedWaypoints = selectedWaypoints(sortOrder, :);
    
    % Create full path: start -> ordered waypoints -> goal
    fullPath = [startPoint; orderedWaypoints; goalPoint];
    
    % Calculate path length
    pathLength = 0;
    for i = 1:size(fullPath, 1)-1
        pathLength = pathLength + norm(fullPath(i+1, :) - fullPath(i, :));
    end
    
    % Calculate collision penalty
    collisionPenalty = 0;
    for i = 1:size(fullPath, 1)-1
        if ~isCollisionFree(fullPath(i, :), fullPath(i+1, :), obstacles, trees, terrainGrid, terrainX, terrainY)
            collisionPenalty = collisionPenalty + 1000;
        end
    end
    
    % Calculate smoothness penalty (turning angles)
    smoothnessPenalty = 0;
    if size(fullPath, 1) > 2
        for i = 2:size(fullPath, 1)-1
            v1 = fullPath(i, :) - fullPath(i-1, :);
            v2 = fullPath(i+1, :) - fullPath(i, :);
            
            if norm(v1) > 0 && norm(v2) > 0
                cosAngle = dot(v1, v2) / (norm(v1) * norm(v2));
                cosAngle = max(-1, min(1, cosAngle));
                angle = acos(cosAngle);
                smoothnessPenalty = smoothnessPenalty + angle * 10;
            end
        end
    end
    
    % Waypoint count penalty (prefer fewer waypoints)
    waypointPenalty = sum(binarySelection) * 5;
    
    % Combined fitness (lower is better)
    fitness = pathLength + collisionPenalty + smoothnessPenalty + waypointPenalty;
end

