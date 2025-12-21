function globalPath = convertBinarySelectionToPath(binarySelection, candidateWaypoints, startPoint, goalPoint)
    % Convert binary waypoint selection to continuous path
    
    selectedIndices = find(binarySelection == 1);
    
    if isempty(selectedIndices)
        % Direct path if no waypoints selected
        globalPath = [startPoint; goalPoint];
        return;
    end
    
    selectedWaypoints = candidateWaypoints(selectedIndices, :);
    
    % Sort waypoints by distance from start
    distFromStart = vecnorm(selectedWaypoints - startPoint, 2, 2);
    [~, sortOrder] = sort(distFromStart);
    orderedWaypoints = selectedWaypoints(sortOrder, :);
    
    % Create full path
    globalPath = [startPoint; orderedWaypoints; goalPoint];
end

