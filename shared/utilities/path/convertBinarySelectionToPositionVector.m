function positionVector = convertBinarySelectionToPositionVector(binarySelection, candidateWaypoints, numWaypoints)
    % Convert binary waypoint selection to continuous position vector format
    % This ensures compatibility with standard PSO position representation
    
    selectedIndices = find(binarySelection == 1);
    
    if isempty(selectedIndices)
        % If no waypoints selected, use first few candidates
        selectedIndices = (1:min(numWaypoints, size(candidateWaypoints, 1)))';
    elseif length(selectedIndices) < numWaypoints
        % If too few waypoints, add more from unselected candidates
        unselectedIndices = find(binarySelection == 0);
        additionalNeeded = numWaypoints - length(selectedIndices);
        if length(unselectedIndices) >= additionalNeeded
            % Ensure both are column vectors before concatenation
            selectedIndices = selectedIndices(:);
            additionalIndices = unselectedIndices(1:additionalNeeded);
            additionalIndices = additionalIndices(:);
            selectedIndices = [selectedIndices; additionalIndices];
        else
            % Repeat waypoints if necessary
            selectedIndices = selectedIndices(:); % Ensure column vector
            while length(selectedIndices) < numWaypoints
                numToAdd = min(length(selectedIndices), numWaypoints - length(selectedIndices));
                additionalIndices = selectedIndices(1:numToAdd);
                selectedIndices = [selectedIndices; additionalIndices];
            end
        end
    elseif length(selectedIndices) > numWaypoints
        % If too many waypoints, take the first numWaypoints
        selectedIndices = selectedIndices(1:numWaypoints);
    end
    
    % Ensure we have exactly numWaypoints indices
    selectedIndices = selectedIndices(1:numWaypoints);
    
    % Select waypoints and sort by distance from origin for consistency
    selectedWaypoints = candidateWaypoints(selectedIndices, :);
    
    % Sort waypoints to create a reasonable path ordering
    distances = vecnorm(selectedWaypoints, 2, 2);
    [~, sortOrder] = sort(distances);
    orderedWaypoints = selectedWaypoints(sortOrder, :);
    
    % Convert to position vector format: [x1, y1, z1, x2, y2, z2, ...]
    positionVector = reshape(orderedWaypoints', 1, []);
end

