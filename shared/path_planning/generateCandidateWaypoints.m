function candidateWaypoints = generateCandidateWaypoints(startPoint, goalPoint, mapSize, terrainGrid, terrainX, terrainY, numCandidates)
    % Generate a diverse set of candidate waypoints for binary selection
    
    candidateWaypoints = zeros(numCandidates, 3);
    
    % Strategy 1: Grid-based sampling (40% of candidates)
    gridCandidates = floor(0.4 * numCandidates);
    gridSize = ceil(sqrt(gridCandidates));
    idx = 1;
    
    for i = 1:gridSize
        for j = 1:gridSize
            if idx > gridCandidates, break; end
            
            x = (i / (gridSize + 1)) * mapSize(1);
            y = (j / (gridSize + 1)) * mapSize(2);
            
            % Find terrain height and add clearance
            [~, xIndex] = min(abs(terrainX(1,:) - x));
            [~, yIndex] = min(abs(terrainY(:,1) - y));
            terrainHeight = terrainGrid(yIndex, xIndex);
            z = terrainHeight + 15 + rand() * 20; % 15-35 units above terrain
            
            candidateWaypoints(idx, :) = [x, y, z];
            idx = idx + 1;
        end
        if idx > gridCandidates, break; end
    end
    
    % Strategy 2: Straight-line interpolation (30% of candidates)
    straightCandidates = floor(0.3 * numCandidates);
    for i = 1:straightCandidates
        if idx > numCandidates, break; end
        
        t = rand(); % Random interpolation factor
        basePoint = startPoint + t * (goalPoint - startPoint);
        
        % Add some lateral deviation
        lateralOffset = (rand(1, 3) - 0.5) * 0.3 * norm(goalPoint - startPoint);
        lateralOffset(3) = abs(lateralOffset(3)); % Keep positive Z offset
        
        waypoint = basePoint + lateralOffset;
        waypoint = max([1, 1, 10], min(waypoint, mapSize));
        
        candidateWaypoints(idx, :) = waypoint;
        idx = idx + 1;
    end
    
    % Strategy 3: Random sampling (30% of candidates)
    for i = idx:numCandidates
        x = rand() * mapSize(1);
        y = rand() * mapSize(2);
        
        [~, xIndex] = min(abs(terrainX(1,:) - x));
        [~, yIndex] = min(abs(terrainY(:,1) - y));
        terrainHeight = terrainGrid(yIndex, xIndex);
        z = terrainHeight + 10 + rand() * 30;
        
        candidateWaypoints(i, :) = [x, y, z];
    end
end

