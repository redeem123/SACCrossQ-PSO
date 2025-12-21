function pathCost = calculatePathQualityCost(path, terrainGrid, terrainX, terrainY)
    % Calculate Path Quality Cost (C_path)
    pathCost = struct();
    
    % Path Length
    pathCost.pathLength = calculatePathLength(path);
    
    % Smoothness Penalty
    pathCost.smoothnessPenalty = calculateSmoothnessPenalty(path);
    
    % Height Penalty (deviation from optimal flight altitude)
    pathCost.heightPenalty = calculateHeightPenalty(path, terrainGrid, terrainX, terrainY);
    
    % NEW: Average Path Height
    pathCost.averagePathHeight = calculateAveragePathHeight(path, terrainGrid, terrainX, terrainY);
    
    % NEW: Smoothness Index
    pathCost.smoothnessIndex = calculateSmoothnessIndex(path);
    
    % Total Path Quality Cost
    pathCost.total = pathCost.pathLength + pathCost.smoothnessPenalty + pathCost.heightPenalty;
end

