function minClearance = calculateMinTerrainClearance(path, terrainGrid, terrainX, terrainY)
    % Calculate minimum terrain clearance along path
    minClearance = inf;
    
    for i = 1:size(path, 1)
        point = path(i,:);
        
        terrainHeight = getTerrainHeight(point(1), point(2), terrainGrid, terrainX, terrainY);
        clearance = point(3) - terrainHeight;
        minClearance = min(minClearance, clearance);
    end
    
    if minClearance == inf
        minClearance = 50; % Large value if no terrain data
    end
end

