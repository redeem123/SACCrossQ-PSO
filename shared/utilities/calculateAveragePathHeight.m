function avgHeight = calculateAveragePathHeight(path, terrainGrid, terrainX, terrainY)
    % Calculate average height of path above terrain
    totalHeight = 0;
    validPoints = 0;
    
    for i = 1:size(path, 1)
        point = path(i,:);
        terrainHeight = getTerrainHeight(point(1), point(2), terrainGrid, terrainX, terrainY);
        heightAboveTerrain = point(3) - terrainHeight;
        totalHeight = totalHeight + heightAboveTerrain;
        validPoints = validPoints + 1;
    end
    
    if validPoints > 0
        avgHeight = totalHeight / validPoints;
    else
        avgHeight = 0;
    end
end

