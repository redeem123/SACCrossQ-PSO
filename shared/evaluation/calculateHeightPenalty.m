function heightPenalty = calculateHeightPenalty(path, terrainGrid, terrainX, terrainY)
    % Calculate height penalty (deviation from optimal altitude)
    heightPenalty = 0;
    
    for i = 1:size(path, 1)
        point = path(i,:);

        % Find terrain height
        terrainHeight = getTerrainHeight(point(1), point(2), terrainGrid, terrainX, terrainY);
        heightAboveTerrain = point(3) - terrainHeight;

        % Ensure minimum safety clearance
        if heightAboveTerrain < 8
            heightPenalty = heightPenalty + (8 - heightAboveTerrain) * 200;
        else
            % Penalty for excessive absolute height
            heightPenalty = heightPenalty + point(3) * 2; % Use absolute height
        end
    end
end

