function collisionRisk = calculateCollisionRisk(path, obstacles, trees, terrainGrid, terrainX, terrainY)
    % Calculate collision risk probability
    collisionRisk = 0;
    riskThreshold = 8.0; % Safety threshold
    
    for i = 1:size(path, 1)
        point = path(i,:);
        
        % Risk from obstacles
        for j = 1:size(obstacles, 1)
            obstaclePos = obstacles(j,1:3);
            obstacleRadius = obstacles(j,4);
            
            distance = norm(point - obstaclePos) - obstacleRadius;
            if distance < riskThreshold
                collisionRisk = collisionRisk + (riskThreshold - distance) / riskThreshold;
            end
        end
        
        % Risk from terrain
        terrainHeight = getTerrainHeight(point(1), point(2), terrainGrid, terrainX, terrainY);
        clearance = point(3) - terrainHeight;

        if clearance < riskThreshold
            collisionRisk = collisionRisk + (riskThreshold - clearance) / riskThreshold;
        end
    end
end

