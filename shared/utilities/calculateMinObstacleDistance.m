function minDist = calculateMinObstacleDistance(path, obstacles)
    % Calculate minimum distance to obstacles along path
    minDist = inf;
    
    for i = 1:size(path, 1)
        point = path(i,:);
        
        for j = 1:size(obstacles, 1)
            obstaclePos = obstacles(j,1:3);
            obstacleRadius = obstacles(j,4);
            
            distance = norm(point - obstaclePos) - obstacleRadius;
            minDist = min(minDist, distance);
        end
    end
    
    if minDist == inf
        minDist = 100; % Large value if no obstacles
    end
end

