function obstacles = updateObstaclePositions(initialObstacles, obstacleDynamics, currentTime, mapSize, terrainGrid, terrainX, terrainY)
    % Update obstacle positions based on their dynamics and current time
    obstacles = initialObstacles;
    
    for i = 1:size(obstacles, 1)
        % Get the dynamics parameters for this obstacle
        velocity = obstacleDynamics.velocities(i,:);
        oscillation = obstacleDynamics.oscillations(i,:);
        phase = obstacleDynamics.phases(i,:);
        
        % Calculate the new position
        linearMovement = velocity * currentTime;
        oscillatoryMovement = oscillation .* sin(2*pi*obstacleDynamics.frequencies(i,:)*currentTime + phase);
        
        % Update obstacle position
        obstacles(i,1:3) = initialObstacles(i,1:3) + linearMovement + oscillatoryMovement;
        
        % Ensure obstacles stay within boundaries
        for j = 1:3
            if obstacles(i,j) - obstacles(i,4) < 0
                obstacles(i,j) = obstacles(i,4);
            elseif obstacles(i,j) + obstacles(i,4) > mapSize(j)
                obstacles(i,j) = mapSize(j) - obstacles(i,4);
            end
        end
        
        % Ensure obstacles stay above terrain
        [~, xIndex] = min(abs(terrainX(1,:) - obstacles(i,1)));
        [~, yIndex] = min(abs(terrainY(:,1) - obstacles(i,2)));
        terrainHeight = terrainGrid(yIndex, xIndex);
        
        minHeight = terrainHeight + obstacles(i,4) + 3; % Obstacle radius plus safety margin
        if obstacles(i,3) < minHeight
            obstacles(i,3) = minHeight;
        end
    end
end

