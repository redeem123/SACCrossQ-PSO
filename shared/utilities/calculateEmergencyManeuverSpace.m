function emergencySpace = calculateEmergencyManeuverSpace(path, obstacles, trees)
    % Calculate available emergency maneuver space
    emergencySpace = inf;
    maneuverRadius = 10.0; % Required space for emergency maneuvers
    
    for i = 1:size(path, 1)
        point = path(i,:);
        availableSpace = maneuverRadius;
        
        % Check space around each path point
        for j = 1:size(obstacles, 1)
            obstaclePos = obstacles(j,1:3);
            obstacleRadius = obstacles(j,4);
            
            distance = norm(point - obstaclePos) - obstacleRadius;
            availableSpace = min(availableSpace, distance);
        end
        
        emergencySpace = min(emergencySpace, availableSpace);
    end
    
    if emergencySpace == inf
        emergencySpace = 50; % Large value if no constraints
    end
end

