function duplicatePenalty = calculateDuplicateWaypointPenalty(fullPath)
    % Calculate penalty for duplicate consecutive waypoints
    % High penalty discourages duplicate waypoints
    
    duplicatePenalty = 0;
    tolerance = 1e-6; % Tolerance for considering points identical
    
    for i = 2:size(fullPath, 1)
        currentPoint = fullPath(i,:);
        previousPoint = fullPath(i-1,:);
        
        % Check if current point is too close to previous point (duplicate)
        distance = norm(currentPoint - previousPoint);
        
        if distance <= tolerance
            % High penalty for duplicate waypoints
            duplicatePenalty = duplicatePenalty + 500000;
        elseif distance < 0.5  % Also penalize points that are very close
            % Moderate penalty for points that are too close
            proximityPenalty = (2.0 - distance) * 10000;
            duplicatePenalty = duplicatePenalty + proximityPenalty;
        end
    end
end


