function additionalSamples = createGoodWaypointSelections(candidateWaypoints, startPoint, goalPoint, numSamples)
    % Create additional good waypoint selections manually
    
    numCandidates = size(candidateWaypoints, 1);
    additionalSamples = zeros(numSamples, numCandidates);
    
    for i = 1:numSamples
        % Select waypoints that form a reasonable path from start to goal
        selection = zeros(1, numCandidates);
        
        % Calculate distances from start and to goal for each waypoint
        distFromStart = vecnorm(candidateWaypoints - startPoint, 2, 2);
        distToGoal = vecnorm(candidateWaypoints - goalPoint, 2, 2);
        
        % Select waypoints that are roughly on the path from start to goal
        for j = 1:numCandidates
            pathFactor = (distFromStart(j) + distToGoal(j)) / norm(goalPoint - startPoint);
            
            % Higher probability for waypoints closer to straight-line path
            if pathFactor < 1.5 && rand() < 0.4
                selection(j) = 1;
            end
        end
        
        % Ensure at least 2 waypoints are selected
        if sum(selection) < 2
            [~, sortedIdx] = sort(distFromStart + distToGoal);
            selection(sortedIdx(1:3)) = 1;
        end
        
        additionalSamples(i, :) = selection;
    end
end

