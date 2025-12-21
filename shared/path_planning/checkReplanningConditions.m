function [needGlobalReplan, replanReason] = checkReplanningConditions(currentTime, lastGlobalPlanTime, currentPosition, globalPath, obstacles, prevObstacles, globalPlanInterval, pathDeviationThreshold, obstacleChangeThreshold)
    % Unified replanning condition checking
    needGlobalReplan = false;
    replanReason = '';

    % Check time interval
    if (currentTime - lastGlobalPlanTime) >= globalPlanInterval
        needGlobalReplan = true;
        replanReason = 'Time interval reached';
        return;
    end
    
    % Check path deviation
    if size(globalPath, 1) > 1
        pathDeviation = pointPathDistance(currentPosition, globalPath);
        if pathDeviation > pathDeviationThreshold
            needGlobalReplan = true;
            replanReason = 'Path deviation threshold exceeded';
            return;
        end
    end
    
    % Check for significant obstacle movement
    for i = 1:size(obstacles, 1)
        if norm(obstacles(i,1:3) - prevObstacles(i,1:3)) > obstacleChangeThreshold
            needGlobalReplan = true;
            replanReason = 'Significant obstacle movement detected';
            return;
        end
    end
end

