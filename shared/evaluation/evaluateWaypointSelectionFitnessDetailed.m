function [fitness, components] = evaluateWaypointSelectionFitnessDetailed(binarySelection, candidateWaypoints, startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY)
    % Evaluate fitness of binary waypoint selection with detailed component breakdown
    % This function ensures compatibility with the standard fitness component structure
    
    % Extract selected waypoints
    selectedIndices = find(binarySelection == 1);
    
    if isempty(selectedIndices)
        % Return high penalty fitness with zero components
        fitness = 10000;
        components = struct();
        components.pathLength = 0;
        components.collisionPenalty = 10000;
        components.turningPenalty = 0;
        components.climbingPenalty = 0;
        components.heightPenalty = 0;
        components.terrainPenalty = 0;
        components.obstaclePenalty = 0;
        components.duplicatePenalty = 0;
        components.totalFitness = fitness;
        return;
    end
    
    selectedWaypoints = candidateWaypoints(selectedIndices, :);
    
    % Sort waypoints by distance from start to create ordered path
    distFromStart = vecnorm(selectedWaypoints - startPoint, 2, 2);
    [~, sortOrder] = sort(distFromStart);
    orderedWaypoints = selectedWaypoints(sortOrder, :);
    
    % Create full path: start -> ordered waypoints -> goal
    fullPath = [startPoint; orderedWaypoints; goalPoint];
    
    % Calculate detailed fitness components using the same logic as evaluatePathFitness
    
    % 1. Path length
    pathLength = 0;
    for i = 1:size(fullPath, 1)-1
        pathLength = pathLength + norm(fullPath(i+1, :) - fullPath(i, :));
    end
    
    % 2. Collision penalty
    collisionPenalty = 0;
    for i = 1:size(fullPath, 1)-1
        if ~isCollisionFree(fullPath(i, :), fullPath(i+1, :), obstacles, trees, terrainGrid, terrainX, terrainY)
            collisionPenalty = collisionPenalty + 2000;
        end
    end
    
    % 3. Turning angles penalty
    turningPenalty = 0;
    if size(fullPath, 1) > 2
        for i = 2:size(fullPath, 1)-1
            segment1 = fullPath(i,:) - fullPath(i-1,:);
            segment2 = fullPath(i+1,:) - fullPath(i,:);
            
            len1 = norm(segment1);
            len2 = norm(segment2);
            
            if len1 > 1e-10 && len2 > 1e-10
                seg1_unit = segment1 / len1;
                seg2_unit = segment2 / len2;
                dot_product = dot(seg1_unit, seg2_unit);
                dot_product = max(-0.999999, min(0.999999, dot_product));
                turnAngle = acos(dot_product);
                
                if turnAngle > 0.001
                    turningPenalty = turningPenalty + turnAngle * 1;
                end
            end
        end
    end
    
    % 4. Climbing angles penalty
    climbingPenalty = 0;
    if size(fullPath, 1) > 2
        for i = 2:size(fullPath, 1)-1
            segment1 = fullPath(i,:) - fullPath(i-1,:);
            segment2 = fullPath(i+1,:) - fullPath(i,:);
            
            if norm(segment1) > 1e-6 && norm(segment2) > 1e-6
                norm1_h = norm(segment1(1:2));
                norm2_h = norm(segment2(1:2));
                
                if norm1_h > 1e-6
                    climbAngle1 = atan2(segment1(3), norm1_h);
                else
                    climbAngle1 = sign(segment1(3)) * pi/2;
                end
                
                if norm2_h > 1e-6
                    climbAngle2 = atan2(segment2(3), norm2_h);
                else
                    climbAngle2 = sign(segment2(3)) * pi/2;
                end
                
                climbAngleChange = abs(climbAngle2 - climbAngle1);
                steepClimbPenalty = abs(climbAngle1) + abs(climbAngle2);
                
                climbingPenalty = climbingPenalty + climbAngleChange * 2 + steepClimbPenalty * 1;
            end
        end
    end
    
    % 5. Height penalty
    heightPenalty = 0;
    for i = 1:size(fullPath, 1)
        point = fullPath(i,:);
        heightPenalty = heightPenalty + point(3);
    end
    
    % 6. Terrain penalty
    terrainPenalty = 0;
    for i = 1:size(fullPath, 1)
        point = fullPath(i,:);
        [~, xIndex] = min(abs(terrainX(1,:) - point(1)));
        [~, yIndex] = min(abs(terrainY(:,1) - point(2)));
        xIndex = max(1, min(xIndex, size(terrainX, 2)));
        yIndex = max(1, min(yIndex, size(terrainY, 1)));
        terrainHeight = terrainGrid(yIndex, xIndex);
        clearance = point(3) - terrainHeight;
        if clearance < 8
            terrainPenalty = terrainPenalty + (8 - clearance) * 200;
        end
    end
    
    % 7. Obstacle penalty
    obstaclePenalty = 0;
    for i = 1:size(fullPath, 1)
        point = fullPath(i,:);
        for j = 1:size(obstacles, 1)
            obstaclePos = obstacles(j,1:3);
            obstacleRadius = obstacles(j,4);
            distance = norm(point - obstaclePos) - obstacleRadius;
            if distance < 5
                obstaclePenalty = obstaclePenalty + (5 - distance) * 200;
            end
        end
    end
    
    % 8. Duplicate waypoint penalty (waypoint count penalty for IGPSO)
    duplicatePenalty = sum(binarySelection) * 5; % Penalty for using too many waypoints
    
    % Calculate total fitness using same weights as evaluatePathFitness
    fitness = 1.0*pathLength + 1.0*collisionPenalty + 7.0*turningPenalty + 10.0*climbingPenalty + 0.2*heightPenalty + 1.0*terrainPenalty + 1.0*obstaclePenalty + 1.0*duplicatePenalty;
    
    % Store components in standard format
    components = struct();
    components.pathLength = pathLength;
    components.collisionPenalty = collisionPenalty;
    components.turningPenalty = turningPenalty;
    components.climbingPenalty = climbingPenalty;
    components.heightPenalty = heightPenalty;
    components.terrainPenalty = terrainPenalty;
    components.obstaclePenalty = obstaclePenalty;
    components.duplicatePenalty = duplicatePenalty;
    components.totalFitness = fitness;
end

