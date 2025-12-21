function collisionFree = isCollisionFree(point1, point2, dangerZones, terrainGrid, terrainX, terrainY)
    % Check if a path segment is collision-free - optimized version
    collisionFree = true;

    % Pre-calculate segment vector for efficiency
    segmentVec = point2 - point1;
    segmentLength = norm(segmentVec);
    
    % Skip collision checking for very short segments
    if segmentLength < 0.1
        return;
    end
    
    % Use adaptive step size based on segment length
    stepSize = min(0.05, segmentLength / 20);
    tValues = 0:stepSize:1;
    
    % Pre-calculate all test points along the segment
    numPoints = length(tValues);
    testPoints = zeros(numPoints, 3);
    for i = 1:numPoints
        testPoints(i, :) = point1 + tValues(i) * segmentVec;
    end
    
    % Check collision with dynamic obstacles - vectorized distance check
    if ~isempty(obstacles)
        for i = 1:size(obstacles, 1)
            obstaclePos = obstacles(i, 1:3);
            obstacleRadius = obstacles(i, 4);
            
            % Calculate distances to all test points at once
            distances = sqrt(sum((testPoints - obstaclePos).^2, 2));
            if any(distances <= obstacleRadius)
                collisionFree = false;
                return;
            end
        end
    end
        
    % Check collision with trees and terrain for each test point
    for pointIdx = 1:numPoints
        point = testPoints(pointIdx, :);

        % Check collision with trees
        for i = 1:size(trees, 1)
            treePos = trees(i,1:3);
            trunkRadius = trees(i,4);
            canopyRadius = trees(i,5);
            isCylinder = trees(i,6);
            treeHeight = trees(i,7);  % Get actual height

            % Check trunk (cylinder)
            treeTop = treePos + [0, 0, treeHeight]; % Use actual height
            [closestPoint, ~] = projectPointOnSegment(point, treePos, treeTop);
            trunkDistance = norm(point - closestPoint) - trunkRadius;

            if ~isCylinder
                % Check canopy (sphere) - only for non-cylinders
                canopyCenter = treePos + [0, 0, treeHeight * 0.8]; % Canopy at 80% height
                canopyDistance = norm(point - canopyCenter) - canopyRadius;

                % If point is inside either trunk or canopy
                if trunkDistance <= 0 || canopyDistance <= 0
                    collisionFree = false;
                    return;
                end
            else
                % For cylinders, only check trunk collision
                if trunkDistance <= 0
                    collisionFree = false;
                    return;
                end
            end
        end

        % Check terrain clearance
        terrainHeight = getTerrainHeight(point(1), point(2), terrainGrid, terrainX, terrainY);
        
        % If point is below terrain or too close
        if point(3) < terrainHeight + 5 % 5 units minimum clearance
            collisionFree = false;
            return;
        end
    end
end
    
