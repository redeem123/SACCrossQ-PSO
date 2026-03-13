function [fitness, components] = evaluatePathFitness(position, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
    % Extract waypoints from position vector
    waypoints = zeros(numWaypoints, 3);
    for i = 1:numWaypoints
        idx = (i-1)*3 + 1;
        waypoints(i,:) = position(idx:idx+2);
    end

    % Create full path with start and goal
    fullPath = [startPoint; waypoints; goalPoint];

    % Calculate duplicate waypoint penalty instead of removing duplicates
    duplicatePenalty = calculateDuplicateWaypointPenalty(fullPath);

    % Use full path for all calculations (no cleaning)
    cleanPath = fullPath;

    % Calculate path length
    pathLength = calculatePathLength(cleanPath);

    % Calculate collision penalty with danger zones (infinite-height cylinders)
    % dangerZones is [x, y, radius]
    collisionPenalty = 0;
    
    % Check each segment for intersection with any danger zone cylinder
    for i = 1:size(cleanPath, 1)-1
        p1 = cleanPath(i, 1:2); % XY only
        p2 = cleanPath(i+1, 1:2); % XY only
        
        for j = 1:size(dangerZones, 1)
            zoneCenter = dangerZones(j, 1:2);
            zoneRadius = dangerZones(j, 3);
            
            % Check segment-circle intersection in 2D
            if isSegmentIntersectingCircle(p1, p2, zoneCenter, zoneRadius)
                collisionPenalty = collisionPenalty + inf;
                break; % Stop checking other zones for this segment if one hit
            end
        end
        
        if isinf(collisionPenalty)
            break; % Stop checking other segments if path is invalid
        end
    end

    % Calculate turning angles penalty
    turningAnglesPenalty = 0;
    if size(cleanPath, 1) > 2
        for i = 2:size(cleanPath, 1)-1
            % Get two consecutive segments
            segment1 = cleanPath(i,:) - cleanPath(i-1,:);
            segment2 = cleanPath(i+1,:) - cleanPath(i,:);

            % Check segment lengths
            len1 = norm(segment1);
            len2 = norm(segment2);

            if len1 > 1e-10 && len2 > 1e-10
                % Normalize the segments
                seg1_unit = segment1 / len1;
                seg2_unit = segment2 / len2;

                % Calculate dot product
                dot_product = dot(seg1_unit, seg2_unit);

                % Clamp to valid range for acos
                dot_product = max(-0.999999, min(0.999999, dot_product));

                % Calculate turning angle 
                turnAngle = acos(dot_product);

                % Apply penalty for any deviation from straight line
                if turnAngle > 0.001  % About 0.057 degrees threshold
                    penaltyContribution = turnAngle * 1;
                    turningAnglesPenalty = turningAnglesPenalty + penaltyContribution;
                end
            end
        end
    end

    % Calculate climbing angles penalty
    climbingAnglesPenalty = 0;
    if size(cleanPath, 1) > 2
        for i = 2:size(cleanPath, 1)-1
            segment1 = cleanPath(i,:) - cleanPath(i-1,:);
            segment2 = cleanPath(i+1,:) - cleanPath(i,:);

            if norm(segment1) > 1e-6 && norm(segment2) > 1e-6
                norm1_h = norm(segment1(1:2));
                norm2_h = norm(segment2(1:2));

                % Calculate climbing angles
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

                changePenalty = climbAngleChange * 2;
                steepPenalty = steepClimbPenalty * 1;

                climbingAnglesPenalty = climbingAnglesPenalty + changePenalty + steepPenalty;
            end
        end
    end

    % Calculate height penalty
    heightPenalty = 0;
    for i = 1:size(cleanPath, 1)
        point = cleanPath(i,:);
        excessHeight = point(3);
        heightPenalty = heightPenalty + excessHeight;
    end

    % Calculate terrain penalty (vectorized segment sampling)
    terrainPenalty = 0;
    samplingStep = 2.0; % Check collision every 2 units
    minClearance = 8.0;
    
    for i = 1:size(cleanPath, 1)-1
        p1 = cleanPath(i,:);
        p2 = cleanPath(i+1,:);
        segmentDist = norm(p2 - p1);
        
        if segmentDist > 0
            % Generate sample points along the segment
            numSamples = ceil(segmentDist / samplingStep);
            t = linspace(0, 1, numSamples + 1)'; % Column vector from 0 to 1
            
            % Interpolate positions (Vectorized)
            % P = P1 + t * (P2 - P1)
            segmentPoints = p1 + t .* (p2 - p1);
            
            pts_x = segmentPoints(:, 1);
            pts_y = segmentPoints(:, 2);
            pts_z = segmentPoints(:, 3);
            
            % Vectorized terrain height query
            tHeights = getTerrainHeight(pts_x, pts_y, terrainGrid, terrainX, terrainY);
            
            % Vectorized clearance check
            clearances = pts_z - tHeights;
            minC = min(clearances);
            
            if minC < minClearance
                % Soft penalty: proportional to how deep we are violated
                % Base penalty 10000 + depth * 1000
                % This allows PSO to optimize out of collision instead of getting stuck at Inf
                penetration = minClearance - minC;
                terrainPenalty = terrainPenalty + 10000 + penetration * 1000;
            end
        end
    end

    % Danger zone penalty (infinite-height cylinders) - Redundant with collisionPenalty but keeps gradient
    dangerZonePenalty = 0;
    for i = 1:size(cleanPath, 1)
        point = cleanPath(i,:);
        for j = 1:size(dangerZones, 1)
            zoneX = dangerZones(j,1);
            zoneY = dangerZones(j,2);
            zoneRadius = dangerZones(j,3);
            % Calculate horizontal distance to danger zone center
            distance = norm([point(1) - zoneX, point(2) - zoneY]) - zoneRadius;
            if distance < 0
                dangerZonePenalty = dangerZonePenalty + inf;  % Inside danger zone
            elseif distance < 5
                dangerZonePenalty = dangerZonePenalty + (5 - distance) * 100;  % Close to danger zone
            end
        end
    end

    % === BALANCED FITNESS: Each component contributes ~25% ===
    % Map size: 100x100x100
    % Normalization factors based on typical values in this map:
    pathNorm = 150;        % Max diagonal ~173, typical path ~100-150
    turningNorm = 3;       % Typical cumulative turning angles (radians)
    climbingNorm = 3;      % Typical cumulative climbing angles (radians)
    heightNorm = 50;       % Typical cumulative height (0-100 range)

    % Normalized components (each roughly 0-1 range)
    pathNormalized = pathLength / pathNorm;
    turningNormalized = turningAnglesPenalty / turningNorm;
    climbingNormalized = climbingAnglesPenalty / climbingNorm;
    heightNormalized = heightPenalty / heightNorm;

    weight = 1000;
    % Equal weights = 25% contribution each
    weight_path = 0.25*weight;
    weight_turning = 0.25*weight;
    weight_climbing = 0.25*weight;
    weight_height = 0.2*weight;

    % Balanced fitness (normalized components with equal weights)
    balancedFitness = weight_path * pathNormalized + ...
                      weight_turning * turningNormalized + ...
                      weight_climbing * climbingNormalized + ...
                      weight_height * heightNormalized;

    % Safety constraints remain absolute (inf for infeasible paths)
    fitness = balancedFitness + collisionPenalty + terrainPenalty + dangerZonePenalty + duplicatePenalty;

    % Store components (now includes duplicate penalty and normalized values)
    components = struct();
    % Raw values
    components.pathLength = pathLength;
    components.collisionPenalty = collisionPenalty;
    components.turningPenalty = turningAnglesPenalty;
    components.climbingPenalty = climbingAnglesPenalty;
    components.heightPenalty = heightPenalty;
    components.terrainPenalty = terrainPenalty;
    components.dangerZonePenalty = dangerZonePenalty;
    components.duplicatePenalty = duplicatePenalty;

    % Normalized values (each ~0-1 range)
    components.pathNormalized = pathNormalized;
    components.turningNormalized = turningNormalized;
    components.climbingNormalized = climbingNormalized;
    components.heightNormalized = heightNormalized;

    % Weighted contributions (each ~25%)
    components.pathContribution = weight_path * pathNormalized;
    components.turningContribution = weight_turning * turningNormalized;
    components.climbingContribution = weight_climbing * climbingNormalized;
    components.heightContribution = weight_height * heightNormalized;

    % Fitness components
    components.balancedFitness = balancedFitness;
    components.totalFitness = fitness;
end

function isIntersect = isSegmentIntersectingCircle(p1, p2, circleCenter, r)
    % Check if line segment p1-p2 intersects circle (center, r)
    d = p2 - p1;
    f = p1 - circleCenter;
    
    a = dot(d, d);
    b = 2 * dot(f, d);
    c = dot(f, f) - r^2;
    
    discriminant = b^2 - 4*a*c;
    
    if discriminant < 0
        isIntersect = false;
    else
        discriminant = sqrt(discriminant);
        t1 = (-b - discriminant) / (2*a);
        t2 = (-b + discriminant) / (2*a);
        
        if (t1 >= 0 && t1 <= 1) || (t2 >= 0 && t2 <= 1)
            isIntersect = true;
        else
            isIntersect = false;
        end
    end
end