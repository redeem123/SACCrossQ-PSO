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


% function [fitness, components] = evaluatePathFitness(position, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
%     % Fitness Function based on paper: s41598-025-85912-4
%     % "A spherical vector-based adaptive evolutionary particle swarm optimization
%     %  for UAV path planning under threat conditions"
%     %
%     % Equation (10): F_fit = ω1*F1 + ω2*F2 + ω3*F3 + ω4*F4
%     % where:
%     %   F1 = Path cost (Eq. 6)
%     %   F2 = Safety cost (Eq. 7)
%     %   F3 = Height variation cost (Eq. 8)
%     %   F4 = Angle variation cost (Eq. 9)
% 
%     % ===== WEIGHT COEFFICIENTS =====
%     % Note: Paper does not specify exact values for ω1-ω4, a1, a2
%     % BALANCED WEIGHTS to prevent F3 (height variation) from dominating fitness
%     % Previous issue: F3 with M*g multiplier (490) was contributing ~95% of fitness
%     omega1 = 1.0;   % Path length weight
%     omega2 = 5.0;   % Safety cost weight (increased for better obstacle avoidance)
%     omega3 = 0.002; % Height variation weight (reduced from 0.1 to balance M*g=490 multiplier)
%     omega4 = 10.0;  % Angle variation weight (increased for smoother paths)
% 
%     a1 = 5.0;       % Turning angle penalty coefficient (increased for smoother turns)
%     a2 = 5.0;       % Climbing angle change penalty coefficient (increased for smoother climbs)
% 
%     % UAV physical parameters (from paper, page 6, Eq. 8)
%     M = 50;         % UAV mass (kg)
%     g = 9.8;        % Gravitational acceleration (m/s²)
% 
%     % Threat zone parameters (from paper, page 4, Eq. 1-2)
%     % Note: Adjust these based on your obstacle configuration
%     U = 5;          % UAV radius (same as obstacle safety distance in old code)
%     T = 2 * U;      % Risk distance
% 
%     % Extract waypoints from position vector
%     waypoints = zeros(numWaypoints, 3);
%     for i = 1:numWaypoints
%         idx = (i-1)*3 + 1;
%         waypoints(i,:) = position(idx:idx+2);
%     end
% 
%     % Create full path with start and goal
%     fullPath = [startPoint; waypoints; goalPoint];
% 
%     % Calculate duplicate waypoint penalty (keep for practical reasons)
%     duplicatePenalty = calculateDuplicateWaypointPenalty(fullPath);
% 
%     % Use full path for all calculations (no cleaning)
%     cleanPath = fullPath;
% 
%     % ========== F1: PATH COST (Equation 6) ==========
%     % F1 = Σ ||L_i L_{i+1}|| (Euclidean distance)
%     F1 = calculatePathLength(cleanPath);
% 
%     % ========== F2: SAFETY COST (Equation 7) ==========
%     % F2 = Σ Σ C_m(S_i) where C_m(S_i) depends on distance to threat zones
%     % For each segment, calculate distance to all danger zones
%     F2 = 0;
% 
%     for i = 1:size(cleanPath, 1)-1
%         % Calculate segment midpoint for distance measurement
%         segmentMid = (cleanPath(i,:) + cleanPath(i+1,:)) / 2;
% 
%         % Check against obstacles (destruction zones)
%         for j = 1:size(obstacles, 1)
%             obstacleCenter = obstacles(j, 1:3);
%             Dm = obstacles(j, 4);  % Original destruction zone radius
% 
%             % Calculate effective radii (Equations 1-2 from paper)
%             Rm1 = Dm + U;          % Destruction zone (collision)
%             Rm2 = Dm + U + T;      % Threat zone (warning)
% 
%             % Distance from segment to obstacle center
%             Si = norm(segmentMid - obstacleCenter);
% 
%             % Apply safety cost based on zone (Equation 7)
%             if Si <= Rm1
%                 % Inside destruction zone → infeasible
%                 F2 = inf;
%                 break;
%             elseif Si <= Rm2
%                 % Inside threat zone → add penalty
%                 F2 = F2 + (Rm2 - Si);
%             end
%             % else: Si > Rm2 → safe zone, no penalty
%         end
% 
%         if isinf(F2)
%             break;  % Path infeasible, stop checking
%         end
% 
%         % Also check terrain clearance as safety constraint
%         terrainHeight = getTerrainHeight(segmentMid(1), segmentMid(2), terrainGrid, terrainX, terrainY);
%         clearance = segmentMid(3) - terrainHeight;
%         minClearance = 8;  % Minimum safe clearance
% 
%         if clearance < 0
%             % Collision with terrain
%             F2 = inf;
%             break;
%         elseif clearance < minClearance
%             % Too close to terrain
%             F2 = F2 + (minClearance - clearance) * 100;
%         end
%     end
% 
%     % ========== F3: HEIGHT VARIATION COST (Equation 8) ==========
%     % F3 = Σ M*g*|z_{i+1} - z_i|
%     F3 = 0;
%     for i = 1:size(cleanPath, 1)-1
%         heightChange = abs(cleanPath(i+1, 3) - cleanPath(i, 3));
%         F3 = F3 + M * g * heightChange;
%     end
% 
%     % ========== F4: ANGLE VARIATION COST (Equation 9) ==========
%     % F4 = Σ (a1*|Δφ_i| + a2*|Δξ_i|)
%     % where Δφ = turning angle change, Δξ = climbing angle change
%     F4 = 0;
% 
%     if size(cleanPath, 1) > 2
%         for i = 2:size(cleanPath, 1)-1
%             % Get two consecutive segments
%             segment1 = cleanPath(i,:) - cleanPath(i-1,:);
%             segment2 = cleanPath(i+1,:) - cleanPath(i,:);
% 
%             len1 = norm(segment1);
%             len2 = norm(segment2);
% 
%             if len1 > 1e-10 && len2 > 1e-10
%                 % ===== Turning angle Δφ (horizontal plane projection) =====
%                 % Project segments onto XY plane
%                 seg1_xy = segment1(1:2);
%                 seg2_xy = segment2(1:2);
% 
%                 len1_xy = norm(seg1_xy);
%                 len2_xy = norm(seg2_xy);
% 
%                 if len1_xy > 1e-10 && len2_xy > 1e-10
%                     % Calculate turning angle using cross product for direction
%                     crossProd = seg1_xy(1)*seg2_xy(2) - seg1_xy(2)*seg2_xy(1);
%                     dotProd = dot(seg1_xy, seg2_xy);
%                     delta_phi = abs(atan2(crossProd, dotProd));
%                 else
%                     delta_phi = 0;  % Vertical flight, no horizontal turn
%                 end
% 
%                 % ===== Climbing angle change Δξ =====
%                 % Climbing angle ξ = arctan(z / horizontal_distance)
%                 norm1_h = norm(segment1(1:2));
%                 norm2_h = norm(segment2(1:2));
% 
%                 if norm1_h > 1e-6
%                     xi1 = atan2(segment1(3), norm1_h);
%                 else
%                     xi1 = sign(segment1(3)) * pi/2;
%                 end
% 
%                 if norm2_h > 1e-6
%                     xi2 = atan2(segment2(3), norm2_h);
%                 else
%                     xi2 = sign(segment2(3)) * pi/2;
%                 end
% 
%                 delta_xi = abs(xi2 - xi1);
% 
%                 % Add to F4 (Equation 9)
%                 F4 = F4 + a1 * delta_phi + a2 * delta_xi;
%             end
%         end
%     end
% 
%     % ========== FINAL FITNESS (Equation 10) ==========
%     % F_fit = ω1*F1 + ω2*F2 + ω3*F3 + ω4*F4 + duplicate penalty
%     %
%     % Note: Duplicate penalty added for practical implementation
%     %       (not in original paper but prevents degenerate solutions)
%     fitness = omega1 * F1 + omega2 * F2 + omega3 * F3 + omega4 * F4 + duplicatePenalty;
% 
%     % ========== OUTPUT COMPONENTS ==========
%     components = struct();
% 
%     % Paper's four main components
%     components.F1_pathCost = F1;
%     components.F2_safetyCost = F2;
%     components.F3_heightVariation = F3;
%     components.F4_angleVariation = F4;
% 
%     % Weighted components (for analysis)
%     components.F1_weighted = omega1 * F1;
%     components.F2_weighted = omega2 * F2;
%     components.F3_weighted = omega3 * F3;
%     components.F4_weighted = omega4 * F4;
% 
%     % Additional penalty (not in paper)
%     components.duplicatePenalty = duplicatePenalty;
% 
%     % Total fitness
%     components.totalFitness = fitness;
% 
%     % Weight coefficients (for reference)
%     components.weights = struct('omega1', omega1, 'omega2', omega2, ...
%                                 'omega3', omega3, 'omega4', omega4, ...
%                                 'a1', a1, 'a2', a2);
% end
% 
