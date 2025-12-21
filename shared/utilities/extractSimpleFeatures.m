function features = extractSimpleFeatures(position, fitness, environmentState)
    % !!!!! DEPRECATED !!!!!
    % This function extracts 32 features (old system)
    % USE extractAdvancedFeatures.m (14 features) instead
    % This function is kept only for backward compatibility with old code
    % !!!!! DEPRECATED !!!!!

    warning('extractSimpleFeatures:deprecated', ...
        'extractSimpleFeatures is DEPRECATED. Use extractAdvancedFeatures (14 features) instead. This function returns 32 features which may cause dimension mismatch.');

    % Enhanced feature extraction with 32 meaningful features (no padding)
    
    position = position(:)';
    numWaypoints = length(position) / 3;
    
    % Basic position statistics (9 features)
    if numWaypoints > 0 && length(position) >= 3
        posMatrix = reshape(position, 3, [])';
        meanPos = mean(posMatrix, 1);
        stdPos = std(posMatrix, 0, 1);
        minPos = min(posMatrix, [], 1);
        maxPos = max(posMatrix, [], 1);
        
        % Ensure row vectors and handle NaN
        meanPos = meanPos(:)';
        stdPos = stdPos(:)';
        minPos = minPos(:)';
        maxPos = maxPos(:)';
        
        if any(isnan(stdPos))
            stdPos = zeros(1, 3);
        end
    else
        meanPos = zeros(1, 3);
        stdPos = zeros(1, 3);
        minPos = zeros(1, 3);
        maxPos = zeros(1, 3);
    end
    
    % Path characteristics (16 features)
    pathLength = 0;
    totalTurnAngle = 0;
    totalClimbAngle = 0;
    maxSegmentLength = 0;
    minSegmentLength = inf;
    
    if numWaypoints > 1
        for i = 1:numWaypoints-1
            idx1 = (i-1)*3 + 1;
            idx2 = i*3 + 1;
            if idx2+2 <= length(position)
                segment = position(idx2:idx2+2) - position(idx1:idx1+2);
                segmentLength = norm(segment);
                pathLength = pathLength + segmentLength;
                maxSegmentLength = max(maxSegmentLength, segmentLength);
                minSegmentLength = min(minSegmentLength, segmentLength);
                
                % Turn and climb angles
                if i < numWaypoints-1
                    idx3 = (i+1)*3 + 1;
                    if idx3+2 <= length(position)
                        nextSegment = position(idx3:idx3+2) - position(idx2:idx2+2);
                        if norm(segment) > 0 && norm(nextSegment) > 0
                            % Turn angle
                            turnAngle = acos(max(-1, min(1, dot(segment(1:2)/norm(segment(1:2)), nextSegment(1:2)/norm(nextSegment(1:2))))));
                            totalTurnAngle = totalTurnAngle + turnAngle;
                            
                            % Climb angle
                            climbAngle1 = atan2(segment(3), norm(segment(1:2)));
                            climbAngle2 = atan2(nextSegment(3), norm(nextSegment(1:2)));
                            totalClimbAngle = totalClimbAngle + abs(climbAngle2 - climbAngle1);
                        end
                    end
                end
            end
        end
        
        if minSegmentLength == inf
            minSegmentLength = 0;
        end
    end
    
    % Path smoothness metrics
    avgTurnAngle = totalTurnAngle / max(1, numWaypoints-2);
    avgClimbAngle = totalClimbAngle / max(1, numWaypoints-2);
    pathVariability = maxSegmentLength - minSegmentLength;
    
    % Fitness-related features
    normalizedFitness = min(fitness / 1000, 1);
    fitnessGradient = tanh(fitness / 500);
    
    % Environment state features
    diversity = environmentState.diversity;
    convergence = environmentState.convergenceRate;
    iteration = environmentState.iterationRatio;
    stagnation = environmentState.noImprovementRatio;
    
    % Derived features
    explorationFactor = diversity * (1 - iteration);
    exploitationFactor = convergence * iteration;
    adaptationNeed = stagnation * (1 - convergence);
    
    % Height feature (1 feature) - Average Height Above Terrain
    avgHeightAboveTerrain = 0;
    if numWaypoints > 0
        % Approximate terrain height calculation (simplified)
        % In real implementation, use actual terrainGrid, terrainX, terrainY
        totalHeightAboveTerrain = 0;
        validPoints = 0;
        
        for i = 1:numWaypoints
            idx = (i-1)*3 + 1;
            if idx+2 <= length(position)
                point = position(idx:idx+2);
                % Simplified terrain height estimation
                estimatedTerrainHeight = 5 + 15 * sin(point(1)/50) * cos(point(2)/50);
                heightAboveTerrain = point(3) - estimatedTerrainHeight;
                totalHeightAboveTerrain = totalHeightAboveTerrain + heightAboveTerrain;
                validPoints = validPoints + 1;
            end
        end
        
        if validPoints > 0
            avgHeightAboveTerrain = totalHeightAboveTerrain / validPoints;
        else
            avgHeightAboveTerrain = 0;  % Default value instead of NaN
        end
    end
    
    % Swarm nonlinear dynamic features (2 features)
    earlyExplorationEffectiveness = diversity * (1 - iteration);
    convergenceStagnationBalance = convergence * (1 - stagnation);
    
    % Current interaction features (4 features)
    diversityConvergence = diversity * convergence;
    iterationStagnation = iteration * stagnation;
    explorationFitness = explorationFactor * normalizedFitness;
    convergenceFitness = convergence * normalizedFitness;
    % Build features array - exactly 32 features, no padding
    features = [];
    
    % Position features (9 total)
    features = [features, meanPos / 50];           % Features 1-3
    features = [features, stdPos / 20];            % Features 4-6
    features = [features, (maxPos - minPos) / 100]; % Features 7-9
    
    % Path characteristics (16 total)
    features = [features, pathLength / 200];        % Feature 10
    features = [features, avgTurnAngle / pi];       % Feature 11
    features = [features, avgClimbAngle / pi];      % Feature 12
    features = [features, pathVariability / 50];    % Feature 13
    features = [features, normalizedFitness];       % Feature 14
    features = [features, fitnessGradient];         % Feature 15
    features = [features, diversity];               % Feature 16
    features = [features, convergence];             % Feature 17
    features = [features, iteration];               % Feature 18
    features = [features, stagnation];              % Feature 19
    features = [features, explorationFactor];       % Feature 20
    features = [features, exploitationFactor];      % Feature 21
    features = [features, adaptationNeed];          % Feature 22
    features = [features, maxSegmentLength / 50];   % Feature 23
    features = [features, minSegmentLength / 50];   % Feature 24
    features = [features, numWaypoints / 10];       % Feature 25
    
    % Height feature (1 total)
    features = [features, avgHeightAboveTerrain / 50]; % Feature 26
    
    % Swarm nonlinear dynamic features (2 total)
    features = [features, earlyExplorationEffectiveness]; % Feature 27
    features = [features, convergenceStagnationBalance];  % Feature 28
    
    % Current interaction features (4 total)
    features = [features, diversityConvergence];    % Feature 29
    features = [features, iterationStagnation];     % Feature 30
    features = [features, explorationFitness];      % Feature 31
    features = [features, convergenceFitness];      % Feature 32
    
    % Convert to column vector
    features = features(:);
    
    % Final validation - ensure exactly 32 features
    if length(features) ~= 32
        error('Feature extraction produced %d features instead of 32', length(features));
    end
    
    % Ensure all features are finite
    features(~isfinite(features)) = 0;
end

