function features = extractAblatedFeatures(particles, particleIdx, environmentState, config, globalBestPosition, globalBestFitness, startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY)
    % Extract features based on ablation configuration
    % NOTE: Now uses extractAdvancedFeatures (14 features) instead of simplified version

    switch config.featureSet
        case 'full'
            % Use all 14 advanced features
            position = particles.cartesianPositions(particleIdx, :);
            fitness = particles.fitness(particleIdx);
            features = extractAdvancedFeatures(position, fitness, particles, particleIdx, environmentState, globalBestPosition, globalBestFitness, startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY);

        case 'without_stagnation'
            % Use first 10 features only (exclude convergence-stagnation index and spatial consistency)
            position = particles.cartesianPositions(particleIdx, :);
            fitness = particles.fitness(particleIdx);
            allFeatures = extractAdvancedFeatures(position, fitness, particles, particleIdx, environmentState, globalBestPosition, globalBestFitness, startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY);
            % Take first 10 features (exclude features 11-12: convergence-stagnation and spatial consistency)
            features = allFeatures(1:10);

        case 'basic'
            % Basic features only - exactly 10 features
            position = particles.cartesianPositions(particleIdx, :);
            posMatrix = reshape(position, 3, [])';
            meanPos = mean(posMatrix, 1);
            stdPos = std(posMatrix, 0, 1);
            if any(isnan(stdPos))
                stdPos = zeros(1, 3);
            end
            fitness = particles.fitness(particleIdx) / 1000;
            
            features = [meanPos(1)/50, meanPos(2)/50, meanPos(3)/50, ...
                       stdPos(1)/20, stdPos(2)/20, stdPos(3)/20, ...
                       fitness, ...
                       environmentState.diversity, ...
                       environmentState.iterationRatio, ...
                       environmentState.convergenceRate];
            features = features(:);  % Ensure column vector
            
        case 'position'
            % Position features only - exactly 15 features
            position = particles.cartesianPositions(particleIdx, :);
            features = zeros(15, 1);
            for i = 1:min(15, length(position))
                features(i) = position(i) / 50;  % Normalize
            end
            
        case 'environment'
            % Environment features only - exactly 5 features
            features = [environmentState.diversity; ...
                       environmentState.convergenceRate; ...
                       environmentState.iterationRatio; ...
                       environmentState.noImprovementRatio; ...
                       environmentState.fitnessVariance / 1000];
            
        otherwise
            % Default to full features
            position = particles.cartesianPositions(particleIdx, :);
            fitness = particles.fitness(particleIdx);
            features = extractSimpleFeatures(position, fitness, environmentState);
    end
    
    % Ensure features is a column vector
    features = features(:);
end

