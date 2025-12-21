function basicFeatures = extractRichFeatures(particles, iter, maxIterations, lastImprovementIteration, globalBestFitness)
    % Extract rich features from PSO state for Transformer encoder
    %
    % Returns 9 basic features that capture different aspects of PSO dynamics:
    %   1. Iteration progress
    %   2. Population diversity (spatial spread)
    %   3. Fitness diversity (fitness spread)
    %   4. Stagnation duration
    %   5. Best fitness (normalized)
    %   6. Velocity magnitude (mean)
    %   7. Convergence speed
    %   8. Exploration vs exploitation indicator
    %   9. Fitness improvement rate
    %
    % These features provide comprehensive information about:
    %   - Search phase (early exploration vs late exploitation)
    %   - Population health (diversity, convergence)
    %   - Progress indicators (stagnation, improvement rate)
    %
    % Inputs:
    %   particles: Struct with PSO particle data
    %   iter: Current iteration
    %   maxIterations: Maximum iterations
    %   lastImprovementIteration: Last iteration with improvement
    %   globalBestFitness: Current global best fitness value
    %
    % Outputs:
    %   basicFeatures: [9×1] feature vector

    positions = particles.cartesianPositions;
    velocities = particles.velocities;
    fitness = particles.fitness;
    [numParticles, dims] = size(positions);

    % Initialize feature vector
    basicFeatures = zeros(9, 1);

    % ===== 1. ITERATION PROGRESS =====
    % Normalized iteration count [0, 1]
    basicFeatures(1) = iter / maxIterations;

    % ===== 2. POPULATION DIVERSITY (Spatial) =====
    % Mean standard deviation across dimensions (normalized)
    diversityPerDim = std(positions, 0, 1);
    diversityRaw = mean(diversityPerDim);

    % Normalize by search space diameter
    minPos = min(positions, [], 1);
    maxPos = max(positions, [], 1);
    spaceDiameter = norm(maxPos - minPos);
    
    if isinf(spaceDiameter) || isnan(spaceDiameter)
        spaceDiameter = 1e5; % Default large value if invalid
    end

    if spaceDiameter > 1e-10
        basicFeatures(2) = diversityRaw / spaceDiameter;
    else
        basicFeatures(2) = 0;
    end

    % ===== 3. FITNESS DIVERSITY =====
    % Standard deviation of fitness values (normalized)
    validFitness = fitness(~isinf(fitness));
    if isempty(validFitness)
        fitnessStd = 0;
        fitnessMean = 1; % Avoid division by zero
    else
        fitnessStd = std(validFitness);
        fitnessMean = mean(validFitness);
    end

    if abs(fitnessMean) > 1e-10
        basicFeatures(3) = fitnessStd / abs(fitnessMean);
    else
        basicFeatures(3) = 0;
    end

    % ===== 4. STAGNATION DURATION =====
    % How long since last improvement (normalized)
    basicFeatures(4) = (iter - lastImprovementIteration) / maxIterations;

    % ===== 5. BEST FITNESS (Normalized) =====
    % Use log-scale for large fitness values
    if globalBestFitness > 0 && ~isinf(globalBestFitness)
        basicFeatures(5) = tanh(log10(globalBestFitness + 1));
    else
        basicFeatures(5) = 0;
    end

    % ===== 6. VELOCITY MAGNITUDE =====
    % Mean velocity magnitude across all particles
    velocityMagnitudes = sqrt(sum(velocities.^2, 2));
    meanVelocity = mean(velocityMagnitudes);

    % Normalize by space diameter
    if spaceDiameter > 1e-10
        basicFeatures(6) = tanh(meanVelocity / spaceDiameter);
    else
        basicFeatures(6) = 0;
    end

    % ===== 7. CONVERGENCE SPEED =====
    % How quickly particles are converging toward gbest
    globalBestPosition = particles.bestPositions(1, :);  % Assume stored
    distancesToGbest = sqrt(sum((positions - globalBestPosition).^2, 2));
    meanDistanceToGbest = mean(distancesToGbest);

    % Normalize and invert (closer = higher value)
    if spaceDiameter > 1e-10
        basicFeatures(7) = 1.0 - tanh(meanDistanceToGbest / spaceDiameter);
    else
        basicFeatures(7) = 1.0;  % Fully converged
    end

    % ===== 8. EXPLORATION VS EXPLOITATION INDICATOR =====
    % High diversity + high velocity = exploration
    % Low diversity + low velocity = exploitation
    explorationScore = basicFeatures(2) * basicFeatures(6);
    basicFeatures(8) = explorationScore;

    % ===== 9. FITNESS IMPROVEMENT RATE =====
    % Estimate of how quickly fitness is improving
    % (This is a proxy; actual rate computed in reward function)
    if iter > 1 && lastImprovementIteration > 0
        iterationsSinceImprovement = iter - lastImprovementIteration;
        improvementRate = 1.0 / (1.0 + iterationsSinceImprovement);
        basicFeatures(9) = improvementRate;
    else
        basicFeatures(9) = 1.0;  % Early in search
    end

    % Clip all features to reasonable range [-2, 2]
    basicFeatures = max(-2, min(2, basicFeatures));
end
