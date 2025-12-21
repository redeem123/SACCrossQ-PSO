function rlnnpsoState = extractRLNNPSOState(particles, iter, maxIterations, lastImprovementIteration, particleIdx, globalBestPosition, mapSize)
    % Extract particle-specific 3-feature state for RL-NNPSO
    % Uses particle-specific features with sin-encoding
    %
    % Inputs:
    %   particles: Particle structure with cartesianPositions, velocities, bestPositions, lastImprovement
    %   iter: Current iteration
    %   maxIterations: Maximum iterations
    %   lastImprovementIteration: Last iteration with global improvement
    %   particleIdx: Index of current particle
    %   globalBestPosition: Global best position
    %   mapSize: Map dimensions [width, height, depth]
    %
    % Output:
    %   rlnnpsoState: 15-dimensional state vector (3 features × 5 sin-encodings)

    % Feature 1: Distance to global best (normalized) [0, 1]
    distToGlobalBest = norm(particles.cartesianPositions(particleIdx,:) - globalBestPosition);
    maxDistance = norm(mapSize);  % Maximum possible distance in search space
    normalizedDistToGlobalBest = min(1.0, distToGlobalBest / maxDistance);

    % Feature 2: Particle velocity magnitude (normalized) [0, 1]
    velocityMagnitude = norm(particles.velocities(particleIdx,:));
    maxVelocity = 0.15 * mean(mapSize);  % Based on actual velocity clamping
    normalizedVelocity = min(1.0, velocityMagnitude / maxVelocity);

    % Feature 3: Personal stagnation (iterations since personal best improvement) [0, 1]
    personalStagnation = min(1.0, particles.lastImprovement(particleIdx) / 50);

    % Sin-encoding: Each feature gets 5 encodings (i = 0, 1, 2, 3, 4)
    basicFeatures = [normalizedDistToGlobalBest; normalizedVelocity; personalStagnation];
    rlnnpsoState = [];

    for featureIdx = 1:3
        x = basicFeatures(featureIdx);
        for i = 0:4
            statei = sin(x * 2^i);
            rlnnpsoState = [rlnnpsoState; statei];
        end
    end

    % Result: 15-dimensional state vector
end
