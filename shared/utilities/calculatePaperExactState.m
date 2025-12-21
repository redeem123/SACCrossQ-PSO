function rlamState = calculatePaperExactState(particles, iter, maxIterations, lastImprovementIteration)
    % PAPER Equations 7-10: Exact state calculation with sin-encoding
    
    % PAPER Equation 7: Iteration progress
    iterationProgress = iter / maxIterations;
    
    % PAPER Equation 8: Diversity (modified for matrix input)
    positions = particles.cartesianPositions;
    [numParticles, dims] = size(positions);
    meanPosition = mean(positions, 1);
    
    totalDistance = 0;
    for i = 1:numParticles
        totalDistance = totalDistance + norm(positions(i,:) - meanPosition);
    end
    diversity = totalDistance / numParticles;
    
    % Normalize diversity
    maxPossibleDiversity = 200;
    diversity = min(1.0, diversity / maxPossibleDiversity);
    
    % PAPER Equation 9: Stagnation duration
    stagnationDuration = (iter - lastImprovementIteration) / maxIterations;
    
    % PAPER Equation 10: Sin-encoding for each basic feature
    basicFeatures = [iterationProgress; diversity; stagnationDuration];
    rlamState = [];
    
    for featureIdx = 1:3
        x = basicFeatures(featureIdx);
        % PAPER: i takes 0, 1, 2, 3, 4
        for i = 0:4
            statei = sin(x * 2^i);
            rlamState = [rlamState; statei];
        end
    end
    
    % Result: 15-dimensional state vector (3 features Ã— 5 encodings)
end

