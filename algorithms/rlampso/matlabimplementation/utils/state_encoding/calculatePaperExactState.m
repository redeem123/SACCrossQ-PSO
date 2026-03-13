function rlamState = calculatePaperExactState(particles, iter, maxIterations, lastImprovementIteration)
    % PAPER Equations 7-10: Exact state calculation with sin-encoding
    
    % PAPER Equation 7: Iteration progress (paper-exact)
    % Range: [0, 1] as specified in RLAMPSO.pdf Equation 7 (page 10)
    iterationProgress = iter / maxIterations;

    % PAPER Equation 8: Diversity (mean std across dimensions)
    % FIXED: Match Python baseAgent.py:156 + NORMALIZE for sin-encoding
    % Python: diversity = np.mean(np.std(self.xs, axis=0))
    positions = particles.cartesianPositions;
    [numParticles, dims] = size(positions);

    % Calculate standard deviation along dimension axis (across particles)
    diversityPerDim = std(positions, 0, 1);  % std along particles (dim 1)
    diversityRaw = mean(diversityPerDim);

    % CRITICAL FIX: Normalize diversity to [0, 1] range for proper sin-encoding
    % Calculate space diameter for normalization
    minPos = min(positions, [], 1);
    maxPos = max(positions, [], 1);
    spaceDiameter = norm(maxPos - minPos);

    % Normalize: raw diversity / max possible diversity (space diameter)
    % This ensures sin(diversity * 2^i) stays in meaningful range
    if spaceDiameter > 1e-10
        diversity = diversityRaw / spaceDiameter;  % Normalized to ~[0, 1]
    else
        diversity = 0;  % All particles at same position
    end
    
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

