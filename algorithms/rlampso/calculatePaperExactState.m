function rlamState = calculatePaperExactState(particles, iter, maxIterations, lastImprovementIteration)
    % Paper Equations 7-10: State calculation with sin-encoding
    %
    % Reference: Yin et al., 2023, Section 4.1.1

    % Eq. 7: Iteration progress [0, 1]
    iterationProgress = iter / maxIterations;

    % Eq. 8: Diversity = mean of per-dimension std across particles
    % (matches official Python: np.mean(np.std(self.xs, axis=0)))
    positions = particles.cartesianPositions;
    diversity = mean(std(positions, 0, 1));

    % Eq. 9: Stagnation duration [0, 1]
    stagnationDuration = (iter - lastImprovementIteration) / maxIterations;

    % Eq. 10: Sin-encoding — sin(x * 2^i) for i = 0,1,2,3,4
    basicFeatures = [iterationProgress; diversity; stagnationDuration];
    rlamState = zeros(15, 1);
    idx = 1;
    for f = 1:3
        x = basicFeatures(f);
        for i = 0:4
            rlamState(idx) = sin(x * 2^i);
            idx = idx + 1;
        end
    end
end
