function environmentState = calculateEnvironmentState(particles, iter, maxIterations, globalBestPosition)
    % Enhanced environment state calculation with tracking
    popSize = size(particles.cartesianPositions, 1);
    
    % Calculate diversity (fitness-based)
    fitnessStd = std(particles.fitness);
    fitnessMean = mean(particles.fitness);
    if fitnessMean > 1e-10
        diversity = fitnessStd / fitnessMean;  % Coefficient of variation
    else
        diversity = 0;
    end
    
    % Calculate convergence rate
    fitnessVariance = var(particles.fitness);
    convergenceRate = 1 / (1 + fitnessVariance);
    
    % Calculate exploration-exploitation ratio
    iterationRatio = iter / maxIterations;
    
    % No improvement ratio
    noImprovementCount = sum(particles.lastImprovement > 5);
    noImprovementRatio = noImprovementCount / popSize;
    
    % Population clustering around global best
    globalBestPos = globalBestPosition;
    distances = zeros(popSize, 1);
    for i = 1:popSize
        distances(i) = norm(particles.cartesianPositions(i,:) - globalBestPos);
    end
    
    % Calculate space diameter for normalization
    spaceDiameter = calculateSpaceDiameter(particles.cartesianPositions);
    
    % Clustering coefficient (particles within 20% of space diameter)
    clusterThreshold = spaceDiameter * 0.2;
    clusteredCount = sum(distances < clusterThreshold);
    clusteringCoeff = clusteredCount / popSize;
    
    environmentState = struct();
    environmentState.diversity = diversity;
    environmentState.convergenceRate = convergenceRate;
    environmentState.iterationRatio = iterationRatio;
    environmentState.noImprovementRatio = noImprovementRatio;
    environmentState.fitnessVariance = fitnessVariance;
    environmentState.clusteringCoeff = clusteringCoeff;
    environmentState.spaceDiameter = spaceDiameter;
    environmentState.currentIteration = iter;
end

