function basicFeatures = extractRichFeatures(particles, iter, maxIterations, lastImprovementIteration, globalBestFitness)
    % Extract 9 swarm-level features for the cross-scale state encoder.
    %
    % Feature layout:
    %   1. Iteration progress
    %   2. Spatial diversity ratio
    %   3. Elite fitness advantage
    %   4. Global stagnation pressure
    %   5. Feasible-particle ratio
    %   6. Constraint-pressure intensity
    %   7. Swarm attraction to the actual global best
    %   8. Velocity alignment toward the actual global best
    %   9. Particle-level stagnation pressure

    positions = particles.cartesianPositions;
    velocities = particles.velocities;
    fitness = particles.fitness;
    [~, dims] = size(positions);

    epsVal = 1e-8;

    % Initialize feature vector
    basicFeatures = zeros(9, 1);

    % ===== 1. ITERATION PROGRESS =====
    basicFeatures(1) = iter / maxIterations;

    % ===== 2. POPULATION DIVERSITY (Spatial) =====
    diversityPerDim = std(positions, 0, 1);
    minPos = min(positions, [], 1);
    maxPos = max(positions, [], 1);
    positionSpan = max(maxPos - minPos, epsVal);
    spanMean = mean(positionSpan);
    basicFeatures(2) = mean(diversityPerDim ./ positionSpan);

    % ===== 3. ELITE FITNESS ADVANTAGE =====
    validFitness = fitness(isfinite(fitness));
    if isempty(validFitness)
        eliteGap = 0;
    else
        fitnessMedian = median(validFitness);
        eliteGap = (fitnessMedian - min(validFitness)) / (abs(fitnessMedian) + epsVal);
    end
    basicFeatures(3) = tanh(4.0 * eliteGap);

    % ===== 4. STAGNATION DURATION =====
    stagnationSpan = max(5, 0.25 * maxIterations);
    basicFeatures(4) = min(1.0, max(0.0, (iter - lastImprovementIteration) / stagnationSpan));

    % ===== 5. FEASIBILITY-QUALITY READINESS =====
    feasibleRatio = 0;
    if isfield(particles, 'isFeasible') && ~isempty(particles.isFeasible)
        feasibleRatio = mean(double(particles.isFeasible(:)));
    end
    qualityScore = 0;
    if globalBestFitness > 0 && isfinite(globalBestFitness)
        qualityScore = 1.0 / (1.0 + log1p(globalBestFitness));
    end
    basicFeatures(5) = 0.55 * feasibleRatio + 0.45 * qualityScore;

    % ===== 6. CONSTRAINT PRESSURE =====
    velocityMagnitudes = sqrt(sum(velocities.^2, 2));
    collisionRatio = 0;
    terrainPressure = 0;
    dangerPressure = 0;
    duplicatePressure = 0;
    if isfield(particles, 'collisionPenalty') && ~isempty(particles.collisionPenalty)
        collisionPenalty = particles.collisionPenalty(:);
        collisionRatio = mean(~isfinite(collisionPenalty) | collisionPenalty > 0);
    end
    if isfield(particles, 'terrainPenalty') && ~isempty(particles.terrainPenalty)
        terrainPenalty = particles.terrainPenalty(:);
        terrainPenalty(~isfinite(terrainPenalty)) = 1e4;
        terrainPressure = log1p(mean(max(terrainPenalty, 0))) / 10;
    end
    if isfield(particles, 'dangerZonePenalty') && ~isempty(particles.dangerZonePenalty)
        dangerPenalty = particles.dangerZonePenalty(:);
        dangerPenalty(~isfinite(dangerPenalty)) = 1e4;
        dangerPressure = log1p(mean(max(dangerPenalty, 0))) / 10;
    end
    if isfield(particles, 'duplicatePenalty') && ~isempty(particles.duplicatePenalty)
        duplicatePenalty = particles.duplicatePenalty(:);
        duplicatePenalty(~isfinite(duplicatePenalty)) = 1e4;
        duplicatePressure = log1p(mean(max(duplicatePenalty, 0))) / 6;
    end
    basicFeatures(6) = min(1.0, collisionRatio + terrainPressure + dangerPressure + duplicatePressure);

    % Resolve actual global-best particle, not particle 1.
    bestFitnessVector = particles.bestFitness(:);
    validBestMask = isfinite(bestFitnessVector);
    if any(validBestMask)
        validIdx = find(validBestMask);
        [~, bestRelIdx] = min(bestFitnessVector(validBestMask));
        bestIdx = validIdx(bestRelIdx);
        globalBestPosition = particles.bestPositions(bestIdx, :);
    else
        globalBestPosition = mean(positions, 1);
    end

    % ===== 7. GLOBAL-BEST ATTRACTION =====
    distancesToGbest = sqrt(sum((positions - globalBestPosition).^2, 2));
    meanDistanceToGbest = mean(distancesToGbest);
    basicFeatures(7) = 1.0 - tanh(meanDistanceToGbest / (spanMean * sqrt(max(dims, 1)) + epsVal));

    % ===== 8. VELOCITY ALIGNMENT TO GLOBAL BEST =====
    directionToGbest = globalBestPosition - positions;
    directionNorm = sqrt(sum(directionToGbest.^2, 2));
    validAlignment = (directionNorm > epsVal) & (velocityMagnitudes > epsVal);
    if any(validAlignment)
        cosineAlignment = sum(velocities(validAlignment, :) .* directionToGbest(validAlignment, :), 2) ./ ...
            (velocityMagnitudes(validAlignment) .* directionNorm(validAlignment) + epsVal);
        basicFeatures(8) = 0.5 * (mean(cosineAlignment) + 1.0);
    else
        basicFeatures(8) = 0.5;
    end

    % ===== 9. PARTICLE-LEVEL STAGNATION PRESSURE =====
    if isfield(particles, 'stagnationCounter') && ~isempty(particles.stagnationCounter)
        localStagnationScale = max(4, 0.10 * maxIterations);
        stagnationCounter = particles.stagnationCounter(:);
        basicFeatures(9) = mean(min(stagnationCounter / localStagnationScale, 1.0));
    else
        basicFeatures(9) = basicFeatures(4);
    end

    % Clip all features to a compact range for stable critic conditioning.
    basicFeatures = max(0, min(1, basicFeatures));
end
