function [bestPath, bestFitness, fitnessHistory, agent, stateEncoder, parameterHistory, learningStats] = globalPathPlanningSACSAPSO( ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config)
    % SAC-SAPSO (paper-aligned) global path planning.

    fprintf('\n============================================\n');
    fprintf('SAC-SAPSO (Paper) Global Path Planning\n');
    fprintf('============================================\n\n');

    stateEncoder = [];
    agent = RRSACPSO_Agent(config);

    % Initialize PSO
    [particles, mapSize, numWaypoints] = initializePSOParticles(config, startPoint, goalPoint, ...
        terrainGrid, terrainX, terrainY);

    % Initial evaluation (paper-style)
    globalBestFitness = Inf;
    globalBestPath = [];
    for i = 1:config.popSize
        particles.fitness(i) = evaluatePathFitness(particles.cartesianPositions(i,:), ...
            startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        if isinf(particles.fitness(i))
            particles.fitness(i) = 1e9;
        end
        particles.bestPositions(i, :) = particles.cartesianPositions(i, :);
        particles.bestFitness(i) = particles.fitness(i);

        if particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestPath = particles.cartesianPositions(i, :);
        end
    end

    fitnessHistory = zeros(1, config.maxIterations);
    parameterHistory_w = zeros(1, config.maxIterations);
    parameterHistory_c1 = zeros(1, config.maxIterations);
    parameterHistory_c2 = zeros(1, config.maxIterations);
    parameterHistory_w_samples = zeros(config.maxIterations, config.popSize);
    parameterHistory_c1_samples = zeros(config.maxIterations, config.popSize);
    parameterHistory_c2_samples = zeros(config.maxIterations, config.popSize);
    rewardHistory = nan(1, config.maxIterations);
    criticLossHistory = nan(1, config.maxIterations);

    % Initial observation/action
    initialParams = [0.729844, 1.49618, 1.49618];
    currentState = buildPaperState(particles, 0, config.maxIterations, mapSize, numWaypoints, ...
        initialParams, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY);
    currentAction = agent.getAction(currentState, true);
    particleParams = convertActionToPerParticleParams(currentAction, config);
    lastObsState = currentState;
    lastObsAction = currentAction;
    lastObsBestFitness = globalBestFitness;

    for iter = 1:config.maxIterations
        % Update PSO particles
        [particles, newBestFitness] = updatePSOParticles(particles, particleParams, ...
            startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, ...
            mapSize, numWaypoints);

        if newBestFitness < globalBestFitness
            globalBestFitness = newBestFitness;
            [~, bestIdx] = min(particles.bestFitness);
            globalBestPath = particles.bestPositions(bestIdx, :);
        end

        fitnessHistory(iter) = globalBestFitness;
        parameterHistory_w(iter) = mean(particleParams(:, 1));
        parameterHistory_c1(iter) = mean(particleParams(:, 2));
        parameterHistory_c2(iter) = mean(particleParams(:, 3));
        parameterHistory_w_samples(iter, :) = particleParams(:, 1)';
        parameterHistory_c1_samples(iter, :) = particleParams(:, 2)';
        parameterHistory_c2_samples(iter, :) = particleParams(:, 3)';

        shouldObserve = mod(iter, config.observationInterval) == 0 || iter == config.maxIterations;
        if shouldObserve
            currentParams = [parameterHistory_w(iter), parameterHistory_c1(iter), parameterHistory_c2(iter)];
            nextState = buildPaperState(particles, iter, config.maxIterations, mapSize, numWaypoints, ...
                currentParams, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY);
            reward = calculateRelativeReward(lastObsBestFitness, globalBestFitness);
            done = (iter == config.maxIterations);

            agent.storeTransition(lastObsState, lastObsAction, reward, nextState, done);
            rewardHistory(iter) = reward;

            if iter > config.warmupPeriod
                criticLosses = zeros(1, config.gradientStepsPerTraining);
                for g = 1:config.gradientStepsPerTraining
                    losses = agent.train();
                    criticLosses(g) = losses.critic;
                end
                criticLossHistory(iter) = mean(criticLosses);
            end

            lastObsState = nextState;
            lastObsBestFitness = globalBestFitness;
            lastObsAction = agent.getAction(nextState, true);
            particleParams = convertActionToPerParticleParams(lastObsAction, config);
        end

        if mod(iter, config.logInterval) == 0 || iter == config.maxIterations
            fprintf('  [Iter %3d/%d] Best: %.4f\n', iter, config.maxIterations, globalBestFitness);
        end
    end

    bestFitness = globalBestFitness;
    bestPath = constructPathFromPSO(globalBestPath, startPoint, goalPoint, numWaypoints, config.mapSize);

    parameterHistory = struct();
    parameterHistory.w = parameterHistory_w;
    parameterHistory.c1 = parameterHistory_c1;
    parameterHistory.c2 = parameterHistory_c2;
    parameterHistory.w_samples = parameterHistory_w_samples;
    parameterHistory.c1_samples = parameterHistory_c1_samples;
    parameterHistory.c2_samples = parameterHistory_c2_samples;

    learningStats = struct();
    learningStats.rewardHistory = rewardHistory;
    learningStats.criticLossHistory = criticLossHistory;
end

function [particles, mapSize, numWaypoints] = initializePSOParticles(config, startPoint, goalPoint, ...
    terrainGrid, terrainX, terrainY)
    % Initialize PSO particles for path planning
    numWaypoints = config.numWaypoints;
    mapSize = config.mapSize;
    popSize = config.popSize;
    dims = numWaypoints * 3;

    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.fitness = ones(popSize, 1) * 1e9;
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = ones(popSize, 1) * 1e9;

    for i = 1:popSize
        for j = 1:numWaypoints
            idx = (j-1)*3 + 1;
            x = rand() * mapSize(1);
            y = rand() * mapSize(2);
            [~, xIndex] = min(abs(terrainX(1,:) - x));
            [~, yIndex] = min(abs(terrainY(:,1) - y));
            terrainHeight = terrainGrid(yIndex, xIndex);
            z = terrainHeight + 10 + rand() * 20;

            particles.cartesianPositions(i, idx:idx+2) = [x, y, z];
        end
        particles.bestPositions(i, :) = particles.cartesianPositions(i, :);
    end
end

function [particles, bestFitness] = updatePSOParticles(particles, particleParams, ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, numWaypoints)
    % Update PSO particles using per-particle parameters
    [popSize, dims] = size(particles.cartesianPositions);

    bestFitness = min(particles.bestFitness);

    for i = 1:popSize
        w = particleParams(i, 1);
        c1 = particleParams(i, 2);
        c2 = particleParams(i, 3);

        r1 = rand(1, dims);
        r2 = rand(1, dims);

        cognitiveComponent = c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));
        [~, bestIdx] = min(particles.bestFitness);
        socialComponent = c2 * r2 .* (particles.bestPositions(bestIdx,:) - particles.cartesianPositions(i,:));

        particles.velocities(i,:) = w * particles.velocities(i,:) + cognitiveComponent + socialComponent;

        % Velocity clamping (paper recommends clamping)
        maxVelocity = 0.15 * (mapSize(1) + mapSize(2) + mapSize(3)) / 3;
        for j = 1:numWaypoints
            idx = (j-1)*3 + 1;
            segmentVel = particles.velocities(i, idx:idx+2);
            velMag = norm(segmentVel);

            if isnan(velMag) || isinf(velMag)
                particles.velocities(i, idx:idx+2) = (rand(1,3)-0.5) * maxVelocity * 0.1;
                segmentVel = particles.velocities(i, idx:idx+2);
                velMag = norm(segmentVel);
            end

            if velMag > maxVelocity
                particles.velocities(i, idx:idx+2) = segmentVel * (maxVelocity / velMag);
            end
        end

        particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);

        for j = 1:numWaypoints
            idx = (j-1)*3 + 1;
            waypoint = particles.cartesianPositions(i, idx:idx+2);
            waypoint = max([1, 1, 1], min(waypoint, mapSize));

            [~, xIndex] = min(abs(terrainX(1,:) - waypoint(1)));
            [~, yIndex] = min(abs(terrainY(:,1) - waypoint(2)));
            terrainHeight = terrainGrid(yIndex, xIndex);
            waypoint(3) = max(waypoint(3), terrainHeight + 10);

            particles.cartesianPositions(i, idx:idx+2) = waypoint;
        end

        particles.fitness(i) = evaluatePathFitness(particles.cartesianPositions(i,:), ...
            startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

        if isinf(particles.fitness(i))
            particles.fitness(i) = 1e9;
        end

        if particles.fitness(i) < particles.bestFitness(i)
            particles.bestFitness(i) = particles.fitness(i);
            particles.bestPositions(i,:) = particles.cartesianPositions(i,:);

            if particles.fitness(i) < bestFitness
                bestFitness = particles.fitness(i);
            end
        end
    end
end

function state = buildPaperState(particles, iter, maxIterations, mapSize, numWaypoints, currentParams, ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY)
    % Build observation: per-particle velocity + stability + infeasible + completion.
    popSize = size(particles.cartesianPositions, 1);
    dims = numWaypoints * 3;

    lVec = repmat([1, 1, 1], 1, numWaypoints);
    uVec = repmat(mapSize, 1, numWaypoints);
    scale = 2 ./ (uVec - lVec);
    center = (lVec + uVec) / 2;

    velocityFeatures = zeros(popSize, 1);
    for i = 1:popSize
        v = particles.velocities(i, 1:dims);
        vNorm = tanh(scale .* (v - center));
        velocityFeatures(i) = mean(abs(vNorm));
    end

    stablePct = calculateStablePercentage(currentParams);
    infeasiblePct = calculateInfeasiblePercentage(particles, startPoint, goalPoint, dangerZones, ...
        terrainGrid, terrainX, terrainY, numWaypoints);
    completionPct = iter / maxIterations;

    state = [velocityFeatures; stablePct; infeasiblePct; completionPct];
end

function stablePct = calculateStablePercentage(params)
    if numel(params) < 3
        stablePct = 0;
        return;
    end
    w = params(1);
    c1 = params(2);
    c2 = params(3);

    if w < -1 || w > 1
        stablePct = 0;
        return;
    end

    denom = 7 - 5 * w;
    if denom <= 0
        stablePct = 0;
        return;
    end

    threshold = (24 * (1 - w^2)) / denom;
    stablePct = double((c1 + c2) < threshold);
end

function infeasiblePct = calculateInfeasiblePercentage(particles, startPoint, goalPoint, dangerZones, ...
    terrainGrid, terrainX, terrainY, numWaypoints)
    popSize = size(particles.cartesianPositions, 1);
    infeasibleCount = 0;
    for i = 1:popSize
        isFeasible = isPathFeasible(particles.cartesianPositions(i,:), ...
            startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        if ~isFeasible
            infeasibleCount = infeasibleCount + 1;
        end
    end
    infeasiblePct = infeasibleCount / popSize;
end

function reward = calculateRelativeReward(prevBest, currBest)
    if currBest == prevBest
        reward = 0;
        return;
    end

    beta = abs(prevBest) + abs(currBest);

    if prevBest > 0 && currBest > 0
        denom = prevBest + beta;
        if denom == 0
            reward = 0;
        else
            reward = 2 * (prevBest - currBest) / denom;
        end
        return;
    end

    if prevBest < 0 && currBest < 0
        denom = prevBest + 2 * beta;
        if denom == 0
            reward = 0;
        else
            reward = 2 * (prevBest - currBest) / denom;
        end
        return;
    end

    reward = 1;
end
