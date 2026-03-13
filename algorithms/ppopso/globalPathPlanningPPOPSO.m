function [bestPath, bestFitness, fitnessHistory, agent, stateTracker, parameterHistory, learningStats] = globalPathPlanningPPOPSO( ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config)
    % PPOPSO global path planning aligned to the PPOPSO paper.

    fprintf('\n============================================\n');
    fprintf('PPO-PSO Global Path Planning\n');
    fprintf('============================================\n\n');

    if config.numSubgroups > config.popSize
        config.numSubgroups = config.popSize;
        config.stateSize = 1 + config.historyLen * (1 + config.numSubgroups);
    end
    if isempty(config.maxFunctionEvals)
        config.maxFunctionEvals = config.popSize * config.maxIterations;
    end

    agent = PPOPSO_Agent(config);

    % Initialize PSO
    [particles, mapSize, numWaypoints] = initializePSOParticles(config, startPoint, goalPoint, ...
        terrainGrid, terrainX, terrainY);

    % State history (most recent at index 1)
    historyGlobalBest = ones(config.historyLen, 1) * config.historyInitValue;
    historyAvgPbest = ones(config.historyLen, config.numSubgroups) * config.historyInitValue;

    % History trackers
    fitnessHistory = zeros(1, config.maxIterations);
    parameterHistory_w = zeros(1, config.maxIterations);
    parameterHistory_c1 = zeros(1, config.maxIterations);
    parameterHistory_c2 = zeros(1, config.maxIterations);
    parameterHistory_w_samples = zeros(config.maxIterations, config.popSize);
    parameterHistory_c1_samples = zeros(config.maxIterations, config.popSize);
    parameterHistory_c2_samples = zeros(config.maxIterations, config.popSize);
    rewardHistory = zeros(1, config.maxIterations);
    criticLossHistory = nan(1, config.maxIterations);

    % Best solution tracking
    globalBestFitness = inf;
    globalBestPosition = [];
    previousGlobalBest = config.historyInitValue;
    functionEvalCount = 0;

    % Rollout buffer
    bufferStates = zeros(config.stateSize, config.updateInterval);
    bufferActions = zeros(config.numSubgroups, config.updateInterval);
    bufferLogProbs = zeros(1, config.updateInterval);
    bufferValues = zeros(1, config.updateInterval);
    bufferRewards = zeros(1, config.updateInterval);
    bufferDones = zeros(1, config.updateInterval);
    bufferCount = 0;
    lastNextState = [];

    for iter = 1:config.maxIterations
        % Build state from history
        feNorm = min(functionEvalCount / config.maxFunctionEvals, 1.0);
        state = buildStateVector(feNorm, historyGlobalBest, historyAvgPbest, config);

        % PPO action (per subgroup)
        [actionIdx, logProb, value] = agent.getAction(state);
        paramsPerSubgroup = mapActionsToParams(actionIdx, iter, config);
        particleParams = expandParamsToParticles(paramsPerSubgroup, particles.subgroupIds);

        % Update PSO
        [particles, iterBestFitness, iterBestPosition] = updatePSOParticles( ...
            particles, particleParams, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, ...
            mapSize, numWaypoints, config);

        % Global best tracking
        if iterBestFitness < globalBestFitness
            globalBestFitness = iterBestFitness;
            globalBestPosition = iterBestPosition;
        end

        % Reward from global best improvement
        if previousGlobalBest - globalBestFitness > 0
            reward = 1.0;
        else
            reward = -0.1;
        end
        rewardHistory(iter) = reward;
        previousGlobalBest = globalBestFitness;

        % Update state history for next step
        avgPbest = computeAveragePbest(particles, config.numSubgroups);
        [historyGlobalBest, historyAvgPbest] = pushHistory( ...
            historyGlobalBest, historyAvgPbest, globalBestFitness, avgPbest);

        % Next state
        nextFeNorm = min((functionEvalCount + config.popSize) / config.maxFunctionEvals, 1.0);
        nextState = buildStateVector(nextFeNorm, historyGlobalBest, historyAvgPbest, config);

        % Buffer transition
        bufferCount = bufferCount + 1;
        bufferStates(:, bufferCount) = state;
        bufferActions(:, bufferCount) = actionIdx;
        bufferLogProbs(bufferCount) = logProb;
        bufferValues(bufferCount) = value;
        bufferRewards(bufferCount) = reward;
        bufferDones(bufferCount) = double(iter == config.maxIterations);
        lastNextState = nextState;

        % PPO update
        if bufferCount == config.updateInterval || iter == config.maxIterations
            updateCriticLoss = NaN;
            if bufferCount > 0
                if bufferDones(bufferCount) == 1
                    lastValue = 0;
                else
                    lastValue = agent.getValue(lastNextState);
                end
                [returns, advantages] = computeGAE(bufferRewards(1:bufferCount), ...
                    bufferValues(1:bufferCount), lastValue, bufferDones(1:bufferCount), ...
                    config.gamma, config.gaeLambda);

                advantages = advantages - mean(advantages);
                advStd = std(advantages);
                if advStd > 1e-6
                    advantages = advantages / advStd;
                end

                losses = agent.train(bufferStates(:, 1:bufferCount), bufferActions(:, 1:bufferCount), ...
                    bufferLogProbs(1:bufferCount), returns, advantages);
                updateCriticLoss = losses.critic;
            end
            criticLossHistory(iter) = updateCriticLoss;
            bufferCount = 0;
        end

        % Migration (ring topology)
        if mod(iter, config.migrationInterval) == 0 && config.numSubgroups > 1
            particles = performMigration(particles, config, mapSize);
        end

        functionEvalCount = functionEvalCount + config.popSize;

        % Store histories
        fitnessHistory(iter) = globalBestFitness;
        parameterHistory_w(iter) = mean(particleParams(:, 1));
        parameterHistory_c1(iter) = mean(particleParams(:, 2));
        parameterHistory_c2(iter) = mean(particleParams(:, 3));
        parameterHistory_w_samples(iter, :) = particleParams(:, 1)';
        parameterHistory_c1_samples(iter, :) = particleParams(:, 2)';
        parameterHistory_c2_samples(iter, :) = particleParams(:, 3)';

        if mod(iter, config.logInterval) == 0 || iter == config.maxIterations
            fprintf('  [Iter %3d/%d] Best: %.4f\n', iter, config.maxIterations, globalBestFitness);
        end
    end

    bestFitness = globalBestFitness;
    bestPath = constructPathFromPSO(globalBestPosition, startPoint, goalPoint, numWaypoints, config.mapSize);

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

    stateTracker = struct();
    stateTracker.historyGlobalBest = historyGlobalBest;
    stateTracker.historyAvgPbest = historyAvgPbest;
    stateTracker.stateSize = config.stateSize;
end

function state = buildStateVector(feNorm, historyGlobalBest, historyAvgPbest, config)
    state = zeros(config.stateSize, 1);
    state(1) = feNorm;
    idx = 2;
    for k = 1:config.historyLen
        state(idx) = historyGlobalBest(k);
        idx = idx + 1;
        state(idx:idx+config.numSubgroups-1) = historyAvgPbest(k, :)';
        idx = idx + config.numSubgroups;
    end
end

function [historyGlobalBest, historyAvgPbest] = pushHistory(historyGlobalBest, historyAvgPbest, ...
    globalBest, avgPbest)
    historyGlobalBest = [globalBest; historyGlobalBest(1:end-1)];
    historyAvgPbest = [avgPbest; historyAvgPbest(1:end-1, :)];
end

function avgPbest = computeAveragePbest(particles, numSubgroups)
    avgPbest = zeros(1, numSubgroups);
    for j = 1:numSubgroups
        idx = particles.subgroups{j};
        avgPbest(j) = mean(particles.bestFitness(idx));
    end
end

function paramsPerSubgroup = mapActionsToParams(actionIdx, iter, config)
    numSubgroups = config.numSubgroups;
    paramsPerSubgroup = zeros(numSubgroups, 3);

    for j = 1:numSubgroups
        action = actionIdx(j);
        params = config.actionTable(action, :);
        w = params(1);
        c1 = params(2);
        c2 = params(3);

        if action == config.linearWActionIndex
            if config.maxIterations > 1
                ratio = (iter - 1) / (config.maxIterations - 1);
            else
                ratio = 1;
            end
            w = config.linearWStart + (config.linearWEnd - config.linearWStart) * ratio;
        end

        paramsPerSubgroup(j, :) = [w, c1, c2];
    end
end

function particleParams = expandParamsToParticles(paramsPerSubgroup, subgroupIds)
    particleParams = paramsPerSubgroup(subgroupIds, :);
end

function [returns, advantages] = computeGAE(rewards, values, lastValue, dones, gamma, lambda)
    T = numel(rewards);
    advantages = zeros(1, T);
    returns = zeros(1, T);
    gae = 0;
    for t = T:-1:1
        if t == T
            nextValue = lastValue;
        else
            nextValue = values(t + 1);
        end
        delta = rewards(t) + gamma * nextValue * (1 - dones(t)) - values(t);
        gae = delta + gamma * lambda * (1 - dones(t)) * gae;
        advantages(t) = gae;
        returns(t) = advantages(t) + values(t);
    end
end

function [particles, mapSize, numWaypoints] = initializePSOParticles(config, startPoint, goalPoint, ...
    terrainGrid, terrainX, terrainY)
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

    [subgroupIds, subgroups] = assignSubgroups(popSize, config.numSubgroups);
    particles.subgroupIds = subgroupIds;
    particles.subgroups = subgroups;

    maxVelPerDim = config.velocityClampFactor * mapSize;

    for i = 1:popSize
        for j = 1:numWaypoints
            idx = (j-1) * 3 + 1;
            x = rand() * mapSize(1);
            y = rand() * mapSize(2);
            [~, xIndex] = min(abs(terrainX(1,:) - x));
            [~, yIndex] = min(abs(terrainY(:,1) - y));
            terrainHeight = terrainGrid(yIndex, xIndex);
            z = terrainHeight + 10 + rand() * 20;

            particles.cartesianPositions(i, idx:idx+2) = [x, y, z];
            particles.velocities(i, idx:idx+2) = (rand(1, 3) * 2 - 1) .* maxVelPerDim;
        end
        particles.bestPositions(i, :) = particles.cartesianPositions(i, :);
    end
end

function [subgroupIds, subgroups] = assignSubgroups(popSize, numSubgroups)
    subgroupIds = zeros(popSize, 1);
    subgroups = cell(1, numSubgroups);
    counts = floor(popSize / numSubgroups) * ones(1, numSubgroups);
    remainder = popSize - sum(counts);
    for j = 1:remainder
        counts(j) = counts(j) + 1;
    end
    current = 1;
    for j = 1:numSubgroups
        idx = current:(current + counts(j) - 1);
        subgroupIds(idx) = j;
        subgroups{j} = idx;
        current = current + counts(j);
    end
end

function [particles, bestFitness, bestPosition] = updatePSOParticles(particles, particleParams, ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, numWaypoints, config)
    [popSize, dims] = size(particles.cartesianPositions);

    [subgroupBestPositions, subgroupBestFitness] = computeSubgroupBest(particles, config.numSubgroups);
    bestFitness = min(subgroupBestFitness);
    bestPosition = subgroupBestPositions(find(subgroupBestFitness == bestFitness, 1, 'first'), :);

    maxVelPerDim = config.velocityClampFactor * mapSize;

    for i = 1:popSize
        w = particleParams(i, 1);
        c1 = particleParams(i, 2);
        c2 = particleParams(i, 3);

        r1 = rand(1, dims);
        r2 = rand(1, dims);

        cognitiveComponent = c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));
        subgroupId = particles.subgroupIds(i);
        socialTarget = subgroupBestPositions(subgroupId, :);
        socialComponent = c2 * r2 .* (socialTarget - particles.cartesianPositions(i,:));

        particles.velocities(i,:) = w * particles.velocities(i,:) + cognitiveComponent + socialComponent;

        for j = 1:numWaypoints
            idx = (j-1) * 3 + 1;
            segmentVel = particles.velocities(i, idx:idx+2);
            segmentVel = max(min(segmentVel, maxVelPerDim), -maxVelPerDim);
            particles.velocities(i, idx:idx+2) = segmentVel;
        end

        particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);

        for j = 1:numWaypoints
            idx = (j-1) * 3 + 1;
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
        end
    end

    [subgroupBestPositions, subgroupBestFitness] = computeSubgroupBest(particles, config.numSubgroups);
    bestFitness = min(subgroupBestFitness);
    bestPosition = subgroupBestPositions(find(subgroupBestFitness == bestFitness, 1, 'first'), :);
end

function [subgroupBestPositions, subgroupBestFitness] = computeSubgroupBest(particles, numSubgroups)
    dims = size(particles.bestPositions, 2);
    subgroupBestPositions = zeros(numSubgroups, dims);
    subgroupBestFitness = zeros(numSubgroups, 1);
    for j = 1:numSubgroups
        idx = particles.subgroups{j};
        [subBest, localIdx] = min(particles.bestFitness(idx));
        subgroupBestFitness(j) = subBest;
        subgroupBestPositions(j, :) = particles.bestPositions(idx(localIdx), :);
    end
end

function particles = performMigration(particles, config, mapSize)
    numSubgroups = config.numSubgroups;
    maxVelPerDim = config.velocityClampFactor * mapSize;

    for j = 1:numSubgroups
        sourceIdx = particles.subgroups{j};
        targetGroup = mod(j, numSubgroups) + 1;
        targetIdx = particles.subgroups{targetGroup};

        numMigrants = max(1, floor(config.migrationRate * numel(sourceIdx)));
        numMigrants = min(numMigrants, numel(targetIdx));

        [~, sourceOrder] = sort(particles.bestFitness(sourceIdx), 'ascend');
        [~, targetOrder] = sort(particles.bestFitness(targetIdx), 'descend');

        sourceElite = sourceIdx(sourceOrder(1:numMigrants));
        targetWorst = targetIdx(targetOrder(1:numMigrants));

        for k = 1:numMigrants
            sIdx = sourceElite(k);
            tIdx = targetWorst(k);
            particles.cartesianPositions(tIdx, :) = particles.bestPositions(sIdx, :);
            particles.bestPositions(tIdx, :) = particles.bestPositions(sIdx, :);
            particles.bestFitness(tIdx) = particles.bestFitness(sIdx);
            particles.fitness(tIdx) = particles.bestFitness(sIdx);

            for wpt = 1:config.numWaypoints
                idx = (wpt-1) * 3 + 1;
                particles.velocities(tIdx, idx:idx+2) = (rand(1, 3) * 2 - 1) .* maxVelPerDim;
            end
        end
    end
end
