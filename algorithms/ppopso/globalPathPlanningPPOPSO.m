function [bestPath, bestFitness, fitnessHistory, agent, stateTracker, parameterHistory, learningStats] = globalPathPlanningPPOPSO( ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config)
    % PPO-PSO global path planning.
    % Adapted from Klein et al. 2024 (iSOMA-RL) to PSO.
    %
    % Continuous PPO controls global (w, c1, c2) every iteration.
    % State: FE completion + history of fitness, fitness diff, actions.
    % Online learning with GAE and clipped PPO objective.

    fprintf('\n============================================\n');
    fprintf('PPO-PSO Global Path Planning (Klein 2024)\n');
    fprintf('============================================\n\n');

    agent = PPOPSO_Agent(config);

    % Initialize PSO
    [particles, mapSize, numWaypoints] = initializePSOParticles(config, startPoint, goalPoint, ...
        terrainGrid, terrainX, terrainY);

    % State history buffers (most recent first)
    hl = config.historyLen;
    histFitness = ones(hl, 1) * 1e9;
    histFitDiff = zeros(hl, 1);
    histActions = zeros(hl, config.actionDim);  % raw (pre-tanh) actions

    % Tracking arrays
    fitnessHistory = zeros(1, config.maxIterations);
    parameterHistory_w = zeros(1, config.maxIterations);
    parameterHistory_c1 = zeros(1, config.maxIterations);
    parameterHistory_c2 = zeros(1, config.maxIterations);
    % Global control: all particles get same params, so samples = scalars
    parameterHistory_w_samples = zeros(config.maxIterations, 1);
    parameterHistory_c1_samples = zeros(config.maxIterations, 1);
    parameterHistory_c2_samples = zeros(config.maxIterations, 1);
    rewardHistory = zeros(1, config.maxIterations);
    criticLossHistory = nan(1, config.maxIterations);

    globalBestFitness = inf;
    globalBestPosition = [];
    previousGlobalBest = 1e9;
    functionEvalCount = 0;

    % Rollout buffer
    bufSize = config.updateInterval;
    bufStates = zeros(config.stateSize, bufSize);
    bufRawActions = zeros(config.actionDim, bufSize);  % raw (pre-tanh)
    bufLogProbs = zeros(1, bufSize);
    bufValues = zeros(1, bufSize);
    bufRewards = zeros(1, bufSize);
    bufDones = zeros(1, bufSize);
    bufCount = 0;
    lastNextState = [];

    for iter = 1:config.maxIterations
        % Build state
        feNorm = min(functionEvalCount / (config.popSize * config.maxIterations), 1.0);
        state = buildState(feNorm, histFitness, histFitDiff, histActions, config);

        % Get continuous action from PPO
        [params, logProb, value] = agent.getAction(state);
        w = params(1); c1 = params(2); c2 = params(3);

        % Recover raw action for storage (inverse tanh of squashed)
        squashed = [2*(w - config.wMin)/(config.wMax - config.wMin) - 1; ...
                    2*(c1 - config.c1Min)/(config.c1Max - config.c1Min) - 1; ...
                    2*(c2 - config.c2Min)/(config.c2Max - config.c2Min) - 1];
        squashed = max(-0.999, min(0.999, squashed));
        rawAction = atanh(squashed);

        % All particles get same params (global control, like the paper)
        particleParams = repmat([w, c1, c2], config.popSize, 1);

        % Update PSO
        [particles, iterBestFitness, iterBestPosition] = updatePSOParticles( ...
            particles, particleParams, startPoint, goalPoint, dangerZones, ...
            terrainGrid, terrainX, terrainY, mapSize, numWaypoints, config);

        % Track global best
        if iterBestFitness < globalBestFitness
            globalBestFitness = iterBestFitness;
            globalBestPosition = iterBestPosition;
        end

        % Reward: relative fitness improvement (same formula as AFSACPSO)
        reward = 2 * (previousGlobalBest - globalBestFitness) / ...
            (abs(previousGlobalBest) + abs(globalBestFitness) + 1e-8);
        rewardHistory(iter) = reward;

        % Update history
        fitDiff = previousGlobalBest - globalBestFitness;
        previousGlobalBest = globalBestFitness;
        histFitness = [globalBestFitness; histFitness(1:end-1)];
        histFitDiff = [fitDiff; histFitDiff(1:end-1)];
        histActions = [rawAction'; histActions(1:end-1, :)];

        % Next state
        nextFeNorm = min((functionEvalCount + config.popSize) / (config.popSize * config.maxIterations), 1.0);
        nextState = buildState(nextFeNorm, histFitness, histFitDiff, histActions, config);

        % Store in rollout buffer
        bufCount = bufCount + 1;
        bufStates(:, bufCount) = state;
        bufRawActions(:, bufCount) = rawAction;
        bufLogProbs(bufCount) = logProb;
        bufValues(bufCount) = value;
        bufRewards(bufCount) = reward;
        bufDones(bufCount) = double(iter == config.maxIterations);
        lastNextState = nextState;

        % PPO update when buffer full
        updateCriticLoss = NaN;
        if bufCount == bufSize || iter == config.maxIterations
            if bufCount > 0
                if bufDones(bufCount) == 1
                    lastValue = 0;
                else
                    lastValue = agent.getValue(lastNextState);
                end
                [returns, advantages] = computeGAE(bufRewards(1:bufCount), ...
                    bufValues(1:bufCount), lastValue, bufDones(1:bufCount), ...
                    config.gamma, config.gaeLambda);

                advantages = advantages - mean(advantages);
                advStd = std(advantages);
                if advStd > 1e-6
                    advantages = advantages / advStd;
                end

                losses = agent.train(bufStates(:, 1:bufCount), bufRawActions(:, 1:bufCount), ...
                    bufLogProbs(1:bufCount), returns, advantages);
                updateCriticLoss = losses.critic;
            end
            criticLossHistory(iter) = updateCriticLoss;
            bufCount = 0;
        end

        functionEvalCount = functionEvalCount + config.popSize;

        % Store histories
        fitnessHistory(iter) = globalBestFitness;
        parameterHistory_w(iter) = w;
        parameterHistory_c1(iter) = c1;
        parameterHistory_c2(iter) = c2;
        parameterHistory_w_samples(iter, :) = w;
        parameterHistory_c1_samples(iter, :) = c1;
        parameterHistory_c2_samples(iter, :) = c2;

        if mod(iter, config.logInterval) == 0 || iter == config.maxIterations
            fprintf('  [Iter %3d/%d] Best: %.4f  w=%.3f c1=%.3f c2=%.3f\n', ...
                iter, config.maxIterations, globalBestFitness, w, c1, c2);
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
    stateTracker.stateSize = config.stateSize;
end

%% State builder (adapted from Klein 2024 Table 5)
function state = buildState(feNorm, histFitness, histFitDiff, histActions, config)
    % Normalize fitness values to prevent explosion
    fNorm = histFitness / (abs(histFitness(1)) + 1e-8);
    dNorm = histFitDiff / (abs(histFitness(1)) + 1e-8);

    state = zeros(config.stateSize, 1);
    state(1) = feNorm;
    idx = 2;
    for k = 1:config.historyLen
        state(idx) = fNorm(k);                          idx = idx + 1;
        state(idx) = dNorm(k);                          idx = idx + 1;
        state(idx:idx+config.actionDim-1) = histActions(k,:)'; idx = idx + config.actionDim;
    end
end

%% GAE computation
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

%% PSO initialization
function [particles, mapSize, numWaypoints] = initializePSOParticles(config, startPoint, goalPoint, ...
    terrainGrid, terrainX, terrainY)
    numWaypoints = config.numWaypoints;
    mapSize = config.mapSize;
    popSize = config.popSize;
    dims = numWaypoints * 3;

    particles = struct();
    particles.cartesianPositions = generateRandomPSOPositions( ...
        popSize, numWaypoints, mapSize, terrainGrid, terrainX, terrainY);
    particles.velocities = zeros(popSize, dims);
    particles.fitness = ones(popSize, 1) * 1e9;
    particles.bestPositions = particles.cartesianPositions;
    particles.bestFitness = ones(popSize, 1) * 1e9;
end

%% PSO update (global topology, per-dimension velocity clamping)
function [particles, bestFitness, bestPosition] = updatePSOParticles(particles, particleParams, ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, numWaypoints, config)
    [popSize, dims] = size(particles.cartesianPositions);

    [bestFitness, gbestIdx] = min(particles.bestFitness);
    gbestPos = particles.bestPositions(gbestIdx, :);
    bestPosition = gbestPos;

    vMaxDim = config.velocityClampDelta * (mapSize - 1);

    % Precompute terrain grid spacing for O(1) index lookup
    txVec = terrainX(1,:);
    tyVec = terrainY(:,1);
    txMin = txVec(1); txStep = txVec(2) - txVec(1); txN = numel(txVec);
    tyMin = tyVec(1); tyStep = tyVec(2) - tyVec(1); tyN = numel(tyVec);

    % Pre-tile velocity limits for vectorized clamping
    vMaxLo = repmat(-vMaxDim, 1, numWaypoints);
    vMaxHi = repmat( vMaxDim, 1, numWaypoints);

    for i = 1:popSize
        w = particleParams(i, 1);
        c1 = particleParams(i, 2);
        c2 = particleParams(i, 3);

        r1 = rand(1, dims);
        r2 = rand(1, dims);

        particles.velocities(i,:) = w * particles.velocities(i,:) ...
            + c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) ...
            + c2 * r2 .* (gbestPos - particles.cartesianPositions(i,:));

        % Per-dimension velocity clamping (vectorized)
        vi = particles.velocities(i,:);
        badMask = isnan(vi) | isinf(vi);
        if any(badMask)
            vi(badMask) = (rand(1, sum(badMask))-0.5) .* vMaxHi(badMask) * 0.1;
        end
        vi = max(vMaxLo, min(vMaxHi, vi));
        particles.velocities(i,:) = vi;

        particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);

        % Boundary + terrain constraints
        for j = 1:numWaypoints
            idx = (j-1)*3 + 1;
            waypoint = particles.cartesianPositions(i, idx:idx+2);
            waypoint = max([1, 1, 1], min(waypoint, mapSize));
            % O(1) terrain index lookup (regular grid)
            xIndex = max(1, min(txN, round((waypoint(1) - txMin) / txStep) + 1));
            yIndex = max(1, min(tyN, round((waypoint(2) - tyMin) / tyStep) + 1));
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
                bestPosition = particles.cartesianPositions(i,:);
            end
        end
    end
end
