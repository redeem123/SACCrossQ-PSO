function [globalPath, convergenceHistory, algorithmSpecificStats] = globalPathPlanningMPSORL( ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, ...
    popSize, maxIterations, wMaxInput, c1MaxInput, c2MaxInput, alpha, gamma, epsilon, learningPeriod, pop1Ratio)
% MPSORL: Multi-Strategy Self-Learning PSO based on Q-learning
% Reference: Meng et al., MBE 2023

    disp('Starting MPSORL (Multi-Strategy Self-Learning PSO) path planning...');

    % Paper-consistent defaults
    if nargin < 15 || isempty(alpha)
        alpha = 0.6;
    end
    if nargin < 16 || isempty(gamma)
        gamma = 0.8;
    end
    if nargin < 17 || isempty(epsilon)
        epsilon = 0.8;
    end
    if nargin < 18 || isempty(learningPeriod)
        learningPeriod = 50;
    end
    if nargin < 19 || isempty(pop1Ratio)
        pop1Ratio = 0.4;
    end

    wMax = fallbackValue(wMaxInput, 0.9);
    c1Max = fallbackValue(c1MaxInput, 2.5);
    c2Max = fallbackValue(c2MaxInput, 2.5);

    wMin = 0.2;
    c1Min = 0.5;
    c2Min = 0.5;
    cMax = 3.0;
    cMin = 1.5;

    chi = 0.7298;   % LIPS constriction factor
    nsize = 3;      % LIPS/UPSO neighborhood size
    phi = 0.5;      % UPSO unification factor

    numWaypoints = 5;
    dims = 3 * numWaypoints;

    convergenceHistory = [];
    rewardHistory = nan(maxIterations, 1);
    parameterHistory = struct( ...
        'w', nan(1, maxIterations), ...
        'c1', nan(1, maxIterations), ...
        'c2', nan(1, maxIterations));

    % Pop1/Pop2 split
    N1 = max(1, round(popSize * pop1Ratio));
    N1 = min(popSize - 1, N1);
    pop1Idx = 1:N1;
    pop2Idx = (N1 + 1):popSize;

    % Initialize particles
    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);

    globalBestPosition = zeros(1, dims);
    globalBestFitness = Inf;
    globalBestComponents = struct();
    globalBestComponents.pathLength = 0;
    globalBestComponents.collisionPenalty = 0;
    globalBestComponents.turningPenalty = 0;
    globalBestComponents.climbingPenalty = 0;
    globalBestComponents.heightPenalty = 0;
    globalBestComponents.terrainPenalty = 0;
    globalBestComponents.obstaclePenalty = 0;

    % Q-learning setup for pop2
    numStates = 5;
    numActions = 4;
    Q = zeros(numStates, numActions);

    % CLPSO exemplar tracking
    learningExemplar = zeros(popSize, dims);
    lastUpdateIteration = zeros(popSize, 1);
    Pc = computeClpsoLearningProb(popSize);
    clpso_m = 7;  % refreshing gap

    % Initialize positions and velocities
    randMatrix = rand(popSize, dims);
    xIndices = 1:3:dims;
    yIndices = 2:3:dims;
    zIndices = 3:3:dims;

    particles.cartesianPositions(:, xIndices) = randMatrix(:, xIndices) * mapSize(1);
    particles.cartesianPositions(:, yIndices) = randMatrix(:, yIndices) * mapSize(2);
    particles.cartesianPositions(:, zIndices) = 10 + randMatrix(:, zIndices) * (mapSize(3) - 10);
    particles.velocities = (rand(popSize, dims) - 0.5) * 10;

    vmaxVec = buildVelocityClamp(mapSize, numWaypoints);

    % Initial fitness evaluation
    for i = 1:popSize
        [particles.fitness(i), fitnessComponents] = evaluatePathFitness( ...
            particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, ...
            terrainGrid, terrainX, terrainY, numWaypoints);
        particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
        particles.bestFitness(i) = particles.fitness(i);

        if particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestPosition = particles.cartesianPositions(i,:);
            globalBestComponents = fitnessComponents;
        end
    end

    % Initialize CLPSO exemplars
    learningExemplar = initializeClpsoExemplars(particles.bestFitness, Pc, popSize, dims);

    convergenceHistory = [convergenceHistory; globalBestFitness];

    % Initial state/action for pop2
    stateIdxPop2 = assignStatesNonUniform(particles.fitness(pop2Idx));
    actionIdxPop2 = selectActions(Q, stateIdxPop2, epsilon, numActions);

    % Main MPSORL loop
    for iter = 1:maxIterations
        w_current = wMax - (wMax - wMin) * iter / maxIterations;
        c1_current = c1Max - (c1Max - c1Min) * iter / maxIterations;
        c2_current = c2Min + (c2Max - c2Min) * iter / maxIterations;
        c_current = cMax - (cMax - cMin) * iter / maxIterations;

        parameterHistory.w(iter) = w_current;
        parameterHistory.c1(iter) = c1_current;
        parameterHistory.c2(iter) = c2_current;

        prevStateIdxPop2 = stateIdxPop2;

        if mod(iter - 1, learningPeriod) == 0
            actionIdxPop2 = selectActions(Q, prevStateIdxPop2, epsilon, numActions);
        end

        % pop1: CLPSO update
        for idx = 1:numel(pop1Idx)
            i = pop1Idx(idx);
            [particles.velocities(i,:), learningExemplar, lastUpdateIteration] = updateClpso( ...
                particles, i, w_current, c_current, Pc, clpso_m, learningExemplar, lastUpdateIteration, popSize, dims, iter);

            particles.velocities(i,:) = clampVelocityVector(particles.velocities(i,:), vmaxVec);
            particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);
            particles.cartesianPositions(i,:) = clampPosition(particles.cartesianPositions(i,:), mapSize, numWaypoints);

            [particles.fitness(i), fitnessComponents] = evaluatePathFitness( ...
                particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, ...
                terrainGrid, terrainX, terrainY, numWaypoints);

            if particles.fitness(i) < particles.bestFitness(i)
                particles.bestFitness(i) = particles.fitness(i);
                particles.bestPositions(i,:) = particles.cartesianPositions(i,:);

                if particles.fitness(i) < globalBestFitness
                    globalBestFitness = particles.fitness(i);
                    globalBestPosition = particles.cartesianPositions(i,:);
                    globalBestComponents = fitnessComponents;
                end
            end
        end

        % pop2: multi-strategy updates
        for idx = 1:numel(pop2Idx)
            i = pop2Idx(idx);
            action = actionIdxPop2(idx);

            switch action
                case 1  % LIPS
                    particles.velocities(i,:) = updateLips( ...
                        particles, i, w_current, c1_current, c2_current, chi, nsize);

                case 2  % UPSO
                    particles.velocities(i,:) = updateUpso( ...
                        particles, i, globalBestPosition, w_current, c1_current, c2_current, phi, nsize);

                case 3  % LDWPSO
                    particles.velocities(i,:) = updateStandardPso( ...
                        particles, i, globalBestPosition, w_current, c1_current, c2_current);

                case 4  % CLPSO
                    [particles.velocities(i,:), learningExemplar, lastUpdateIteration] = updateClpso( ...
                        particles, i, w_current, c_current, Pc, clpso_m, learningExemplar, lastUpdateIteration, popSize, dims, iter);
            end

            particles.velocities(i,:) = clampVelocityVector(particles.velocities(i,:), vmaxVec);
            particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);
            particles.cartesianPositions(i,:) = clampPosition(particles.cartesianPositions(i,:), mapSize, numWaypoints);

            [particles.fitness(i), fitnessComponents] = evaluatePathFitness( ...
                particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, ...
                terrainGrid, terrainX, terrainY, numWaypoints);

            if particles.fitness(i) < particles.bestFitness(i)
                particles.bestFitness(i) = particles.fitness(i);
                particles.bestPositions(i,:) = particles.cartesianPositions(i,:);

                if particles.fitness(i) < globalBestFitness
                    globalBestFitness = particles.fitness(i);
                    globalBestPosition = particles.cartesianPositions(i,:);
                    globalBestComponents = fitnessComponents;
                end
            end
        end

        % Update states for pop2
        stateIdxPop2 = assignStatesNonUniform(particles.fitness(pop2Idx));

        % Q-table update every LP iterations
        if mod(iter, learningPeriod) == 0
            rewards = zeros(numel(pop2Idx), 1);
            for idx = 1:numel(pop2Idx)
                reward = double(stateIdxPop2(idx) < prevStateIdxPop2(idx));
                rewards(idx) = reward;
                s = prevStateIdxPop2(idx);
                a = actionIdxPop2(idx);
                s2 = stateIdxPop2(idx);
                Q(s, a) = (1 - alpha) * Q(s, a) + alpha * (reward + gamma * max(Q(s2, :)));
            end
            rewardHistory(iter) = mean(rewards);
        end

        convergenceHistory = [convergenceHistory; globalBestFitness];

        if mod(iter, 50) == 0
            disp(['MPSORL iteration: ', num2str(iter), '/', num2str(maxIterations), ...
                ' - Best fitness: ', num2str(globalBestFitness)]);
        end
    end

    globalPath = constructPathFromPSO(globalBestPosition, startPoint, goalPoint, numWaypoints, mapSize);

    for i = 1:size(globalPath, 1)
        x = globalPath(i,1);
        y = globalPath(i,2);
        [~, xIndex] = min(abs(terrainX(1,:) - x));
        [~, yIndex] = min(abs(terrainY(:,1) - y));
        minHeight = terrainGrid(yIndex, xIndex) + 8;
        globalPath(i,3) = max(globalPath(i,3), minHeight);
    end

    if ~exist('globalBestComponents', 'var')
        [~, globalBestComponents] = evaluatePathFitness(globalBestPosition, startPoint, goalPoint, ...
            dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
    end

    algorithmSpecificStats.actualBestFitness = globalBestFitness;
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, globalBestFitness, globalBestComponents);
    algorithmSpecificStats.convergenceHistory = convergenceHistory;
    algorithmSpecificStats.rewardHistory = rewardHistory;
    algorithmSpecificStats.parameterHistory = parameterHistory;
    algorithmSpecificStats.qTable = Q;

    disp(['MPSORL completed with final fitness: ', num2str(globalBestFitness)]);
end

function value = fallbackValue(inputValue, defaultValue)
    if isempty(inputValue) || ~isfinite(inputValue)
        value = defaultValue;
    else
        value = inputValue;
    end
end

function vmaxVec = buildVelocityClamp(mapSize, numWaypoints)
    rangeZ = max(1, mapSize(3) - 10);
    base = [mapSize(1), mapSize(2), rangeZ];
    vmaxVec = 0.5 * repmat(base, 1, numWaypoints);
end

function Pc = computeClpsoLearningProb(popSize)
    if popSize <= 1
        Pc = 0.5;
        return;
    end
    pc_indices = 5.0 * (0:popSize-1) / (popSize-1);
    Pc = 0.5 * (exp(pc_indices) - exp(0)) / (exp(5.0) - exp(0));
    Pc = Pc(:);
end

function learningExemplar = initializeClpsoExemplars(bestFitness, Pc, popSize, dims)
    learningExemplar = zeros(popSize, dims);
    for i = 1:popSize
        for d = 1:dims
            if rand() < Pc(i)
                idx1 = randi(popSize);
                idx2 = randi(popSize);
                if bestFitness(idx1) < bestFitness(idx2)
                    learningExemplar(i, d) = idx1;
                else
                    learningExemplar(i, d) = idx2;
                end
            else
                learningExemplar(i, d) = i;
            end
        end
    end
end

function actions = selectActions(Q, stateIdx, epsilon, numActions)
    popSize = numel(stateIdx);
    actions = zeros(popSize, 1);
    for i = 1:popSize
        if rand() < epsilon
            actions(i) = randi(numActions);
        else
            qRow = Q(stateIdx(i), :);
            maxQ = max(qRow);
            bestActions = find(qRow == maxQ);
            actions(i) = bestActions(randi(numel(bestActions)));
        end
    end
end

function states = assignStatesNonUniform(fitness)
    popSize = numel(fitness);
    states = zeros(popSize, 1);

    if popSize < 5
        states(:) = 3;
        return;
    end

    [~, sortedIdx] = sort(fitness, 'ascend');
    ratios = [0.1, 0.2, 0.4, 0.2, 0.1];
    counts = round(popSize * ratios);
    diffCount = popSize - sum(counts);
    counts(3) = counts(3) + diffCount;

    if all(counts <= 0)
        counts(3) = popSize;
    end

    cursor = 1;
    for s = 1:5
        if cursor > popSize
            break;
        end
        endIdx = min(popSize, cursor + counts(s) - 1);
        if endIdx < cursor
            continue;
        end
        states(sortedIdx(cursor:endIdx)) = s;
        cursor = endIdx + 1;
    end

    if any(states == 0)
        states(states == 0) = 3;
    end
end

function v = updateStandardPso(particles, i, globalBestPosition, w, c1, c2)
    dims = size(particles.cartesianPositions, 2);
    r1 = rand(1, dims);
    r2 = rand(1, dims);
    v = w * particles.velocities(i,:) + ...
        c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) + ...
        c2 * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));
end

function v = updateLips(particles, i, w, c1, c2, chi, nsize)
    dims = size(particles.cartesianPositions, 2);
    popSize = size(particles.cartesianPositions, 1);
    nsize = min(nsize, popSize);

    neighbors = zeros(nsize, 1);
    for k = 1:nsize
        neighbors(k) = mod(i - 1 + k, popSize) + 1;
    end

    [~, bestIdx] = min(particles.bestFitness(neighbors));
    lbest = particles.bestPositions(neighbors(bestIdx), :);

    r1 = rand(1, dims);
    r2 = rand(1, dims);
    v = chi * ( ...
        w * particles.velocities(i,:) + ...
        c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) + ...
        c2 * r2 .* (lbest - particles.cartesianPositions(i,:)) ...
    );
end

function v = updateUpso(particles, i, globalBestPosition, w, c1, c2, phi, nsize)
    dims = size(particles.cartesianPositions, 2);
    popSize = size(particles.cartesianPositions, 1);
    nsize = min(nsize, popSize);

    neighbors = zeros(nsize, 1);
    for k = 1:nsize
        neighbors(k) = mod(i - 1 + k, popSize) + 1;
    end

    [~, bestIdx] = min(particles.bestFitness(neighbors));
    lbest = particles.bestPositions(neighbors(bestIdx), :);

    r1 = rand(1, dims);
    r2 = rand(1, dims);

    v_global = w * particles.velocities(i,:) + ...
        c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) + ...
        c2 * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));

    v_local = w * particles.velocities(i,:) + ...
        c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) + ...
        c2 * r2 .* (lbest - particles.cartesianPositions(i,:));

    v = phi * v_global + (1 - phi) * v_local;
end

function [v, learningExemplar, lastUpdateIteration] = updateClpso( ...
    particles, i, w, c, Pc, m, learningExemplar, lastUpdateIteration, popSize, dims, iter)

    if (iter - lastUpdateIteration(i)) > m
        for d = 1:dims
            if rand() < Pc(i)
                idx1 = randi(popSize);
                idx2 = randi(popSize);
                if particles.bestFitness(idx1) < particles.bestFitness(idx2)
                    learningExemplar(i, d) = idx1;
                else
                    learningExemplar(i, d) = idx2;
                end
            else
                learningExemplar(i, d) = i;
            end
        end
        lastUpdateIteration(i) = iter;
    end

    exemplarPosition = zeros(1, dims);
    for d = 1:dims
        exemplarIdx = learningExemplar(i, d);
        exemplarPosition(d) = particles.bestPositions(exemplarIdx, d);
    end

    r = rand(1, dims);
    v = w * particles.velocities(i,:) + c * r .* (exemplarPosition - particles.cartesianPositions(i,:));
end

function v = clampVelocityVector(v, vmaxVec)
    v = max(-vmaxVec, min(vmaxVec, v));
end

function pos = clampPosition(pos, mapSize, numWaypoints)
    for j = 1:numWaypoints
        idx = (j-1)*3 + 1;
        pos(idx) = max(0, min(pos(idx), mapSize(1)));
        pos(idx+1) = max(0, min(pos(idx+1), mapSize(2)));
        pos(idx+2) = max(10, min(pos(idx+2), mapSize(3)));
    end
end
