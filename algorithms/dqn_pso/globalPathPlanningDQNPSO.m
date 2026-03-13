function [globalPath, convergenceHistory, algorithmSpecificStats] = globalPathPlanningDQNPSO( ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, ...
    popSize, maxIterations, w_initial, c1_initial, c2_initial, paramMode, pretrainedModel)
% DQN-PSO (Deep Q-Network-Enhanced Self-Tuning Control of PSO)
% Reference: Aoun 2024, Modelling 5, 1709-1728.

    if nargin < 14 || isempty(paramMode)
        paramMode = 'global';
    end
    if nargin < 15
        pretrainedModel = [];
    end

    if ~strcmp(paramMode, 'global')
        warning('DQN-PSO (paper) is homogeneous; forcing paramMode to global.');
        paramMode = 'global';
    end

    config = getDQNPSOConfig(w_initial, c1_initial, c2_initial);

    disp('Starting DQN-PSO (HMM + DQN) - Mode: global');

    numWaypoints = 5;
    dims = 3 * numWaypoints;
    convergenceHistory = zeros(maxIterations, 1);
    rewardHistory = zeros(maxIterations, 1);
    criticLossHistory = nan(maxIterations, 1);

    % Initialize particles
    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);

    % Initialize DQN agent
    dqn = initializeDQNAgent(config, pretrainedModel);

    % Initialize positions/velocities
    randMatrix = rand(popSize, dims);
    xIndices = 1:3:dims;
    yIndices = 2:3:dims;
    zIndices = 3:3:dims;

    particles.cartesianPositions(:, xIndices) = randMatrix(:, xIndices) * mapSize(1);
    particles.cartesianPositions(:, yIndices) = randMatrix(:, yIndices) * mapSize(2);
    particles.cartesianPositions(:, zIndices) = 10 + randMatrix(:, zIndices) * (mapSize(3) - 10);
    particles.velocities = (rand(popSize, dims) - 0.5) * 10;

    % Initial fitness evaluation
    [particles.fitness(1), fitnessComponents] = evaluatePathFitness( ...
        particles.cartesianPositions(1,:), startPoint, goalPoint, dangerZones, ...
        terrainGrid, terrainX, terrainY, numWaypoints);
    particles.bestPositions(1,:) = particles.cartesianPositions(1,:);
    particles.bestFitness(1) = particles.fitness(1);

    globalBestFitness = particles.fitness(1);
    globalBestPosition = particles.cartesianPositions(1,:);
    globalBestComponents = fitnessComponents;

    for i = 2:popSize
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

    % Initialize HMM and observation sequence
    hmm = initializeHMM();
    obsSeq = [];
    f = computeEvolutionaryFactor(particles.cartesianPositions, particles.bestFitness);
    obsSeq(end + 1) = discretizeObservation(f);
    if config.useBaumWelch && numel(obsSeq) > 1
        hmm = baumWelchUpdate(hmm, obsSeq, config.hmmUpdateIterations);
    end
    stateIdx = viterbiState(hmm, obsSeq);
    stateVec = oneHotState(stateIdx);

    prevBestFitness = globalBestFitness;
    w = config.w;
    c1 = config.c1;
    c2 = config.c2;

    wHistory = zeros(maxIterations, 1);
    c1History = zeros(maxIterations, 1);
    c2History = zeros(maxIterations, 1);

    for iter = 1:maxIterations
        % DQN action selection
        [action, dqn] = selectDQNAction(dqn, stateVec, config);
        [w, c1, c2] = applyAction(action, w, c1, c2, f, config);

        wHistory(iter) = w;
        c1History(iter) = c1;
        c2History(iter) = c2;

        % PSO update with global parameters
        [particles, globalBestFitness, globalBestPosition, globalBestComponents] = updateParticles( ...
            particles, w, c1, c2, globalBestFitness, globalBestPosition, ...
            startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, ...
            mapSize, numWaypoints, globalBestComponents);

        rawReward = prevBestFitness - globalBestFitness;
        reward = rawReward;
        if config.normalizeReward
            denom = max(abs(prevBestFitness), 1.0);
            reward = rawReward / denom;
            reward = max(-1.0, min(1.0, reward));
        end
        prevBestFitness = globalBestFitness;
        rewardHistory(iter) = reward;

        % Next observation/state
        f = computeEvolutionaryFactor(particles.cartesianPositions, particles.bestFitness);
        obsSeq(end + 1) = discretizeObservation(f);
        if config.useBaumWelch
            hmm = baumWelchUpdate(hmm, obsSeq, config.hmmUpdateIterations);
        end
        nextStateIdx = viterbiState(hmm, obsSeq);
        nextStateVec = oneHotState(nextStateIdx);

        done = (iter == maxIterations);
        dqn = storeTransition(dqn, stateVec, action, reward, nextStateVec, done);
        [dqn, avgLoss] = trainDQN(dqn, config);
        criticLossHistory(iter) = avgLoss;
        if mod(dqn.stepCount, config.targetUpdateInterval) == 0
            dqn.targetNet = dqn.qNet;
        end

        stateVec = nextStateVec;
        convergenceHistory(iter) = globalBestFitness;

        if mod(iter, 50) == 0 || iter == maxIterations
            disp(sprintf('DQN-PSO iter %d/%d | Fitness: %.2f | w=%.3f c1=%.3f c2=%.3f', ...
                iter, maxIterations, globalBestFitness, w, c1, c2));
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

    [~, globalBestComponents] = evaluatePathFitness(globalBestPosition, startPoint, goalPoint, ...
        dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

    algorithmSpecificStats.actualBestFitness = globalBestFitness;
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, globalBestFitness, globalBestComponents);
    algorithmSpecificStats.dqnModel = packDQNModel(dqn, config);
    algorithmSpecificStats.hmm = hmm;
    algorithmSpecificStats.parameterHistory = struct('w', wHistory, 'c1', c1History, 'c2', c2History);
    algorithmSpecificStats.rewardHistory = rewardHistory;
    algorithmSpecificStats.criticLossHistory = criticLossHistory;

    disp(['DQN-PSO completed with final fitness: ', num2str(globalBestFitness)]);
end

function config = getDQNPSOConfig(w_initial, c1_initial, c2_initial)
    config = struct();
    config.wMin = 0.1;
    config.wMax = 0.9;
    config.w = fallbackValue(w_initial, config.wMax);
    config.c1 = fallbackValue(c1_initial, 2.0);
    config.c2 = fallbackValue(c2_initial, 2.0);
    config.cMin = 0.5;
    config.cMax = 2.5;
    config.cInc = 0.1;
    config.cIncSmall = 0.05;

    config.gamma = 0.9;
    config.epsilon = 0.2;
    config.learningRate = 1e-3;
    config.batchSize = 32;
    config.memoryCapacity = 1000;
    config.targetUpdateInterval = 10;
    config.dqnTrainSteps = 10;

    config.useBaumWelch = true;
    config.hmmUpdateIterations = 1;
    config.normalizeReward = true;  % Normalize to [-1, 1] for cross-algorithm comparability
end

function value = fallbackValue(inputValue, defaultValue)
    if isempty(inputValue) || ~isfinite(inputValue)
        value = defaultValue;
    else
        value = inputValue;
    end
end

function hmm = initializeHMM()
    hmm = struct();
    hmm.Pi = [1, 0, 0, 0];

    hmm.A = zeros(4, 4);
    hmm.A(1,1) = 0.5; hmm.A(1,2) = 0.5;
    hmm.A(2,2) = 0.5; hmm.A(2,3) = 0.5;
    hmm.A(3,3) = 0.5; hmm.A(3,4) = 0.5;
    hmm.A(4,4) = 0.5; hmm.A(4,1) = 0.5;

    hmm.B = [
        0,   0,   0,   0.5, 0.25, 0.25, 0;
        0,   0.25,0.25,0.5, 0,    0,    0;
        2/3, 1/3, 0,   0,   0,    0,    0;
        0,   0,   0,   0,   0,    1/3,  2/3
    ];
end

function f = computeEvolutionaryFactor(positions, bestFitness)
    popSize = size(positions, 1);
    if popSize <= 1
        f = 0;
        return;
    end

    avgDistances = zeros(popSize, 1);
    for i = 1:popSize
        diffs = positions - positions(i,:);
        dists = sqrt(sum(diffs.^2, 2));
        dists(i) = [];
        avgDistances(i) = mean(dists);
    end

    [~, bestIdx] = min(bestFitness);
    d_gbest = avgDistances(bestIdx);
    d_min = min(avgDistances);
    d_max = max(avgDistances);

    if d_max <= d_min
        f = 0;
    else
        f = (d_gbest - d_min) / (d_max - d_min);
    end
    f = max(0, min(1, f));
end

function obs = discretizeObservation(f)
    if f < 0.2
        obs = 1;
    elseif f < 0.3
        obs = 2;
    elseif f < 0.4
        obs = 3;
    elseif f < 0.6
        obs = 4;
    elseif f < 0.7
        obs = 5;
    elseif f < 0.8
        obs = 6;
    else
        obs = 7;
    end
end

function hmm = baumWelchUpdate(hmm, obsSeq, numIters)
    for iter = 1:numIters
        [Pi, A, B] = baumWelchOnce(hmm.Pi, hmm.A, hmm.B, obsSeq);
        hmm.Pi = Pi;
        hmm.A = A;
        hmm.B = B;
    end
end

function [Pi, A, B] = baumWelchOnce(Pi, A, B, obsSeq)
    N = numel(Pi);
    M = size(B, 2);
    T = numel(obsSeq);

    alpha = zeros(N, T);
    scale = zeros(1, T);
    alpha(:,1) = Pi(:) .* B(:, obsSeq(1));
    scale(1) = sum(alpha(:,1)) + 1e-12;
    alpha(:,1) = alpha(:,1) / scale(1);

    for t = 2:T
        alpha(:,t) = (A' * alpha(:,t-1)) .* B(:, obsSeq(t));
        scale(t) = sum(alpha(:,t)) + 1e-12;
        alpha(:,t) = alpha(:,t) / scale(t);
    end

    beta = zeros(N, T);
    beta(:,T) = 1 / scale(T);
    for t = T-1:-1:1
        beta(:,t) = A * (B(:, obsSeq(t+1)) .* beta(:,t+1));
        beta(:,t) = beta(:,t) / scale(t);
    end

    gamma = alpha .* beta;
    gamma = gamma ./ (sum(gamma, 1) + 1e-12);

    xi = zeros(N, N, T-1);
    for t = 1:T-1
        denom = (alpha(:,t)' * A) .* (B(:, obsSeq(t+1))') * beta(:,t+1);
        denom = denom + 1e-12;
        for i = 1:N
            for j = 1:N
                xi(i,j,t) = alpha(i,t) * A(i,j) * B(j, obsSeq(t+1)) * beta(j,t+1) / denom;
            end
        end
    end

    Pi = gamma(:,1)';
    A = sum(xi, 3) ./ (sum(gamma(:,1:T-1), 2) + 1e-12);

    for j = 1:N
        for k = 1:M
            mask = (obsSeq == k);
            B(j,k) = sum(gamma(j, mask)) / (sum(gamma(j, :)) + 1e-12);
        end
    end

    A = normalizeRows(A);
    B = normalizeRows(B);
end

function mat = normalizeRows(mat)
    rowSums = sum(mat, 2) + 1e-12;
    mat = mat ./ rowSums;
end

function stateIdx = viterbiState(hmm, obsSeq)
    N = numel(hmm.Pi);
    T = numel(obsSeq);

    logPi = log(hmm.Pi + 1e-12);
    logA = log(hmm.A + 1e-12);
    logB = log(hmm.B + 1e-12);

    delta = zeros(N, T);
    psi = zeros(N, T);
    delta(:,1) = logPi(:) + logB(:, obsSeq(1));

    for t = 2:T
        for j = 1:N
            [bestVal, bestIdx] = max(delta(:,t-1) + logA(:,j));
            delta(j,t) = bestVal + logB(j, obsSeq(t));
            psi(j,t) = bestIdx;
        end
    end

    [~, stateIdx] = max(delta(:,T));
end

function vec = oneHotState(stateIdx)
    vec = zeros(4, 1);
    vec(stateIdx) = 1;
end

function [w, c1, c2] = applyAction(action, w, c1, c2, f, config)
    switch action
        case 1
            w = config.wMin + (config.wMax - config.wMin) * rand();
            c1 = c1 + config.cInc;
            c2 = c2 - config.cInc;
        case 2
            w = 1 / (1 + 1.5 * exp(-2.6 * f));
            w = min(config.wMax, max(config.wMin, w));
            c1 = c1 + config.cInc;
            c2 = c2 - config.cIncSmall;
        case 3
            w = config.wMax;
            c1 = c1 + config.cIncSmall;
            c2 = c2 - config.cInc;
        case 4
            w = config.wMin;
            c1 = c1 - config.cInc;
            c2 = c2 + config.cInc;
        otherwise
            w = min(config.wMax, max(config.wMin, w));
    end

    c1 = min(config.cMax, max(config.cMin, c1));
    c2 = min(config.cMax, max(config.cMin, c2));
end

function [particles, bestFitness, bestPosition, bestComponents] = updateParticles( ...
    particles, w, c1, c2, bestFitness, bestPosition, startPoint, goalPoint, dangerZones, ...
    terrainGrid, terrainX, terrainY, mapSize, numWaypoints, bestComponents)
    [popSize, dims] = size(particles.cartesianPositions);

    for i = 1:popSize
        r1 = rand(1, dims);
        r2 = rand(1, dims);

        particles.velocities(i,:) = w * particles.velocities(i,:) + ...
            c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) + ...
            c2 * r2 .* (bestPosition - particles.cartesianPositions(i,:));

        maxVelocity = 20.0;
        for d = 1:dims
            if abs(particles.velocities(i,d)) > maxVelocity
                particles.velocities(i,d) = sign(particles.velocities(i,d)) * maxVelocity;
            end
        end

        particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);

        for j = 1:numWaypoints
            idx = (j-1) * 3 + 1;
            particles.cartesianPositions(i, idx) = max(0, min(particles.cartesianPositions(i, idx), mapSize(1)));
            particles.cartesianPositions(i, idx+1) = max(0, min(particles.cartesianPositions(i, idx+1), mapSize(2)));
            particles.cartesianPositions(i, idx+2) = max(10, min(particles.cartesianPositions(i, idx+2), mapSize(3)));
        end

        [particles.fitness(i), fitnessComponents] = evaluatePathFitness( ...
            particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, ...
            terrainGrid, terrainX, terrainY, numWaypoints);

        if particles.fitness(i) < particles.bestFitness(i)
            particles.bestFitness(i) = particles.fitness(i);
            particles.bestPositions(i,:) = particles.cartesianPositions(i,:);

            if particles.fitness(i) < bestFitness
                bestFitness = particles.fitness(i);
                bestPosition = particles.cartesianPositions(i,:);
                bestComponents = fitnessComponents;
            end
        end
    end
end

function dqn = initializeDQNAgent(config, pretrainedModel)
    dqn = struct();
    dqn.qNet = createDQNNetwork();
    dqn.targetNet = dqn.qNet;
    dqn.optimizer = [];
    dqn.epsilon = config.epsilon;
    dqn.gamma = config.gamma;
    dqn.stepCount = 0;

    dqn.memoryCapacity = config.memoryCapacity;
    dqn.memorySize = 0;
    dqn.memoryIndex = 0;
    dqn.memoryStates = zeros(4, config.memoryCapacity);
    dqn.memoryActions = zeros(1, config.memoryCapacity);
    dqn.memoryRewards = zeros(1, config.memoryCapacity);
    dqn.memoryNextStates = zeros(4, config.memoryCapacity);
    dqn.memoryDones = zeros(1, config.memoryCapacity);

    pretrainedInfo = struct('used', false, 'source', '');
    if ~isempty(pretrainedModel)
        modelStruct = [];
        if ischar(pretrainedModel) || isstring(pretrainedModel)
            modelPath = char(pretrainedModel);
            if exist(modelPath, 'file')
                loaded = load(modelPath);
                if isfield(loaded, 'dqnModelSaved')
                    modelStruct = loaded.dqnModelSaved;
                elseif isfield(loaded, 'dqnModel')
                    modelStruct = loaded.dqnModel;
                elseif isfield(loaded, 'pretrainedModel')
                    modelStruct = loaded.pretrainedModel;
                end
                pretrainedInfo.source = modelPath;
            else
                warning('Pretrained DQN-PSO model not found: %s. Using random initialization.', modelPath);
            end
        elseif isstruct(pretrainedModel)
            modelStruct = pretrainedModel;
            pretrainedInfo.source = 'struct';
        end

        if ~isempty(modelStruct)
            if isfield(modelStruct, 'qNet')
                dqn.qNet = modelStruct.qNet;
                dqn.targetNet = modelStruct.qNet;
                pretrainedInfo.used = true;
            end
            if isfield(modelStruct, 'epsilon')
                dqn.epsilon = modelStruct.epsilon;
            end
            if isfield(modelStruct, 'gamma')
                dqn.gamma = modelStruct.gamma;
            end
        end
    end

    dqn.pretrainedInfo = pretrainedInfo;
end

function net = createDQNNetwork()
    layers = [
        featureInputLayer(4, 'Name', 'state', 'Normalization', 'none')
        fullyConnectedLayer(32, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(4, 'Name', 'qValues')
    ];
    lgraph = layerGraph(layers);
    net = dlnetwork(lgraph);
    net = initializeWeights(net);
end

function net = initializeWeights(net)
    learnables = net.Learnables;
    for i = 1:height(learnables)
        paramName = learnables.Parameter{i};
        if strcmp(paramName, 'Weights')
            weights = learnables.Value{i};
            [outputSize, inputSize] = size(weights);
            scale = sqrt(2.0 / inputSize);
            learnables.Value{i} = dlarray(randn(outputSize, inputSize, 'single') * scale);
        elseif strcmp(paramName, 'Bias')
            bias = learnables.Value{i};
            learnables.Value{i} = dlarray(zeros(size(bias), 'single'));
        end
    end
    net.Learnables = learnables;
end

function [action, dqn] = selectDQNAction(dqn, stateVec, config)
    if rand() < dqn.epsilon
        action = randi(4);
        return;
    end

    stateDL = dlarray(stateVec, 'CB');
    qValues = forward(dqn.qNet, stateDL);
    [~, action] = max(extractdata(qValues));
end

function dqn = storeTransition(dqn, state, action, reward, nextState, done)
    dqn.memoryIndex = dqn.memoryIndex + 1;
    if dqn.memoryIndex > dqn.memoryCapacity
        dqn.memoryIndex = 1;
    end

    dqn.memoryStates(:, dqn.memoryIndex) = state;
    dqn.memoryActions(:, dqn.memoryIndex) = action;
    dqn.memoryRewards(:, dqn.memoryIndex) = reward;
    dqn.memoryNextStates(:, dqn.memoryIndex) = nextState;
    dqn.memoryDones(:, dqn.memoryIndex) = done;

    dqn.memorySize = min(dqn.memorySize + 1, dqn.memoryCapacity);
end

function [dqn, avgLoss] = trainDQN(dqn, config)
    avgLoss = NaN;
    if dqn.memorySize < config.batchSize
        return;
    end

    totalLoss = 0;
    lossCount = 0;
    for step = 1:config.dqnTrainSteps
        batchIdx = randi(dqn.memorySize, [1, config.batchSize]);
        states = dqn.memoryStates(:, batchIdx);
        actions = dqn.memoryActions(:, batchIdx);
        rewards = dqn.memoryRewards(:, batchIdx);
        nextStates = dqn.memoryNextStates(:, batchIdx);
        dones = dqn.memoryDones(:, batchIdx);

        [loss, grads] = dlfeval(@dqnLoss, dqn.qNet, dqn.targetNet, states, actions, ...
            rewards, nextStates, dones, dqn.gamma);
        [dqn.qNet, dqn.optimizer] = adamUpdate(dqn.qNet, grads, dqn.optimizer, ...
            config.learningRate, dqn.stepCount);
        dqn.stepCount = dqn.stepCount + 1;

        totalLoss = totalLoss + extractdata(loss);
        lossCount = lossCount + 1;
    end

    if lossCount > 0
        avgLoss = totalLoss / lossCount;
    end
    dqn.lastCriticLoss = avgLoss;
end

function [loss, gradients] = dqnLoss(qNet, targetNet, states, actions, rewards, nextStates, dones, gamma)
    statesDL = dlarray(states, 'CB');
    nextStatesDL = dlarray(nextStates, 'CB');

    qValues = forward(qNet, statesDL);

    targetQ = forward(targetNet, nextStatesDL);
    targetQ = extractdata(targetQ);
    maxNextQ = max(targetQ, [], 1);

    batchSize = size(states, 2);
    qChosen = zeros(1, batchSize, 'like', qValues);
    for i = 1:batchSize
        qChosen(i) = qValues(actions(i), i);
    end

    targets = rewards + gamma * maxNextQ .* (1 - dones);
    targets = reshape(targets, 1, []);

    targetsDL = dlarray(targets, 'CB');
    loss = mean((qChosen - targetsDL).^2);
    gradients = dlgradient(loss, qNet.Learnables);
end

function model = packDQNModel(dqn, config)
    model = struct();
    model.qNet = dqn.qNet;
    model.targetNet = dqn.targetNet;
    model.epsilon = dqn.epsilon;
    model.gamma = dqn.gamma;
    model.config = config;
    model.pretrainedInfo = dqn.pretrainedInfo;
end

function [net, opt] = adamUpdate(net, grads, opt, lr, step)
    if isempty(opt)
        opt = struct('m', grads, 'v', grads, 'beta1', 0.9, 'beta2', 0.999, 'eps', 1e-8);
        for i = 1:height(grads)
            opt.m.Value{i} = zeros(size(grads.Value{i}), 'like', grads.Value{i});
            opt.v.Value{i} = zeros(size(grads.Value{i}), 'like', grads.Value{i});
        end
    end

    for i = 1:height(grads)
        opt.m.Value{i} = opt.beta1 * opt.m.Value{i} + (1 - opt.beta1) * grads.Value{i};
        opt.v.Value{i} = opt.beta2 * opt.v.Value{i} + (1 - opt.beta2) * grads.Value{i}.^2;

        mHat = opt.m.Value{i} / (1 - opt.beta1^(step + 1));
        vHat = opt.v.Value{i} / (1 - opt.beta2^(step + 1));

        net.Learnables.Value{i} = net.Learnables.Value{i} - ...
            lr * mHat ./ (sqrt(vHat) + opt.eps);
    end
end
