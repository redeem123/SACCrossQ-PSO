function [globalPath, convergenceHistory, parameterHistory, algorithmSpecificStats] = globalPathPlanningRLAMPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, initialW, initialC1, initialC2, pretrainedNetworkPath, config, preloadedAgent)
    % RLAM-PSO: Paper-exact DDPG implementation
    %
    % Reference: Yin et al., "Reinforcement-learning-based parameter adaptation
    % method for particle swarm optimization", Complex & Intelligent Systems, 2023
    %
    % Uses MATLAB Deep Learning Toolbox for automatic differentiation.
    %
    % Inputs (optional):
    %   pretrainedNetworkPath: Path to pre-trained networks (default: random init)
    %   config: Configuration from RLAMPSO_Config() (default: baseline)
    %   preloadedAgent: Pre-initialized agent to reuse (default: [])

    if nargin < 13, pretrainedNetworkPath = ''; end
    if nargin < 14 || isempty(config), config = RLAMPSO_Config('baseline'); end
    if nargin < 15, preloadedAgent = []; end

    % Align config with benchmark-dispatched budget
    config.popSize = popSize;
    config.maxIterations = maxIterations;

    if ~isfield(config, 'paramMode') || isempty(config.paramMode)
        config.paramMode = '5subgroup';
    end
    if ~strcmp(config.paramMode, '5subgroup')
        warning('RLAMPSO paper mode uses fixed 5-subgroup control; forcing paramMode to 5subgroup.');
        config.paramMode = '5subgroup';
    end
    config.actionSize = 20;
    config.numSubgroups = 5;

    verbose = isfield(config, 'verbose') && config.verbose;
    numWaypoints = 5;
    dims = 3 * numWaypoints;

    % ===== OUTPUT ACCUMULATORS =====
    convergenceHistory = [];
    rewardHistory = [];
    criticLossHistory = [];

    parameterHistory = struct();
    parameterHistory.w  = [];
    parameterHistory.c1 = [];
    parameterHistory.c2 = [];

    globalBestComponents = struct('pathLength',0,'collisionPenalty',0,...
        'turningPenalty',0,'climbingPenalty',0,'heightPenalty',0,...
        'terrainPenalty',0,'obstaclePenalty',0);

    % ===== INITIALIZE DDPG AGENT =====
    if ~isempty(preloadedAgent)
        rlamAgent = preloadedAgent;
    elseif ~isempty(pretrainedNetworkPath) && ischar(pretrainedNetworkPath) && exist(pretrainedNetworkPath, 'file')
        loaded = load(pretrainedNetworkPath);
        if isfield(loaded, 'trainedAgent'),     rlamAgent = loaded.trainedAgent;
        elseif isfield(loaded, 'agent'),        rlamAgent = loaded.agent;
        else, error('Pretrained file must contain "trainedAgent" or "agent" field');
        end
    else
        rlamAgent = initializeAgentAuto(config, '');
    end

    % ===== INITIALIZE PARTICLES =====
    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities         = zeros(popSize, dims);
    particles.bestPositions      = zeros(popSize, dims);
    particles.bestFitness        = Inf(popSize, 1);
    particles.fitness            = Inf(popSize, 1);

    globalBestPosition = zeros(1, dims);
    globalBestFitness  = Inf;
    lastImprovementIteration = 0;
    firstFeasibleIteration = -1;
    initialFeasibleCount = 0;

    numSubgroups = config.numSubgroups;
    parameterHistory.w_samples  = zeros(maxIterations, numSubgroups);
    parameterHistory.c1_samples = zeros(maxIterations, numSubgroups);
    parameterHistory.c2_samples = zeros(maxIterations, numSubgroups);

    for i = 1:popSize
        for j = 1:numWaypoints
            x = rand() * mapSize(1);
            y = rand() * mapSize(2);
            [~, xIdx] = min(abs(terrainX(1,:) - x));
            [~, yIdx] = min(abs(terrainY(:,1) - y));
            z = terrainGrid(yIdx, xIdx) + 8 + rand() * 20;
            wp = max([1,1,1], min([x,y,z], mapSize));
            idx = (j-1)*3 + 1;
            particles.cartesianPositions(i, idx:idx+2) = wp;
        end

        if isPathFeasible(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
            initialFeasibleCount = initialFeasibleCount + 1;
        end

        particles.velocities(i,:) = (rand(1, dims) - 0.5) * 3;
        [particles.fitness(i), fc] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
        particles.bestFitness(i) = particles.fitness(i);

        if i == 1 || particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestPosition = particles.cartesianPositions(i,:);
            globalBestComponents = fc;
            lastImprovementIteration = 0;
        end
    end

    if initialFeasibleCount > 0, firstFeasibleIteration = 0; end
    convergenceHistory = [convergenceHistory; globalBestFitness];

    % ===== MAIN RLAM-PSO LOOP (Paper Algorithm 1) =====
    previousReward = 0;
    previousNextState = [];
    warmupPeriod = 50;
    trainFreq = config.trainEveryNIterations;

    for iter = 1:maxIterations
        previousGlobalBestFitness = globalBestFitness;
        currentCriticLoss = NaN;

        % --- ONLINE DDPG TRAINING (Paper Algorithm 1, steps 9-10) ---
        shouldTrain = iter > warmupPeriod && ...
                      ~isempty(previousNextState) && ...
                      mod(iter, trainFreq) == 0;

        if shouldTrain
            rlamAgent = updateAgentAuto(rlamAgent, previousNextState, [], previousReward);
            if isfield(rlamAgent, 'lastCriticLoss')
                currentCriticLoss = rlamAgent.lastCriticLoss;
            end
        end

        % --- STATE (Paper Eq. 7–10, 15D sin-encoding) ---
        rlamState = calculatePaperExactState(particles, iter, maxIterations, lastImprovementIteration);

        % --- ACTION (Paper Eq. 11, 14) ---
        [subgroupParams, rlamAgent] = getPaperExactAction(rlamAgent, rlamState, iter);

        % Track parameters
        avgW  = mean(subgroupParams(:,1));
        avgC1 = mean(subgroupParams(:,2));
        avgC2 = mean(subgroupParams(:,3));
        parameterHistory.w  = [parameterHistory.w;  avgW];
        parameterHistory.c1 = [parameterHistory.c1; avgC1];
        parameterHistory.c2 = [parameterHistory.c2; avgC2];
        parameterHistory.w_samples(iter,:)  = subgroupParams(:,1)';
        parameterHistory.c1_samples(iter,:) = subgroupParams(:,2)';
        parameterHistory.c2_samples(iter,:) = subgroupParams(:,3)';

        % --- UPDATE PARTICLES BY SUBGROUP (Paper Section 4.1.2) ---
        % Paper: "subgroup = index % 5" (modulo interleaving, 0-indexed)
        for i = 1:popSize
            sg = mod(i - 1, numSubgroups) + 1;  % 1-indexed modulo
            w_sub  = subgroupParams(sg, 1);
            c1_sub = subgroupParams(sg, 2);
            c2_sub = subgroupParams(sg, 3);
                r1 = rand(1, dims);
                r2 = rand(1, dims);

                particles.velocities(i,:) = w_sub * particles.velocities(i,:) ...
                    + c1_sub * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) ...
                    + c2_sub * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));

                % Velocity clamping
                maxVel = 0.15 * mean(mapSize);
                for j = 1:numWaypoints
                    vi = (j-1)*3 + 1;
                    seg = particles.velocities(i, vi:vi+2);
                    mag = norm(seg);
                    if mag > maxVel
                        particles.velocities(i, vi:vi+2) = seg * (maxVel / mag);
                    end
                end

                % Update position
                particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);

                % Enforce constraints
                for j = 1:numWaypoints
                    vi = (j-1)*3 + 1;
                    wp = particles.cartesianPositions(i, vi:vi+2);
                    wp = max([1,1,1], min(wp, mapSize));
                    [~, xI] = min(abs(terrainX(1,:) - wp(1)));
                    [~, yI] = min(abs(terrainY(:,1) - wp(2)));
                    wp(3) = max(wp(3), terrainGrid(yI, xI) + 10);
                    particles.cartesianPositions(i, vi:vi+2) = wp;
                end

                % Evaluate fitness
                [particles.fitness(i), fc] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

                % Update personal best
                eps_u = 1e-6;
                if (particles.bestFitness(i) - particles.fitness(i)) > eps_u
                    particles.bestFitness(i) = particles.fitness(i);
                    particles.bestPositions(i,:) = particles.cartesianPositions(i,:);

                    if (globalBestFitness - particles.fitness(i)) > eps_u
                        globalBestFitness = particles.fitness(i);
                        globalBestPosition = particles.cartesianPositions(i,:);
                        globalBestComponents = fc;
                        lastImprovementIteration = iter;
                    end
                end
        end

        % --- REWARD (Paper Eq. 13, binary +1/−1) ---
        fitnessImprovement = previousGlobalBestFitness - globalBestFitness;
        if fitnessImprovement > 1e-6
            reward = 1;
        else
            reward = -1;
        end

        % --- NEXT STATE ---
        nextState = calculatePaperExactState(particles, iter+1, maxIterations, lastImprovementIteration);

        % Store for next iteration's training
        previousReward    = reward;
        previousNextState = nextState;

        rewardHistory     = [rewardHistory; reward];
        criticLossHistory = [criticLossHistory; currentCriticLoss];
        convergenceHistory = [convergenceHistory; globalBestFitness];

        % Redundant global-best check
        [minBF, bIdx] = min(particles.bestFitness);
        if (globalBestFitness - minBF) > 1e-6
            globalBestFitness  = minBF;
            globalBestPosition = particles.bestPositions(bIdx,:);
            [~, globalBestComponents] = evaluatePathFitness(globalBestPosition, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
            lastImprovementIteration = iter;
        end

        if verbose && (mod(iter,50)==0 || iter==1 || iter==maxIterations)
            fprintf('[%d/%d] Fitness: %.3f | Stagnation: %d | w=%.3f c1=%.3f c2=%.3f\n', ...
                iter, maxIterations, globalBestFitness, iter-lastImprovementIteration, avgW, avgC1, avgC2);
        end
    end

    % Final training step
    if ~isempty(previousNextState)
        rlamAgent = trainAgentAuto(rlamAgent, previousNextState, [], previousReward);
    end

    % Construct final path
    globalPath = constructPathFromPSO(globalBestPosition, startPoint, goalPoint, numWaypoints, mapSize);

    % Algorithm statistics
    [~, globalBestComponents] = evaluatePathFitness(globalBestPosition, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
    algorithmSpecificStats.actualBestFitness = globalBestFitness;
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, globalBestFitness, globalBestComponents);
    algorithmSpecificStats.rewardHistory     = rewardHistory;
    algorithmSpecificStats.criticLossHistory = criticLossHistory;
    algorithmSpecificStats.trainedAgent      = rlamAgent;
    algorithmSpecificStats.config            = config;
end
