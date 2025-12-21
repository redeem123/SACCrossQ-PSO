function trainRLNNPSO_GPU(numEpisodes, saveFilePath, useGPU, popSize, maxIterations)
    % GPU-Accelerated Pre-training for RL-NNPSO (TD3 Networks)
    % Optimized for particle-level control with curriculum learning
    % Curriculum Learning (Approach C): 70% Pure RL + 30% Adaptive Blending
    %
    % Inputs:
    %   numEpisodes: Number of training episodes (default: 300, optimized for ~2hr training)
    %   saveFilePath: Path to save trained networks (default: 'models/rlnnpso_gpu.mat')
    %   useGPU: true/false to enable/disable GPU (default: auto-detect)
    %   popSize: Population size for TRAINING (default: 50, benchmark uses 200 for testing)
    %   maxIterations: Max iterations for TRAINING (default: 200, benchmark uses 800 for testing)
    %
    % Optimized Settings (particle-level control = 200× more exp than RLAM-PSO):
    %   - Episodes: 300 (210 pure RL + 90 adaptive)
    %   - Training: 50 pop × 200 iter = 10,000 exp/episode (vs RLAM-PSO: 800)
    %   - Total: 3M experiences in ~2 hours
    %   - Testing: Networks generalize to 200 pop × 800 iter (benchmark scale)
    %   - Buffer size: 1,000,000 (fills 3× during training for good recycling)
    %   - Batch size: 256 (larger for GPU efficiency)
    %   - Learning rates: Actor 1e-4, Critic 1e-3
    %   - Gamma: 0.99, Tau: 0.005
    %   - TD3 with twin critics and delayed policy updates
    %
    % Curriculum Learning (fixes credit assignment problem):
    %   - Phase 1 (70% episodes): Pure RL (gamma=1.0) for strong learning signal
    %   - Phase 2 (30% episodes): Adaptive blending (gamma=0.3-0.9) for deployment prep
    %   - Testing: Adaptive blending (same as Phase 2)

    % Add all necessary paths
    currentDir = fileparts(mfilename('fullpath'));
    % Go up from training -> rlnnpso -> algorithms -> project root
    projectRoot = fileparts(fileparts(fileparts(currentDir)));
    addpath(genpath(fullfile(projectRoot, 'algorithms')));
    addpath(genpath(fullfile(projectRoot, 'shared', 'utilities')));
    addpath(genpath(fullfile(projectRoot, 'shared', 'environment')));

    % Parse inputs
    if nargin < 1
        numEpisodes = 300;  % Optimized for particle-level control (210 pure RL + 90 adaptive)
    end
    if nargin < 2
        saveFilePath = 'models/rlnnpso_gpu.mat';
    end
    if nargin < 3
        useGPU = gpuDeviceCount > 0;  % Auto-detect GPU
    end
    if nargin < 4
        popSize = 50;  % Training size (benchmark uses 200 for testing)
    end
    if nargin < 5
        maxIterations = 200;  % Training size (benchmark uses 800 for testing)
    end

    % GPU Setup
    if useGPU
        try
            gpuDevice();
            fprintf('=== GPU-Accelerated RL-NNPSO Training ===\n');
            fprintf('GPU Device: %s\n', gpuDevice().Name);
            fprintf('GPU Memory: %.2f GB\n\n', gpuDevice().AvailableMemory / 1e9);
        catch
            warning('GPU requested but not available. Falling back to CPU.');
            useGPU = false;
        end
    else
        fprintf('=== CPU-Based RL-NNPSO Training ===\n');
    end

    % Curriculum Learning Settings (Approach C)
    PURE_RL_EPISODES = round(0.7 * numEpisodes);  % 70% pure RL training

    fprintf('Training for %d episodes with optimized settings\n', numEpisodes);
    fprintf('Buffer size: 1,000,000 | Batch size: 256\n');
    fprintf('Training size: popSize=%d, maxIterations=%d\n', popSize, maxIterations);
    fprintf('Experiences per episode: %d (vs RLAM-PSO: 800)\n', popSize * maxIterations);
    fprintf('Total training experiences: %.1f million\n', numEpisodes * popSize * maxIterations / 1e6);
    fprintf('\n=== CURRICULUM LEARNING (Approach C) ===\n');
    fprintf('Phase 1 (Episodes 1-%d): Pure RL (gamma=1.0) - Strong learning signal\n', PURE_RL_EPISODES);
    fprintf('Phase 2 (Episodes %d-%d): Adaptive blending (gamma=0.3-0.9) - Deployment prep\n', PURE_RL_EPISODES+1, numEpisodes);
    fprintf('Buffer recycling: %.1fx during training (good experience diversity)\n\n', numEpisodes * popSize * maxIterations / 1e6);

    % Initialize TD3 networks with paper-standard hyperparameters
    numWaypoints = 5;
    dims = 3 * numWaypoints;  % 15 dimensions
    rlNetwork = initializeRLNetworkGPU(dims, useGPU);

    % Override with optimized buffer size
    rlNetwork.bufferSize = 1000000;  % 1M experiences (paper standard)
    rlNetwork.batchSize = 256;  % Larger batch for GPU efficiency

    % Buffer utilization with optimized settings (300 episodes × 50 pop × 200 iter):
    % Total experiences: 3M → Buffer fills 3× during training (excellent recycling)

    % Map configuration
    mapSize = [400, 400, 100];

    % PSO parameters
    current_w = 0.7;
    current_c1 = 1.5;
    current_c2 = 1.5;

    % Create training scenarios
    fprintf('Creating diversified training scenarios...\n');
    trainingScenarios = createRLNNPSOTrainingScenarios();

    % Track training progress
    episodeRewards = zeros(numEpisodes, 1);
    episodeAvgRewards = zeros(numEpisodes, 1);
    episodeFitness = zeros(numEpisodes, 1);
    trainingStartTime = tic;

    % Initialize CSV logging for issue detection
    trainingConfig = struct();
    trainingConfig.numEpisodes = numEpisodes;
    trainingConfig.popSize = popSize;
    trainingConfig.maxIterations = maxIterations;
    trainingConfig.bufferSize = rlNetwork.bufferSize;
    trainingConfig.batchSize = rlNetwork.batchSize;
    logFilePath = initializeTrainingLog('RLNNPSO', trainingConfig);

    % Training loop
    for episode = 1:numEpisodes
        episodeStart = tic;

        % Select scenario (cycle through scenarios)
        scenarioIdx = mod(episode - 1, length(trainingScenarios)) + 1;
        scenario = trainingScenarios{scenarioIdx};

        % Setup training scenario
        [startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY] = ...
            setupRLNNPSOTrainingScenario(scenario, mapSize);

        % Initialize PSO particles
        particles = initializeParticles(popSize, dims, numWaypoints, mapSize, ...
            startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY);

        globalBestPosition = zeros(1, dims);
        globalBestFitness = Inf;
        lastImprovementIteration = 0;

        % Find initial global best
        for i = 1:popSize
            if particles.fitness(i) < globalBestFitness
                globalBestFitness = particles.fitness(i);
                globalBestPosition = particles.cartesianPositions(i,:);
            end
        end

        episodeReward = 0;
        warmupPeriod = 50;

        % Swarm diversity calculation
        positionStd = std(particles.cartesianPositions, 0, 1);
        normVector = repmat(mapSize, 1, numWaypoints);
        normalizedStd = positionStd ./ normVector;
        swarmDiversity = mean(normalizedStd);

        % Track gamma values for logging
        gammaValues = [];

        % Training episode
        for iter = 1:maxIterations
            iterRewards = [];

            % For each particle
            for i = 1:popSize
                % Save previous values for reward calculation
                particles.prevBestFitness(i) = particles.bestFitness(i);
                particles.prevFitness(i) = particles.fitness(i);

                % Extract particle-specific state
                state = extractRLNNPSOState(particles, iter, maxIterations, ...
                    lastImprovementIteration, i, globalBestPosition, mapSize);

                % Get RL guidance (after warmup)
                if iter > warmupPeriod && length(rlNetwork.replayBuffer) >= rlNetwork.batchSize
                    % Normalize state
                    normalizedState = (state - rlNetwork.inputMean) ./ (rlNetwork.inputStd + 1e-8);
                    normalizedState(~isfinite(normalizedState)) = 0;
                    normalizedState = max(-3, min(3, normalizedState));

                    % Move to GPU if enabled
                    if useGPU
                        normalizedState = gpuArray(normalizedState);
                    end

                    % Get action from actor
                    [action, ~, ~, ~] = forwardPassActor(rlNetwork.actor, normalizedState);

                    % Move back to CPU for PSO operations
                    if useGPU
                        action = gather(action);
                    end

                    % Add exploration noise
                    dt = 1.0;
                    rlNetwork.ou_state = rlNetwork.ou_state - rlNetwork.ou_theta * rlNetwork.ou_state * dt + ...
                                         rlNetwork.ou_sigma * sqrt(dt) * randn(size(rlNetwork.ou_state));
                    noise = rlNetwork.ou_state * rlNetwork.explorationNoise;
                    noisyAction = max(-1, min(1, action + noise));

                    % Convert to velocity guidance
                    maxVelocityCorrection = 3.0;
                    velocityGuidance = noisyAction * maxVelocityCorrection;

                    % ========================================
                    % CURRICULUM LEARNING: Two Training Phases
                    % ========================================
                    if episode <= PURE_RL_EPISODES
                        % PHASE 1: Pure RL Learning (70% of training)
                        % gamma = 1.0 -> 100% RL guidance, 0% PSO
                        % Provides clean, undiluted learning signal
                        gamma = 1.0;
                        alpha = 0.0;

                        % Pure RL velocity update
                        particles.velocities(i,:) = velocityGuidance';

                        % Track gamma for logging
                        gammaValues = [gammaValues; gamma];

                    else
                        % PHASE 2: Adaptive Blending (30% of training)
                        % Learns to adapt RL guidance with PSO components
                        % Prepares network for real-world deployment

                        % Calculate adaptive gamma
                        if useGPU
                            normalizedState_gpu = gpuArray(normalizedState);
                            action_gpu = gpuArray(action);
                            [q1, ~, ~, ~] = forwardPassCritic(rlNetwork.critic1, normalizedState_gpu, action_gpu);
                            [q2, ~, ~, ~] = forwardPassCritic(rlNetwork.critic2, normalizedState_gpu, action_gpu);
                            q1 = gather(q1);
                            q2 = gather(q2);
                        else
                            [q1, ~, ~, ~] = forwardPassCritic(rlNetwork.critic1, normalizedState, action);
                            [q2, ~, ~, ~] = forwardPassCritic(rlNetwork.critic2, normalizedState, action);
                        end

                        qConsistency = 1.0 / (1.0 + abs(q1 - q2) + 1e-6);
                        dataSufficiency = min(1.0, length(rlNetwork.replayBuffer) / 1000);
                        iterProgress = min(1.0, (iter - warmupPeriod) / (maxIterations * 0.3));
                        diversityFactor = max(0.3, min(1.0, swarmDiversity / 0.15));

                        gamma = qConsistency * dataSufficiency * iterProgress * diversityFactor;
                        gamma = max(0.3, min(0.9, gamma));  % Higher minimum (0.3 vs 0.1)
                        alpha = 1.0 - gamma;

                        % Track gamma for logging
                        gammaValues = [gammaValues; gamma];

                        % PSO components
                        r1 = rand(1, dims);
                        r2 = rand(1, dims);
                        cognitive = current_c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));
                        social = current_c2 * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));

                        % Blended update
                        particles.velocities(i,:) = current_w * particles.velocities(i,:) + ...
                                                    alpha * (cognitive + social) + ...
                                                    gamma * velocityGuidance';
                    end
                else
                    % Standard PSO during warmup
                    r1 = rand(1, dims);
                    r2 = rand(1, dims);
                    cognitive = current_c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));
                    social = current_c2 * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));
                    particles.velocities(i,:) = current_w * particles.velocities(i,:) + cognitive + social;
                    noisyAction = zeros(dims, 1);
                end

                % Velocity clamping
                maxVelocity = 0.15 * (mapSize(1) + mapSize(2) + mapSize(3)) / 3;
                for j = 1:numWaypoints
                    idx = (j-1)*3 + 1;
                    segmentVel = particles.velocities(i, idx:idx+2);
                    velMag = norm(segmentVel);
                    if velMag > maxVelocity
                        particles.velocities(i, idx:idx+2) = segmentVel * (maxVelocity / velMag);
                    end
                end

                % Update position and apply constraints
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

                % Evaluate fitness
                [particles.fitness(i), ~] = evaluatePathFitness(particles.cartesianPositions(i,:), ...
                    startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY, numWaypoints);

                % Get next state
                nextState = extractRLNNPSOState(particles, iter, maxIterations, ...
                    lastImprovementIteration, i, globalBestPosition, mapSize);

                % Calculate reward
                reward = calculateRLNNPSOReward(particles.prevFitness(i), particles.fitness(i), ...
                    particles.prevBestFitness(i), particles.bestFitness(i), ...
                    particles.lastImprovement(i), swarmDiversity);

                iterRewards = [iterRewards; reward];

                % Store experience
                if iter > warmupPeriod
                    isDone = (iter == maxIterations);
                    rlNetwork = addToReplayBuffer(rlNetwork, state, noisyAction, reward, nextState, isDone);
                end

                % Update personal and global bests
                if particles.fitness(i) < particles.bestFitness(i)
                    particles.bestFitness(i) = particles.fitness(i);
                    particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
                    particles.lastImprovement(i) = 0;
                    if particles.fitness(i) < globalBestFitness
                        globalBestFitness = particles.fitness(i);
                        globalBestPosition = particles.cartesianPositions(i,:);
                        lastImprovementIteration = iter;
                    end
                else
                    particles.lastImprovement(i) = particles.lastImprovement(i) + 1;
                end
            end

            % Train networks (GPU-accelerated)
            if length(rlNetwork.replayBuffer) >= rlNetwork.batchSize && iter > warmupPeriod
                rlNetwork = trainTD3NetworksGPU(rlNetwork, useGPU);
            end

            % Decay exploration noise
            rlNetwork.explorationNoise = max(rlNetwork.minNoise, ...
                rlNetwork.explorationNoise * rlNetwork.noiseDecay);

            if ~isempty(iterRewards)
                episodeReward = episodeReward + mean(iterRewards);
            end

            % Update diversity
            positionStd = std(particles.cartesianPositions, 0, 1);
            normalizedStd = positionStd ./ normVector;
            swarmDiversity = mean(normalizedStd);
        end

        % Track episode metrics
        avgReward = episodeReward / maxIterations;
        episodeRewards(episode) = episodeReward;
        episodeAvgRewards(episode) = avgReward;
        episodeFitness(episode) = globalBestFitness;

        episodeTime = toc(episodeStart);
        cumulativeTime = toc(trainingStartTime);

        % Calculate average gamma for this episode
        if isempty(gammaValues)
            avgGamma = 0.0;  % Warmup period, no RL guidance used
        else
            avgGamma = mean(gammaValues);
        end

        % Log episode data to CSV for issue detection
        episodeData = struct();
        episodeData.episode = episode;
        episodeData.totalReward = episodeReward;
        episodeData.avgReward = avgReward;
        episodeData.bestFitness = globalBestFitness;
        episodeData.bufferSize = length(rlNetwork.replayBuffer);
        episodeData.explorationNoise = rlNetwork.explorationNoise;
        if episode <= PURE_RL_EPISODES
            episodeData.phase = 'PureRL';
        else
            episodeData.phase = 'Adaptive';
        end
        episodeData.avgGamma = avgGamma;
        episodeData.episodeTime = episodeTime;
        episodeData.cumulativeTime = cumulativeTime;
        logTrainingEpisode(logFilePath, episodeData);

        % Progress reporting
        if mod(episode, 10) == 0 || episode == 1 || episode == PURE_RL_EPISODES + 1
            windowAvgReward = mean(episodeAvgRewards(max(1, episode-9):episode));
            bufferSize = length(rlNetwork.replayBuffer);
            elapsedTime = toc(trainingStartTime);
            avgEpisodeTime = elapsedTime / episode;
            eta = avgEpisodeTime * (numEpisodes - episode);

            % Phase indicator
            if episode <= PURE_RL_EPISODES
                phaseStr = 'Phase 1: Pure RL';
            else
                phaseStr = 'Phase 2: Adaptive';
            end

            fprintf('[Ep %d/%d | %s] Fitness: %.2f | Reward: %.3f | Buffer: %d/%d | Noise: %.4f | Time: %.1fs | ETA: %.1fm\n', ...
                episode, numEpisodes, phaseStr, globalBestFitness, windowAvgReward, bufferSize, rlNetwork.bufferSize, ...
                rlNetwork.explorationNoise, episodeTime, eta/60);

            % Phase transition notification
            if episode == PURE_RL_EPISODES + 1
                fprintf('\n>>> PHASE TRANSITION: Switching to Adaptive Blending (gamma will vary 0.3-0.9) <<<\n\n');
            end
        end
    end

    totalTime = toc(trainingStartTime);
    fprintf('\n=== Training Complete ===\n');
    fprintf('Total time: %.2f minutes\n', totalTime/60);
    fprintf('Average episode time: %.2f seconds\n', totalTime/numEpisodes);

    % Save trained networks (move to CPU first)
    fprintf('\nSaving GPU-trained networks to: %s\n', saveFilePath);
    [filepath, ~, ~] = fileparts(saveFilePath);
    if ~isempty(filepath) && ~exist(filepath, 'dir')
        mkdir(filepath);
    end

    % Move networks from GPU to CPU for saving
    if useGPU
        rlNetwork = moveNetworkToCPU(rlNetwork);
    end

    % Extract networks
    trainedActor = rlNetwork.actor;
    trainedCritic1 = rlNetwork.critic1;
    trainedCritic2 = rlNetwork.critic2;
    trainedTargetActor = rlNetwork.targetActor;
    trainedTargetCritic1 = rlNetwork.targetCritic1;
    trainedTargetCritic2 = rlNetwork.targetCritic2;
    inputMean = rlNetwork.inputMean;
    inputStd = rlNetwork.inputStd;

    % Training info
    trainingInfo = struct();
    trainingInfo.numEpisodes = numEpisodes;
    trainingInfo.finalAvgReward = mean(episodeAvgRewards(max(1, end-9):end));
    trainingInfo.episodeRewards = episodeAvgRewards;
    trainingInfo.episodeFitness = episodeFitness;
    trainingInfo.trainingTime = totalTime;
    trainingInfo.usedGPU = useGPU;
    trainingInfo.bufferSize = rlNetwork.bufferSize;
    trainingInfo.batchSize = rlNetwork.batchSize;
    trainingInfo.curriculumLearning = true;
    trainingInfo.pureRLEpisodes = PURE_RL_EPISODES;
    trainingInfo.approachUsed = 'Approach C: Curriculum Learning (70% Pure RL + 30% Adaptive Blending)';

    save(saveFilePath, 'trainedActor', 'trainedCritic1', 'trainedCritic2', ...
        'trainedTargetActor', 'trainedTargetCritic1', 'trainedTargetCritic2', ...
        'inputMean', 'inputStd', 'trainingInfo');

    fprintf('✓ Networks saved successfully!\n');
    fprintf('\n=== CURRICULUM LEARNING RESULTS ===\n');
    fprintf('  Approach: %s\n', trainingInfo.approachUsed);
    fprintf('  Phase 1 (Pure RL):      Episodes 1-%d\n', PURE_RL_EPISODES);
    fprintf('  Phase 2 (Adaptive):     Episodes %d-%d\n', PURE_RL_EPISODES+1, numEpisodes);
    fprintf('  Final avg reward (last 10):  %.3f\n', trainingInfo.finalAvgReward);
    fprintf('  Final avg fitness (last 10): %.3f\n', mean(episodeFitness(max(1, end-9):end)));

    % Plot training progress
    plotTrainingProgress(episodeAvgRewards, episodeFitness);
end

%% Helper Functions

function rlNetwork = initializeRLNetworkGPU(dims, useGPU)
    % Initialize network with optional GPU support
    rlNetwork = initializeRLNetwork(dims);

    if useGPU
        % Move network weights to GPU
        rlNetwork = moveNetworkToGPU(rlNetwork);
    end
end

function rlNetwork = moveNetworkToGPU(rlNetwork)
    % Move all network weights to GPU
    fields = {'actor', 'critic1', 'critic2', 'targetActor', 'targetCritic1', 'targetCritic2'};
    for f = 1:length(fields)
        net = rlNetwork.(fields{f});
        weightFields = fieldnames(net);
        for w = 1:length(weightFields)
            if isnumeric(net.(weightFields{w}))
                net.(weightFields{w}) = gpuArray(net.(weightFields{w}));
            end
        end
        rlNetwork.(fields{f}) = net;
    end
end

function rlNetwork = moveNetworkToCPU(rlNetwork)
    % Move all network weights to CPU
    fields = {'actor', 'critic1', 'critic2', 'targetActor', 'targetCritic1', 'targetCritic2'};
    for f = 1:length(fields)
        net = rlNetwork.(fields{f});
        weightFields = fieldnames(net);
        for w = 1:length(weightFields)
            if isa(net.(weightFields{w}), 'gpuArray')
                net.(weightFields{w}) = gather(net.(weightFields{w}));
            end
        end
        rlNetwork.(fields{f}) = net;
    end
end

function rlNetwork = trainTD3NetworksGPU(rlNetwork, useGPU)
    % Train TD3 networks with GPU support
    numTrainSteps = 2;

    for trainStep = 1:numTrainSteps
        % Sample batch
        batch = sampleReplayBatch(rlNetwork, rlNetwork.batchSize);

        % Normalize states
        normalizedStates = (batch.states - rlNetwork.inputMean) ./ (rlNetwork.inputStd + 1e-8);
        normalizedStates(~isfinite(normalizedStates)) = 0;
        normalizedNextStates = (batch.nextStates - rlNetwork.inputMean) ./ (rlNetwork.inputStd + 1e-8);
        normalizedNextStates(~isfinite(normalizedNextStates)) = 0;

        % Move to GPU
        if useGPU
            normalizedStates = gpuArray(normalizedStates);
            normalizedNextStates = gpuArray(normalizedNextStates);
            batchActions = gpuArray(batch.actions);
            batchRewards = gpuArray(batch.rewards);
            batchDones = gpuArray(batch.dones);
        else
            batchActions = batch.actions;
            batchRewards = batch.rewards;
            batchDones = batch.dones;
        end

        % TD3: Target actions with smoothing
        [targetActions, ~, ~, ~] = forwardPassActor(rlNetwork.targetActor, normalizedNextStates);
        targetNoise = randn(size(targetActions), 'like', targetActions) * rlNetwork.targetNoiseStd;
        targetNoise = max(-rlNetwork.targetNoiseClip, min(rlNetwork.targetNoiseClip, targetNoise));
        targetActions = max(-1, min(1, targetActions + targetNoise));

        % Target Q-values
        [targetQ1, ~, ~, ~] = forwardPassCritic(rlNetwork.targetCritic1, normalizedNextStates, targetActions);
        [targetQ2, ~, ~, ~] = forwardPassCritic(rlNetwork.targetCritic2, normalizedNextStates, targetActions);
        targetQ = min(targetQ1, targetQ2);

        % TD target
        yTarget = batchRewards + rlNetwork.gamma * targetQ .* (1 - batchDones);

        % Update critics
        [currentQ1, h1_c1, h2_c1, h3_c1] = forwardPassCritic(rlNetwork.critic1, normalizedStates, batchActions);
        rlNetwork.critic1 = updateCriticSimple(rlNetwork.critic1, normalizedStates, batchActions, ...
            currentQ1, yTarget, h1_c1, h2_c1, h3_c1, rlNetwork.criticLR);

        [currentQ2, h1_c2, h2_c2, h3_c2] = forwardPassCritic(rlNetwork.critic2, normalizedStates, batchActions);
        rlNetwork.critic2 = updateCriticSimple(rlNetwork.critic2, normalizedStates, batchActions, ...
            currentQ2, yTarget, h1_c2, h2_c2, h3_c2, rlNetwork.criticLR);

        % Delayed actor update
        if mod(trainStep, rlNetwork.policyDelay) == 0
            [actorActions, h1_a, h2_a, h3_a] = forwardPassActor(rlNetwork.actor, normalizedStates);
            rlNetwork.actor = updateActorSimple(rlNetwork.actor, rlNetwork.critic1, ...
                normalizedStates, actorActions, h1_a, h2_a, h3_a, rlNetwork.actorLR);
            rlNetwork = softUpdateTargets(rlNetwork);
        end
    end

    % Update normalization (move to CPU for this)
    if useGPU
        states_cpu = gather(batch.states);
    else
        states_cpu = batch.states;
    end
    rlNetwork = updateNormalizationStatsOnline(rlNetwork, states_cpu);
end

function particles = initializeParticles(popSize, dims, numWaypoints, mapSize, startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY)
    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = Inf(popSize, 1);
    particles.prevBestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);
    particles.lastImprovement = ones(popSize, 1);
    particles.prevFitness = Inf(popSize, 1);

    for i = 1:popSize
        for j = 1:numWaypoints
            x = rand() * mapSize(1);
            y = rand() * mapSize(2);
            [~, xIndex] = min(abs(terrainX(1,:) - x));
            [~, yIndex] = min(abs(terrainY(:,1) - y));
            terrainHeight = terrainGrid(yIndex, xIndex);
            z = terrainHeight + 8 + rand() * 20;
            waypoint = max([1, 1, 1], min([x, y, z], mapSize));
            idx = (j-1)*3 + 1;
            particles.cartesianPositions(i, idx:idx+2) = waypoint;
        end

        particles.velocities(i,:) = (rand(1, dims) - 0.5) * 3;
        [particles.fitness(i), ~] = evaluatePathFitness(particles.cartesianPositions(i,:), ...
            startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY, numWaypoints);
        particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
        particles.bestFitness(i) = particles.fitness(i);
        particles.prevBestFitness(i) = particles.fitness(i);
        particles.prevFitness(i) = particles.fitness(i);
    end
end

function plotTrainingProgress(episodeAvgRewards, episodeFitness)
    figure('Name', 'RL-NNPSO GPU Training Progress', 'Position', [100, 100, 1200, 400]);

    subplot(1, 2, 1);
    plot(1:length(episodeAvgRewards), episodeAvgRewards, 'b-', 'LineWidth', 1);
    hold on;
    movAvg = movmean(episodeAvgRewards, 10);
    plot(1:length(movAvg), movAvg, 'r-', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Average Reward per Iteration');
    title('Training Rewards');
    legend('Episode Reward', '10-Episode Moving Avg');
    grid on;

    subplot(1, 2, 2);
    plot(1:length(episodeFitness), episodeFitness, 'g-', 'LineWidth', 1);
    hold on;
    movAvg = movmean(episodeFitness, 10);
    plot(1:length(movAvg), movAvg, 'r-', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Best Fitness (lower is better)');
    title('Training Fitness');
    legend('Episode Fitness', '10-Episode Moving Avg');
    grid on;
end

function scenarios = createRLNNPSOTrainingScenarios()
    scenarios = cell(4, 1);
    scenarios{1} = struct('numTrees', 0, 'numObstacles', 0, 'difficulty', 'easy');
    scenarios{2} = struct('numTrees', 10, 'numObstacles', 10, 'difficulty', 'medium');
    scenarios{3} = struct('numTrees', 20, 'numObstacles', 20, 'difficulty', 'hard');
    scenarios{4} = struct('numTrees', 30, 'numObstacles', 30, 'difficulty', 'very_hard');
end

function [startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY] = setupRLNNPSOTrainingScenario(scenario, mapSize)
    % Random start and goal with minimum distance constraint
    startPoint = [rand() * mapSize(1), rand() * mapSize(2), 20 + rand() * 30];
    goalPoint = [rand() * mapSize(1), rand() * mapSize(2), 20 + rand() * 30];
    minDistance = 0.3 * mean([mapSize(1), mapSize(2)]);
    while norm(startPoint - goalPoint) < minDistance
        goalPoint = [rand() * mapSize(1), rand() * mapSize(2), 20 + rand() * 30];
    end

    % Generate terrain
    [terrainX, terrainY] = meshgrid(linspace(0, mapSize(1), 50), linspace(0, mapSize(2), 50));
    terrainGrid = 10 + 20 * sin(terrainX / 50) .* cos(terrainY / 50);

    % Generate obstacles and trees
    trees = generateFixedTrees(scenario.numTrees, terrainGrid, terrainX, terrainY, mapSize);
    obstacles = generateFixedObstacles(scenario.numObstacles, mapSize, terrainGrid, terrainX, terrainY);
end
