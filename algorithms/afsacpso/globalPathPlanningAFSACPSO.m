function [bestPath, bestFitness, fitnessHistory, agent, stateEncoder, parameterHistory, learningStats] = globalPathPlanningAFSACPSO(...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config)
    % AFSACPSO: SAC-based attractor-field PSO parameter adaptation
    %
    % RL-based PSO parameter adaptation using:
    %   - SAC with automatic entropy tuning
    %   - Twin critics with target networks
    %   - Compact temporal swarm-state encoding
    %   - Rank-residual parameter adaptation
    %   - Simple fitness-improvement reward
    %
    % Inputs:
    %   startPoint: Start position [x, y, z]
    %   goalPoint: Goal position [x, y, z]
    %   obstacles: Obstacle data structure
    %   trees: Tree data structure
    %   terrainGrid: Terrain height map
    %   terrainX, terrainY: Terrain coordinate grids
    %   config: AFSACPSO configuration
    %
    % Outputs:
    %   bestPath: Best path found
    %   bestFitness: Best fitness value
    %   fitnessHistory: Fitness over training
    %   agent: Trained SAC agent
    %   stateEncoder: State encoder

    fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║              AFSACPSO Global Path Planning              ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

    % Online-only initialization: always start from scratch in-memory.
    agent = AFSACPSO_Agent(config);

    % Load pre-trained weights if specified (for frozen-policy evaluation)
    if isfield(config, 'pretrainedWeightsPath') && ~isempty(config.pretrainedWeightsPath)
        agent.loadWeights(config.pretrainedWeightsPath);
    end

    if isfield(config, 'useCrossScaleState') && config.useCrossScaleState
        stateEncoder = CrossScaleStateEncoder(config);
    else
        stateEncoder = SimpleStateEncoder(config);
    end
    
    % Initialize Training Logger (only for training modes)
    logger = [];
    if config.numEpisodes > 1
        logger = TrainingLogger(config);
    end

    % Initialize PSO
    [particles, mapSize, numWaypoints] = initializePSOParticles(config, startPoint, goalPoint, ...
        terrainGrid, terrainX, terrainY);

    % Training history:
    % - single-episode mode tracks per-iteration convergence
    % - multi-episode mode tracks per-episode convergence
    if config.numEpisodes == 1
        fitnessHistory = zeros(1, config.maxIterations);  % Track every iteration
        % Track population-average PSO parameters for visualization
        parameterHistory_w = zeros(1, config.maxIterations);
        parameterHistory_c1 = zeros(1, config.maxIterations);
        parameterHistory_c2 = zeros(1, config.maxIterations);
        parameterHistory_w_samples = zeros(config.maxIterations, config.popSize);
        parameterHistory_c1_samples = zeros(config.maxIterations, config.popSize);
        parameterHistory_c2_samples = zeros(config.maxIterations, config.popSize);
        rewardHistory = zeros(1, config.maxIterations);
        criticLossHistory = nan(1, config.maxIterations);
    else
        fitnessHistory = zeros(1, config.numEpisodes);    % Track every episode
        parameterHistory_w = [];
        parameterHistory_c1 = [];
        parameterHistory_c2 = [];
        parameterHistory_w_samples = [];
        parameterHistory_c1_samples = [];
        parameterHistory_c2_samples = [];
        rewardHistory = [];
        criticLossHistory = [];
    end
    % Best solution tracking
    globalBestFitness = Inf;
    globalBestPath = [];

    % Main training loop (episodes)
    for episode = 1:config.numEpisodes
        % Log episode start
        fprintf('\n[Episode %d/%d] Starting training...\n', episode, config.numEpisodes);
        fprintf('  Config: mode=%s, episodes=%d, iterations=%d, popSize=%d\n', ...
            config.mode, config.numEpisodes, config.maxIterations, config.popSize);
            
        episodeTimer = tic;
        if ~isempty(logger)
            logger.startEpisode(episode);
        end

        % Reset for new episode
        particles = resetParticles(particles, startPoint, goalPoint, terrainGrid, terrainX, terrainY, mapSize);
        particles = evaluateParticlePopulation(particles, startPoint, goalPoint, dangerZones, ...
            terrainGrid, terrainX, terrainY, numWaypoints);
        episodeBestFitness = min(particles.bestFitness);
        lastImprovementIteration = 0;

        % Mode flags for paper-aligned SAC-SAPSO
        usePaperState = isfield(config, 'stateMode') && strcmp(config.stateMode, 'paper');
        usePaperReward = isfield(config, 'rewardMode') && strcmp(config.rewardMode, 'paper');
        observationInterval = getFieldValue(config, 'observationInterval', 1);

        % Build initial state
        if usePaperState
            config.lastParticleParams = [];
            state = buildPaperState(particles, 1, config.maxIterations, config);
        else
            stateEncoder.reset();
            [initialFeatures, particles] = extractRichFeatures(particles, 1, config.maxIterations, ...
                lastImprovementIteration, episodeBestFitness);
            stateEncoder.addIteration(initialFeatures);
            state = stateEncoder.encode();
        end

        % For paper reward: track fitness at last observation
        lastObsFitness = episodeBestFitness;

        % Get initial action and convert to per-particle params
        if isfield(config, 'forcedZeroAction') && config.forcedZeroAction
            action = zeros(1, config.actionSize);
        else
            action = agent.getAction(state, true);
        end
        particleParams = convertActionToPerParticleParams( ...
            action, config, particles, 1, config.maxIterations);
        if usePaperState
            config.lastParticleParams = particleParams;
        end

        % PSO iterations within episode
        for iter = 1:config.maxIterations
            % Paper mode: only observe/act at observation intervals
            shouldObserve = (observationInterval <= 1) || ...
                (mod(iter, observationInterval) == 0) || (iter == config.maxIterations);

            % Get new action at observation points (initial action taken before loop)
            if shouldObserve && iter > 1
                if isfield(config, 'forcedZeroAction') && config.forcedZeroAction
                    action = zeros(1, config.actionSize);
                else
                    action = agent.getAction(state, true);
                end
                particleParams = convertActionToPerParticleParams( ...
                    action, config, particles, iter, config.maxIterations);
                if usePaperState
                    config.lastParticleParams = particleParams;
                end
            end

            % Log action statistics (every 50 iterations)
            if mod(iter, 50) == 0
                action_mean = mean(action);
                action_std = std(action);
                action_min = min(action);
                action_max = max(action);
                fprintf('      [Iter %3d] Action stats: mean=%.4f±%.4f [%.4f-%.4f]\n', ...
                    iter, action_mean, action_std, action_min, action_max);
            end

            % Store population-average parameters for visualization (online mode only)
            if config.numEpisodes == 1
                parameterHistory_w(iter) = mean(particleParams(:, 1));
                parameterHistory_c1(iter) = mean(particleParams(:, 2));
                parameterHistory_c2(iter) = mean(particleParams(:, 3));
                parameterHistory_w_samples(iter, :) = particleParams(:, 1)';
                parameterHistory_c1_samples(iter, :) = particleParams(:, 2)';
                parameterHistory_c2_samples(iter, :) = particleParams(:, 3)';
            end

            % Update PSO particles with individual parameters
            [particles, newBestFitness] = updatePSOParticles(particles, particleParams, ...
                startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, ...
                mapSize, numWaypoints, config);

            % Track improvement
            previousBestFitness = episodeBestFitness;
            if newBestFitness < episodeBestFitness
                episodeBestFitness = newBestFitness;
                lastImprovementIteration = iter;
            end

            % Calculate diversity
            currentDiversity = calculateDiversity(particles);

            % Log fitness improvement (every 100 iterations)
            if mod(iter, 100) == 0
                fprintf('      [Iter %3d] Fitness: current=%.4f, best=%.4f, diversity=%.6f\n', ...
                    iter, newBestFitness, episodeBestFitness, currentDiversity);
            end

            % --- State, reward, transition, and training (at observation points) ---
            if shouldObserve
                % Build next state
                if usePaperState
                    nextState = buildPaperState(particles, iter, config.maxIterations, config);
                else
                    nextIter = min(iter + 1, config.maxIterations);
                    [nextBasicFeatures, particles] = extractRichFeatures(particles, nextIter, config.maxIterations, ...
                        lastImprovementIteration, episodeBestFitness);
                    stateEncoder.addIteration(nextBasicFeatures);
                    nextState = stateEncoder.encode();
                end

                % Compute reward
                if usePaperReward
                    reward = calculatePaperReward(lastObsFitness, episodeBestFitness);
                    lastObsFitness = episodeBestFitness;
                else
                    reward = calculateImprovementReward(previousBestFitness, episodeBestFitness, currentDiversity);
                end

                if config.numEpisodes == 1
                    rewardHistory(iter) = reward;
                end

                % Log reward (every 100 iterations)
                if mod(iter, 100) == 0
                    fprintf('      [Iter %3d] Reward: %.4f\n', iter, reward);
                end

                % Store transition
                done = (iter == config.maxIterations);
                agent.storeTransition(state, action, reward, nextState, done);

                % Train agent
                losses = [];
                criticLoss = NaN;
                if iter > config.warmupPeriod && mod(iter, config.trainEveryNIterations) == 0
                    criticLosses = zeros(1, config.gradientStepsPerTraining);
                    for g = 1:config.gradientStepsPerTraining
                        losses = agent.train();
                        criticLosses(g) = losses.critic;
                    end
                    criticLoss = mean(criticLosses);
                end

                if config.numEpisodes == 1
                    criticLossHistory(iter) = criticLoss;
                end

                state = nextState;

                % Log training metrics
                if ~isempty(logger)
                    logger.logIteration(reward, episodeBestFitness, losses);
                end
            end

            % Single-episode mode: store per-iteration fitness and print progress.
            if config.numEpisodes == 1
                fitnessHistory(iter) = episodeBestFitness;
                if mod(iter, 50) == 0
                    fprintf('  [Iter %3d/%d] Best: %.4f | Alpha: %.4f\n', ...
                        iter, config.maxIterations, episodeBestFitness, exp(agent.logAlpha));
                end
            end

        end

        % Multi-episode mode: store per-episode fitness.
        if config.numEpisodes > 1
            fitnessHistory(episode) = episodeBestFitness;
            
            if ~isempty(logger)
                logger.endEpisode(episodeBestFitness, toc(episodeTimer));
                if mod(episode, 10) == 0
                    logger.printSummary();
                end
            end
        end

        % Log episode summary
        if config.numEpisodes > 1
            fprintf('\n  ✓ Episode %d COMPLETE: Best=%.4f, Global=%.4f, LastImprove=%d/%d\n', ...
                episode, episodeBestFitness, globalBestFitness, lastImprovementIteration, config.maxIterations);
            fprintf('    Agent Alpha: %.4f | Replay Buffer: %d/%d\n', exp(agent.logAlpha), agent.replayBuffer.size, agent.config.bufferSize);
        end

        % Update global best
        [currentBestFitness, bestIdx] = min(particles.bestFitness);
        if currentBestFitness < globalBestFitness
            globalBestFitness = currentBestFitness;
            globalBestPath = particles.bestPositions(bestIdx, :);
        end

        % Logging
        if mod(episode, config.logInterval) == 0
            fprintf('[Episode %d/%d] Best Fitness: %.4f | Global Best: %.4f | Alpha: %.4f\n', ...
                episode, config.numEpisodes, episodeBestFitness, globalBestFitness, ...
                exp(agent.logAlpha));
        end

        % Periodic checkpoint saving (multi-episode pre-training)
        saveInterval = getFieldValue(config, 'saveInterval', 50);
        if isfield(config, 'savePath') && mod(episode, saveInterval) == 0
            agent.saveWeights(config.savePath);
            fprintf('  [Checkpoint] Saved weights at episode %d to %s\n', episode, config.savePath);
        end

    end

    % Save final weights after all episodes (if multi-episode training)
    if config.numEpisodes > 1 && isfield(config, 'savePath')
        agent.saveWeights(config.savePath);
        fprintf('  [Final] Saved trained weights to %s\n', config.savePath);
    end

    % Return best solution
    if isempty(globalBestPath)
        [globalBestFitness, bestIdx] = min(particles.bestFitness);
        globalBestPath = particles.bestPositions(bestIdx, :);
    end
    bestFitness = globalBestFitness;

    % Convert particle position to proper path format (N×3 matrix with start and goal)
    % Constrain all waypoints to stay within mapSize
    numWaypoints = size(globalBestPath, 2) / 3;
    bestPath = constructPathFromPSO(globalBestPath, startPoint, goalPoint, numWaypoints, config.mapSize);

    % --- Waypoint-Decomposed Nelder-Mead Local Search ---
    if isfield(config, 'useTrajectoryPolish') && config.useTrajectoryPolish
        nmTimer = tic;
        [~, sortIdx] = sort(particles.bestFitness, 'ascend');
        topK = min(getFieldValue(config, 'polishNumSeedPaths', 3), length(sortIdx));
        seedPaths = particles.bestPositions(sortIdx(1:topK), :);
        [nmPath, nmFitness] = waypointDecomposedNM(bestPath, seedPaths, ...
            startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, ...
            numWaypoints, config);
        fprintf('  NM polish: %.1f -> %.1f (%.1f pts improvement, %.1fs)\n', ...
            bestFitness, nmFitness, bestFitness - nmFitness, toc(nmTimer));
        if nmFitness < bestFitness
            bestPath = nmPath;
            bestFitness = nmFitness;
        end
    end

    % --- Visualization Injection ---
    % Only visualize if running serially (main thread) to avoid parfor errors
    task = getCurrentTask();
    shouldVisualize = true;
    if isfield(config, 'disableVisualization') && config.disableVisualization
        shouldVisualize = false;
    end
    if isempty(task) && shouldVisualize
        try
            visualizeSingleRun(bestPath, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config.mapSize);
        catch e
            fprintf('Warning: Visualization failed: %s\n', e.message);
        end
    end
    % -------------------------------

    % Package parameter history for visualization
    parameterHistory = struct();
    if config.numEpisodes == 1
        parameterHistory.w = parameterHistory_w;
        parameterHistory.c1 = parameterHistory_c1;
        parameterHistory.c2 = parameterHistory_c2;
        parameterHistory.w_samples = parameterHistory_w_samples;
        parameterHistory.c1_samples = parameterHistory_c1_samples;
        parameterHistory.c2_samples = parameterHistory_c2_samples;
    else
        parameterHistory.w = [];
        parameterHistory.c1 = [];
        parameterHistory.c2 = [];
        parameterHistory.w_samples = [];
        parameterHistory.c1_samples = [];
        parameterHistory.c2_samples = [];
    end

    learningStats = struct();
    learningStats.rewardHistory = rewardHistory;
    learningStats.criticLossHistory = criticLossHistory;

    % Persist logger metrics only when running multi-episode experiments.
    if config.numEpisodes > 1 && ~isempty(logger)
        [p, n, ~] = fileparts(config.savePath);
        logPath = fullfile(p, [n '_log.mat']);
        logger.save(logPath);
    end

    % Final training summary
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║              AFSACPSO TRAINING COMPLETE                   ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n');
    fprintf('  Mode:              %s\n', config.mode);
    fprintf('  Episodes:          %d\n', config.numEpisodes);
    fprintf('  Iterations/Ep:     %d\n', config.maxIterations);
    fprintf('  Population Size:   %d\n', config.popSize);
    fprintf('  Best Fitness:      %.4f\n', bestFitness);
    if config.numEpisodes > 1
        fprintf('  Min Fitness:       %.4f (Episode %d)\n', min(fitnessHistory), find(fitnessHistory == min(fitnessHistory)));
        fprintf('  Max Fitness:       %.4f (Episode %d)\n', max(fitnessHistory), find(fitnessHistory == max(fitnessHistory)));
        fprintf('  Mean Fitness:      %.4f\n', mean(fitnessHistory));
    end
    fprintf('  Path Length:       %d waypoints\n', numWaypoints);
    fprintf('\n');
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
    particles.stagnationCounter = zeros(popSize, 1);
    particles.collisionPenalty = zeros(popSize, 1);
    particles.terrainPenalty = zeros(popSize, 1);
    particles.dangerZonePenalty = zeros(popSize, 1);
    particles.duplicatePenalty = zeros(popSize, 1);
    particles.balancedFitness = zeros(popSize, 1);
    particles.isFeasible = false(popSize, 1);

    % Initialize random positions (terrain-aware)
    particles.cartesianPositions = generateRandomPSOPositions( ...
        popSize, numWaypoints, mapSize, terrainGrid, terrainX, terrainY);
    particles.bestPositions = particles.cartesianPositions;
end

function reward = calculateImprovementReward(previousBestFitness, currentBestFitness, ~)
    % Scale-invariant relative improvement reward.
    % Matches the SAC-SAPSO paper formulation (von Eschwege & Engelbrecht 2024).
    reward = 2 * (previousBestFitness - currentBestFitness) / ...
        (abs(previousBestFitness) + abs(currentBestFitness) + 1e-8);
end

function particles = resetParticles(particles, startPoint, goalPoint, terrainGrid, terrainX, terrainY, mapSize)
    % Reset particles for new episode (keep learned behavior, reset positions)
    [popSize, dims] = size(particles.cartesianPositions);
    numWaypoints = dims / 3;
    if nargin < 7 || isempty(mapSize)
        % Backward-compatible fallback when map size is not explicitly provided.
        mapSize = [max(terrainX(:)), max(terrainY(:)), 100];
    end

    particles.cartesianPositions = generateRandomPSOPositions( ...
        popSize, numWaypoints, mapSize, terrainGrid, terrainX, terrainY);
    particles.velocities(:) = 0;
    particles.fitness(:) = 1e9;
    particles.bestPositions = particles.cartesianPositions;
    particles.bestFitness(:) = 1e9;
    particles.stagnationCounter(:) = 0;
    particles.collisionPenalty(:) = 0;
    particles.terrainPenalty(:) = 0;
    particles.dangerZonePenalty(:) = 0;
    particles.duplicatePenalty(:) = 0;
    particles.balancedFitness(:) = 0;
    particles.isFeasible(:) = false;
end

function [particles, bestFitness] = updatePSOParticles(particles, particleParams, ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, numWaypoints, config)
    % Update PSO particles using per-particle parameters
    [popSize, dims] = size(particles.cartesianPositions);

    bestFitness = min(particles.bestFitness);

    % Per-dimension velocity clamping (aligned with SAC-SAPSO paper)
    delta = 0.5;
    if nargin >= 11 && isfield(config, 'velocityClampDelta')
        delta = config.velocityClampDelta;
    end
    vMaxDim = delta * (mapSize - 1);  % [vmax_x, vmax_y, vmax_z]

    % Precompute terrain grid spacing for O(1) index lookup
    txVec = terrainX(1,:);
    tyVec = terrainY(:,1);
    txMin = txVec(1); txStep = txVec(2) - txVec(1); txN = numel(txVec);
    tyMin = tyVec(1); tyStep = tyVec(2) - tyVec(1); tyN = numel(tyVec);

    % Pre-tile velocity limits for vectorized clamping
    vMaxLo = repmat(-vMaxDim, 1, numWaypoints);
    vMaxHi = repmat( vMaxDim, 1, numWaypoints);

    % Global best — compute once outside particle loop
    [~, bestIdx] = min(particles.bestFitness);
    globalBestPos = particles.bestPositions(bestIdx,:);

    % Update each particle with its specific parameters
    for i = 1:popSize
        % Get parameters for this particle
        w = particleParams(i, 1);
        c1 = particleParams(i, 2);
        c2 = particleParams(i, 3);

        % PSO velocity update
        r1 = rand(1, dims);
        r2 = rand(1, dims);

        cognitiveComponent = c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));
        socialComponent = c2 * r2 .* (globalBestPos - particles.cartesianPositions(i,:));

        particles.velocities(i,:) = w * particles.velocities(i,:) + cognitiveComponent + socialComponent;

        % Per-dimension velocity clamping (vectorized)
        vi = particles.velocities(i,:);
        badMask = isnan(vi) | isinf(vi);
        if any(badMask)
            vi(badMask) = (rand(1, sum(badMask))-0.5) .* vMaxHi(badMask) * 0.1;
        end
        vi = max(vMaxLo, min(vMaxHi, vi));
        particles.velocities(i,:) = vi;

        % Update position
        particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);

        % Apply constraints
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

        % Evaluate fitness
        [particles.fitness(i), components] = evaluatePathFitness(particles.cartesianPositions(i,:), ...
            startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        
        % Clamp infinite fitness to large number for numerical stability
        if isinf(particles.fitness(i))
            particles.fitness(i) = 1e9;
        end

        particles = assignParticleComponents(particles, i, components);

        % Update personal best
        if particles.fitness(i) < particles.bestFitness(i)
            particles.bestFitness(i) = particles.fitness(i);
            particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
            particles.stagnationCounter(i) = 0;

            if particles.fitness(i) < bestFitness
                bestFitness = particles.fitness(i);
            end
        else
            particles.stagnationCounter(i) = particles.stagnationCounter(i) + 1;
        end
    end
end

function particles = evaluateParticlePopulation(particles, startPoint, goalPoint, dangerZones, ...
    terrainGrid, terrainX, terrainY, numWaypoints)
    popSize = size(particles.cartesianPositions, 1);
    for i = 1:popSize
        [fitness, components] = evaluatePathFitness(particles.cartesianPositions(i,:), ...
            startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        if isinf(fitness)
            fitness = 1e9;
        end
        particles.fitness(i) = fitness;
        particles.bestFitness(i) = fitness;
        particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
        particles = assignParticleComponents(particles, i, components);
    end
end

function particles = assignParticleComponents(particles, idx, components)
    particles.collisionPenalty(idx) = getFieldValue(components, 'collisionPenalty', 0);
    particles.terrainPenalty(idx) = getFieldValue(components, 'terrainPenalty', 0);
    particles.dangerZonePenalty(idx) = getFieldValue(components, 'dangerZonePenalty', 0);
    particles.duplicatePenalty(idx) = getFieldValue(components, 'duplicatePenalty', 0);
    particles.balancedFitness(idx) = getFieldValue(components, 'balancedFitness', particles.fitness(idx));
    particles.isFeasible(idx) = isfinite(particles.collisionPenalty(idx)) && ...
        particles.collisionPenalty(idx) <= 0 && ...
        particles.terrainPenalty(idx) <= 0 && ...
        isfinite(particles.dangerZonePenalty(idx));
end

function diversity = calculateDiversity(particles)
    % Calculate population diversity
    positions = particles.cartesianPositions;
    diversityPerDim = std(positions, 0, 1);
    diversity = mean(diversityPerDim);
end

