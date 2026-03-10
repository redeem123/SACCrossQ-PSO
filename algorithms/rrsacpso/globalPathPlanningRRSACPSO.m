function [bestPath, bestFitness, fitnessHistory, agent, stateEncoder, parameterHistory, learningStats] = globalPathPlanningRRSACPSO(...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config)
    % RRSACPSO: Advanced Parameter Exploration CrossQ-SAC for PSO
    %
    % State-of-the-art RL-based PSO parameter adaptation using:
    %   - SAC with automatic entropy tuning
    %   - CrossQ optimizations (BatchNorm, UTD=1)
    %   - Deterministic cross-scale state encoding
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
    %   config: RRSACPSO configuration
    %
    % Outputs:
    %   bestPath: Best path found
    %   bestFitness: Best fitness value
    %   fitnessHistory: Fitness over training
    %   agent: Trained SAC agent
    %   stateEncoder: State encoder

    fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║              RRSACPSO Global Path Planning              ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

    % Online-only initialization: always start from scratch in-memory.
    agent = RRSACPSO_Agent(config);
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
        stateEncoder.reset();
        episodeBestFitness = min(particles.bestFitness);
        lastImprovementIteration = 0;

        initialFeatures = extractRichFeatures(particles, 1, config.maxIterations, ...
            lastImprovementIteration, episodeBestFitness);
        stateEncoder.addIteration(initialFeatures);
        state = stateEncoder.encode();

        % PSO iterations within episode
        for iter = 1:config.maxIterations
            % Get action from agent in online training mode.
            action = agent.getAction(state, true);

            % Log action statistics (every 50 iterations)
            if mod(iter, 50) == 0
                action_mean = mean(action);
                action_std = std(action);
                action_min = min(action);
                action_max = max(action);
                fprintf('      [Iter %3d] Action stats: mean=%.4f±%.4f [%.4f-%.4f]\n', ...
                    iter, action_mean, action_std, action_min, action_max);
            end

            % Convert latent action to per-particle parameters.
            particleParams = convertActionToPerParticleParams( ...
                action, config, particles, iter, config.maxIterations);

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
                mapSize, numWaypoints);

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

            % Get next state
            nextIter = min(iter + 1, config.maxIterations);
            nextBasicFeatures = extractRichFeatures(particles, nextIter, config.maxIterations, ...
                lastImprovementIteration, episodeBestFitness);
            stateEncoder.addIteration(nextBasicFeatures);
            nextState = stateEncoder.encode();

            reward = calculateImprovementReward(previousBestFitness, episodeBestFitness);

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
    fprintf('║              RRSACPSO TRAINING COMPLETE                   ║\n');
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

    % Initialize random positions
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

function reward = calculateImprovementReward(previousBestFitness, currentBestFitness)
    if currentBestFitness < previousBestFitness
        reward = 1.0;
    else
        reward = -1.0;
    end
end

function particles = resetParticles(particles, startPoint, goalPoint, terrainGrid, terrainX, terrainY, mapSize)
    % Reset particles for new episode (keep learned behavior, reset positions)
    [popSize, dims] = size(particles.cartesianPositions);
    numWaypoints = dims / 3;
    if nargin < 7 || isempty(mapSize)
        % Backward-compatible fallback when map size is not explicitly provided.
        mapSize = [max(terrainX(:)), max(terrainY(:)), 100];
    end

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
        particles.velocities(i, :) = zeros(1, dims);
        particles.fitness(i) = 1e9;
        particles.bestPositions(i, :) = particles.cartesianPositions(i, :);
        particles.bestFitness(i) = 1e9;
        particles.stagnationCounter(i) = 0;
        particles.collisionPenalty(i) = 0;
        particles.terrainPenalty(i) = 0;
        particles.dangerZonePenalty(i) = 0;
        particles.duplicatePenalty(i) = 0;
        particles.balancedFitness(i) = 0;
        particles.isFeasible(i) = false;
    end
end

function [particles, bestFitness] = updatePSOParticles(particles, particleParams, ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, numWaypoints)
    % Update PSO particles using per-particle parameters
    [popSize, dims] = size(particles.cartesianPositions);

    bestFitness = min(particles.bestFitness);

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
        [~, bestIdx] = min(particles.bestFitness);
        socialComponent = c2 * r2 .* (particles.bestPositions(bestIdx,:) - particles.cartesianPositions(i,:));

        particles.velocities(i,:) = w * particles.velocities(i,:) + cognitiveComponent + socialComponent;

        % Velocity clamping
        maxVelocity = 0.15 * (mapSize(1) + mapSize(2) + mapSize(3)) / 3;
        for j = 1:numWaypoints
            idx = (j-1)*3 + 1;
            segmentVel = particles.velocities(i, idx:idx+2);
            velMag = norm(segmentVel);

            % Sanity check for NaN/Inf
            if isnan(velMag) || isinf(velMag)
                % Reset velocity if invalid
                particles.velocities(i, idx:idx+2) = (rand(1,3)-0.5) * maxVelocity * 0.1;
                segmentVel = particles.velocities(i, idx:idx+2);
                velMag = norm(segmentVel);
            end

            if velMag > maxVelocity
                particles.velocities(i, idx:idx+2) = segmentVel * (maxVelocity / velMag);
            end
        end

        % Update position
        particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);

        % Apply constraints
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
    particles.collisionPenalty(idx) = getFieldOrDefault(components, 'collisionPenalty', 0);
    particles.terrainPenalty(idx) = getFieldOrDefault(components, 'terrainPenalty', 0);
    particles.dangerZonePenalty(idx) = getFieldOrDefault(components, 'dangerZonePenalty', 0);
    particles.duplicatePenalty(idx) = getFieldOrDefault(components, 'duplicatePenalty', 0);
    particles.balancedFitness(idx) = getFieldOrDefault(components, 'balancedFitness', particles.fitness(idx));
    particles.isFeasible(idx) = isfinite(particles.collisionPenalty(idx)) && ...
        particles.collisionPenalty(idx) <= 0 && ...
        particles.terrainPenalty(idx) <= 0 && ...
        isfinite(particles.dangerZonePenalty(idx));
end

function value = getFieldOrDefault(s, fieldName, defaultValue)
    if isfield(s, fieldName)
        value = s.(fieldName);
    else
        value = defaultValue;
    end
end

function diversity = calculateDiversity(particles)
    % Calculate population diversity
    positions = particles.cartesianPositions;
    diversityPerDim = std(positions, 0, 1);
    diversity = mean(diversityPerDim);
end

