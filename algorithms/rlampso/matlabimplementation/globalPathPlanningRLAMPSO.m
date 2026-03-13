function [globalPath, convergenceHistory, parameterHistory, algorithmSpecificStats] = globalPathPlanningRLAMPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, initialW, initialC1, initialC2, pretrainedNetworkPath, config, preloadedAgent)
    % RLAM-PSO: DL Toolbox Implementation (REQUIRED)
    % "Reinforcement Learning-based parameter Adaptation Method for PSO"
    %
    % Uses MATLAB Deep Learning Toolbox for automatic differentiation
    % Will error if DL Toolbox is not available
    %
    % Additional inputs (optional):
    %   pretrainedNetworkPath: Path to pre-trained networks (default: empty, random init)
    %   config: Configuration from RLAMPSO_Config() (default: baseline)
    %   preloadedAgent: Pre-initialized agent to reuse (default: empty, will initialize new)

    if nargin < 14
        pretrainedNetworkPath = '';  % Default: no pre-trained networks
    end

    if nargin < 15 || isempty(config)
        config = RLAMPSO_Config('baseline');  % Default configuration
    end

    if nargin < 16
        preloadedAgent = [];  % Default: no preloaded agent
    end

    % Keep local config aligned with the benchmark-dispatched budget.
    % The function inputs are the source of truth for population and iteration count.
    config.popSize = popSize;
    config.maxIterations = maxIterations;

    if ~isfield(config, 'paramMode') || isempty(config.paramMode)
        config.paramMode = '5subgroup';
    end

    switch config.paramMode
        case 'global'
            config.actionSize = 4;
            config.numSubgroups = 1;
        case '5subgroup'
            config.actionSize = 20;
            config.numSubgroups = 5;
        case 'per-particle'
            config.actionSize = config.popSize * 4;
            config.numSubgroups = config.popSize;
        otherwise
            error('Unsupported RLAMPSO paramMode: %s', config.paramMode);
    end

    if isfield(config, 'verbose') && config.verbose
        disp('=================================================================');
        if ~isempty(pretrainedNetworkPath)
            disp('Starting RLAM-PSO with PRE-TRAINED networks...');
        else
            disp('Starting RLAM-PSO with RANDOM initialization...');
        end
        fprintf('Implementation: DL Toolbox (dlnetwork + auto-diff)\n');
        fprintf('Features: TD3=%d, Transformer=%d, PER=%d\n', ...
                config.useTD3, config.useTransformerState, config.usePER);
        disp('=================================================================');
    end
    
    numWaypoints = 5;
    dims = 3 * numWaypoints;
    
    firstFeasibleIteration = -1;
    initialFeasibleCount = 0;
    convergenceHistory = [];
    rewardHistory = [];
    criticLossHistory = [];
    
    parameterHistory = struct();
    parameterHistory.w = [];
    parameterHistory.c1 = [];
    parameterHistory.c2 = [];
    
    % Initialize globalBestComponents
    globalBestComponents = struct();
    globalBestComponents.pathLength = 0;
    globalBestComponents.collisionPenalty = 0;
    globalBestComponents.turningPenalty = 0;
    globalBestComponents.climbingPenalty = 0;
    globalBestComponents.heightPenalty = 0;
    globalBestComponents.terrainPenalty = 0;
    globalBestComponents.obstaclePenalty = 0;
    
    % ===== INITIALIZE AGENT (DL Toolbox REQUIRED) =====

    if ~isempty(preloadedAgent)
        % Use preloaded agent (for training loops - FAST!)
        rlamAgent = preloadedAgent;
    elseif ~isempty(pretrainedNetworkPath) && ischar(pretrainedNetworkPath) && exist(pretrainedNetworkPath, 'file')
        % Load pretrained agent from file
        loaded = load(pretrainedNetworkPath);
        if isfield(loaded, 'trainedAgent')
            rlamAgent = loaded.trainedAgent;
        elseif isfield(loaded, 'agent')
            rlamAgent = loaded.agent;
        else
            error('Pretrained file must contain "trainedAgent" or "agent" field');
        end
    else
        % Initialize new DL Toolbox agent
        % Will error if DL Toolbox is not available
        rlamAgent = initializeAgentAuto(config, '');
    end

    % Initialize Transformer State Encoder (if enabled)
    % CRITICAL FIX: Persist transformer across episodes, don't recreate!
    if config.useTransformerState
        % Check if transformer already exists in agent (from previous episode)
        if isfield(rlamAgent, 'transformerEncoder') && ~isempty(rlamAgent.transformerEncoder)
            % Reuse existing transformer (PERSISTS learning across episodes)
            transformerEncoder = rlamAgent.transformerEncoder;
            temporalBuffer = rlamAgent.temporalBuffer;
        else
            % First episode: initialize new transformer
            fprintf('  → Initializing Transformer state encoder (64D)\n');
            transformerEncoder = initializeTransformerStateEncoder(34, config.transformerHiddenDim, config.transformerNumHeads);
            temporalBuffer = [];  % Will store last N temporal states
        end
    else
        transformerEncoder = [];
        temporalBuffer = [];
    end

    % PAPER: RLAM-PSO is an ONLINE adaptation method - ALWAYS trains during execution
    % Pretrained networks provide initialization from past experience, then continue learning

    % Initialize particles
    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);
    
    globalBestPosition = zeros(1, dims);
    globalBestFitness = Inf;
    previousGlobalBestFitness = Inf;
    
    % PAPER: Track stagnation (Fe_num - Fe_num_last_improve)
    lastImprovementIteration = 0;

    numSubgroups = config.numSubgroups;
    subgroupSize = floor(popSize / numSubgroups);
    parameterHistory.w_samples = zeros(maxIterations, numSubgroups);
    parameterHistory.c1_samples = zeros(maxIterations, numSubgroups);
    parameterHistory.c2_samples = zeros(maxIterations, numSubgroups);
    
    % Initialize particles
    for i = 1:popSize
        for j = 1:numWaypoints
            x = rand() * mapSize(1);
            y = rand() * mapSize(2);
            
            [~, xIndex] = min(abs(terrainX(1,:) - x));
            [~, yIndex] = min(abs(terrainY(:,1) - y));
            terrainHeight = terrainGrid(yIndex, xIndex);
            
            z = terrainHeight + 8 + rand() * 20;
            
            waypoint = [x, y, z];
            waypoint = max([1, 1, 1], min(waypoint, mapSize));
            
            idx = (j-1)*3 + 1;
            particles.cartesianPositions(i, idx:idx+2) = waypoint;
        end
        
        if isPathFeasible(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
            initialFeasibleCount = initialFeasibleCount + 1;
        end
        
        particles.velocities(i,:) = (rand(1, dims) - 0.5) * 3;
        [particles.fitness(i), fitnessComponents] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        
        particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
        particles.bestFitness(i) = particles.fitness(i);

        % Initialize global best (Always take the first one, or if better found)
        if i == 1 || particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestPosition = particles.cartesianPositions(i,:);
            globalBestComponents = fitnessComponents;
            lastImprovementIteration = 0;
        end
    end
    
    if initialFeasibleCount ~= 0
        firstFeasibleIteration = 0;
    end
    if isfield(config, 'verbose') && config.verbose
        disp(['Number of Initial feasible particles: ', num2str(initialFeasibleCount), '/', num2str(popSize)]);
    end
    
    convergenceHistory = [convergenceHistory; globalBestFitness];
    
    % Main RLAM-PSO loop (PAPER ALGORITHM 1)
    % Note: Using epsilon tolerance (1e-6) throughout to avoid floating-point precision issues
    % This ensures reward and stagnation counter are consistent with actual improvements

    % Store previous iteration's reward and nextState for training
    previousReward = 0;
    previousNextState = [];

    % Temporary experience storage for episode-level rewards
    if isfield(config, 'useEpisodeLevelRewards') && config.useEpisodeLevelRewards
        episodeExperiences = [];
    end

    for iter = 1:maxIterations
        previousGlobalBestFitness = globalBestFitness;
        currentCriticLoss = NaN;

        % ===== ONLINE NETWORK TRAINING (DL Toolbox) - 2025 GPU OPTIMIZED =====
        % Train BEFORE getPaperExactAction to avoid overwriting agent.lastState/lastAction
        % This ensures we use the correct (state, action, reward, nextState) tuple
        %
        % 2025 OPTIMIZATION: Use config parameters for flexible training frequency
        % This allows tuning the Update-To-Data (UTD) ratio for optimal GPU utilization

        % Get training parameters from config (with backward-compatible defaults)
        if isfield(config, 'enableMidEpisodeTraining') && config.enableMidEpisodeTraining
            % Use new GPU-optimized training schedule
            warmupPeriod = 50;  % Still collect initial experience
            trainingFrequency = config.trainEveryNIterations;  % From config (default: 100)

            shouldTrain = iter > warmupPeriod && ...              % After warmup
                         ~isempty(previousNextState) && ...       % Have experience
                         mod(iter, trainingFrequency) == 0;       % Every N iterations
        else
            % Legacy behavior: train every 4 iterations (backward compatible)
            warmupPeriod = 50;
            trainingFrequency = 4;

            shouldTrain = iter > warmupPeriod && ...              % After warmup
                         ~isempty(previousNextState) && ...       % Have experience
                         mod(iter, trainingFrequency) == 0;       % Every N iterations
        end

        % EPISODE-LEVEL REWARDS: Store experience instead of training
        if isfield(config, 'useEpisodeLevelRewards') && config.useEpisodeLevelRewards
            % Store experience for later (will add with episode reward)
            if ~isempty(rlamAgent.lastState) && ~isempty(previousNextState)
                exp = struct();
                exp.state = rlamAgent.lastState;
                exp.action = rlamAgent.lastAction;
                exp.nextState = previousNextState;
                exp.done = false;
                episodeExperiences = [episodeExperiences, exp];
            end
        else
            % Original behavior: train immediately with per-iteration reward
            if shouldTrain
                % Pass gradient steps parameter to agent (for multiple updates per call)
                if ~isfield(rlamAgent, 'gradientStepsPerTrainingCall') && ...
                   isfield(config, 'gradientStepsPerTrainingCall')
                    rlamAgent.gradientStepsPerTrainingCall = config.gradientStepsPerTrainingCall;
                end

                % Use DL Toolbox training with automatic differentiation
                % Uses rlamAgent.lastState/lastAction from PREVIOUS iteration (not yet overwritten)
                rlamAgent = updateAgentAuto(rlamAgent, previousNextState, [], previousReward);
                if isfield(rlamAgent, 'lastCriticLoss')
                    currentCriticLoss = rlamAgent.lastCriticLoss;
                end
            end
        end

        % ===== CALCULATE STATE (with Transformer if enabled) =====
        if config.useTransformerState
            % Use transformer state encoding (64D)
            [rlamState, ~] = calculateTransformerState(particles, iter, maxIterations, ...
                                                       lastImprovementIteration, ...
                                                       transformerEncoder, temporalBuffer);

            % Update temporal buffer for next iteration
            if size(temporalBuffer, 2) >= config.transformerTemporalWindow
                temporalBuffer = temporalBuffer(:, 2:end);
            end
            temporalBuffer = [temporalBuffer, rlamState];
        else
            % Use simple state encoding (15D)
            rlamState = calculatePaperExactState(particles, iter, maxIterations, lastImprovementIteration);
        end

        % ===== GET ACTION from actor network with exploration noise =====
        [subgroupParams, rlamAgent] = getPaperExactAction(rlamAgent, rlamState, iter);
        
        % Store average parameters for tracking
        avgW = mean(subgroupParams(:, 1));
        avgC1 = mean(subgroupParams(:, 2));
        avgC2 = mean(subgroupParams(:, 3));
        parameterHistory.w = [parameterHistory.w; avgW];
        parameterHistory.c1 = [parameterHistory.c1; avgC1];
        parameterHistory.c2 = [parameterHistory.c2; avgC2];
        parameterHistory.w_samples(iter, :) = subgroupParams(:, 1)';
        parameterHistory.c1_samples(iter, :) = subgroupParams(:, 2)';
        parameterHistory.c2_samples(iter, :) = subgroupParams(:, 3)';

        % DEBUG: Print parameters every 100 iterations
        if mod(iter, 100) == 0
            fprintf('[RLAMPSO DEBUG iter %d] w=%.4f, c1=%.4f, c2=%.4f | fitness=%.2f\n', ...
                    iter, avgW, avgC1, avgC2, globalBestFitness);
        end

        % PAPER: Update particles by 5 subgroups
        for subgroup = 1:numSubgroups
            startIdx = (subgroup - 1) * subgroupSize + 1;
            % Last subgroup includes all remaining particles to avoid orphans
            if subgroup == numSubgroups
                endIdx = popSize;
            else
                endIdx = subgroup * subgroupSize;
            end

            % PAPER-EXACT: Use subgroup-specific parameters (Equation 11, Section 4.1.2)
            % Each subgroup gets DIFFERENT parameters from the actor network
            w_sub = subgroupParams(subgroup, 1);
            c1_sub = subgroupParams(subgroup, 2);
            c2_sub = subgroupParams(subgroup, 3);
            
            % Update particles in this subgroup
            for i = startIdx:endIdx
                r1 = rand(1, dims);
                r2 = rand(1, dims);
                
                cognitiveComponent = c1_sub * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));
                socialComponent = c2_sub * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));
                
                particles.velocities(i,:) = w_sub * particles.velocities(i,:) + ...
                                           cognitiveComponent + socialComponent;
                
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
                [particles.fitness(i), fitnessComponents] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
                
                % Update personal best (with epsilon tolerance)
                epsilon_update = 1e-6;
                if (particles.bestFitness(i) - particles.fitness(i)) > epsilon_update
                    particles.bestFitness(i) = particles.fitness(i);
                    particles.bestPositions(i,:) = particles.cartesianPositions(i,:);

                    % Update global best (with epsilon tolerance)
                    if (globalBestFitness - particles.fitness(i)) > epsilon_update
                        globalBestFitness = particles.fitness(i);
                        globalBestPosition = particles.cartesianPositions(i,:);
                        globalBestComponents = fitnessComponents;
                        lastImprovementIteration = iter;
                    end
                end
            end
        end
        
        % ===== CALCULATE REWARD (IMPROVED FROM BINARY) =====
        % FIXED: Use magnitude-aware reward instead of binary
        % Paper uses binary (+1/-1), but this is suboptimal for learning
        %
        % Better approach: Continuous reward based on improvement magnitude
        % This allows network to differentiate between:
        % - Huge improvement: high reward
        % - Small improvement: moderate reward
        % - Stagnation: low penalty

        fitnessImprovement = previousGlobalBestFitness - globalBestFitness;

        % Use improved reward shaping (magnitude-aware by default)
        % Can be changed via config.rewardType = 'binary' to revert to paper
        if isfield(config, 'rewardType') && strcmp(config.rewardType, 'binary')
            % Use paper-exact binary reward
            epsilon_reward = 1e-6;
            if fitnessImprovement > epsilon_reward
                reward = 1;
            else
                reward = -1;
            end
        else
            % Use improved continuous reward shaping
            reward = calculateContinuousReward(fitnessImprovement, config);
        end

        % DEBUG: Log reward every 100 iterations
        if mod(iter, 100) == 0
            fprintf('[REWARD DEBUG iter %d] FitnessImprovement: %.4f → Reward: %+.4f (prevFit=%.2f, currFit=%.2f)\n', ...
                    iter, fitnessImprovement, reward, previousGlobalBestFitness, globalBestFitness);
        end

        % ===== CALCULATE NEXT STATE (with Transformer if enabled) =====
        if config.useTransformerState
            [nextState, ~] = calculateTransformerState(particles, iter + 1, maxIterations, ...
                                                       lastImprovementIteration, ...
                                                       transformerEncoder, temporalBuffer);
        else
            nextState = calculatePaperExactState(particles, iter + 1, maxIterations, lastImprovementIteration);
        end

        % ===== SAVE FOR NEXT ITERATION =====
        % Store reward and nextState to train with in next iteration
        previousReward = reward;
        previousNextState = nextState;

        rewardHistory = [rewardHistory; reward];
        criticLossHistory = [criticLossHistory; currentCriticLoss];

        convergenceHistory = [convergenceHistory; globalBestFitness];
        
        % Update global best (with epsilon tolerance)
        [minBestFitness, bestIdx] = min(particles.bestFitness);
        epsilon_global = 1e-6;
        if (globalBestFitness - minBestFitness) > epsilon_global
            globalBestFitness = minBestFitness;
            globalBestPosition = particles.bestPositions(bestIdx, :);
            % Re-evaluate to get components for the new global best
            [~, globalBestComponents] = evaluatePathFitness(globalBestPosition, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
            lastImprovementIteration = iter;
        end
        
        if firstFeasibleIteration == iter || (firstFeasibleIteration == 0 && firstFeasibleIteration+1 == iter)
            if isfield(config, 'verbose') && config.verbose
                disp(['First feasible path found at iteration: ', num2str(firstFeasibleIteration)]);
            end
        end
        
        % Progress reporting (only if verbose enabled)
        if isfield(config, 'verbose') && config.verbose
            if mod(iter, 50) == 0 || iter == 1 || iter == maxIterations
                stagnationDuration = iter - lastImprovementIteration;
                fprintf('[%d/%d] Fitness: %.3f | Stagnation: %d | Reward: %.1f | w=%.3f c1=%.3f c2=%.3f\n', ...
                       iter, maxIterations, globalBestFitness, stagnationDuration, reward, avgW, avgC1, avgC2);
            end
        end

        % ===== EPISODE TERMINATION CONDITIONS =====
        % Benchmarks should run the full budget, so early termination is opt-in only.
        stagnationDuration = iter - lastImprovementIteration;
        enableEarlyTermination = isfield(config, 'enableEarlyTermination') && config.enableEarlyTermination;

        if enableEarlyTermination
            % SUCCESS: Excellent fitness achieved (feasible path with good quality)
            successThreshold = 500.0;  % Empirically chosen for UAV path planning
            if globalBestFitness < successThreshold
                if isfield(config, 'verbose') && config.verbose
                    fprintf('✓ SUCCESS: Excellent fitness (%.2f) achieved at iteration %d/%d\n', ...
                            globalBestFitness, iter, maxIterations);
                end
                break;  % Early termination
            end

            % FAILURE: Prolonged stagnation without improvement
            stagnationThreshold = 150;  % 150 iterations without improvement
            if stagnationDuration > stagnationThreshold
                if isfield(config, 'verbose') && config.verbose
                    fprintf('⚠ STAGNATION: No improvement for %d iterations. Terminating at %d/%d\n', ...
                            stagnationDuration, iter, maxIterations);
                end
                break;  % Early termination
            end
        end
    end

    % ===== TRAIN WITH FINAL ITERATION'S EXPERIENCE =====
    % Don't forget to train with the last iteration's experience!
    if isfield(config, 'useEpisodeLevelRewards') && config.useEpisodeLevelRewards
        % Skip final training - will train with episode reward later
        % Add final experience to collection
        if ~isempty(rlamAgent.lastState) && ~isempty(previousNextState)
            exp = struct();
            exp.state = rlamAgent.lastState;
            exp.action = rlamAgent.lastAction;
            exp.nextState = previousNextState;
            exp.done = true;  % Mark last experience as episode end
            episodeExperiences = [episodeExperiences, exp];
        end
    else
        % Original behavior
        if ~isempty(previousNextState)
            % Ensure gradient steps parameter is set
            if ~isfield(rlamAgent, 'gradientStepsPerTrainingCall') && ...
               isfield(config, 'gradientStepsPerTrainingCall')
                rlamAgent.gradientStepsPerTrainingCall = config.gradientStepsPerTrainingCall;
            end

            % Use DL Toolbox training with automatic differentiation
            rlamAgent = trainAgentAuto(rlamAgent, previousNextState, [], previousReward);
        end
    end

    % Construct final path
    globalPath = constructPathFromPSO(globalBestPosition, startPoint, goalPoint, numWaypoints, mapSize);
    
    % Store algorithm statistics
    % ALWAYS re-evaluate the final path to ensure components match the reported fitness
    % This captures the exact breakdown of why fitness might be Inf (e.g., collision vs terrain)
    [~, globalBestComponents] = evaluatePathFitness(globalBestPosition, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
    
    algorithmSpecificStats.actualBestFitness = globalBestFitness;
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, globalBestFitness, globalBestComponents);
    algorithmSpecificStats.rewardHistory = rewardHistory;
    algorithmSpecificStats.criticLossHistory = criticLossHistory;

    % Return collected experiences for episode-level reward training
    if isfield(config, 'useEpisodeLevelRewards') && config.useEpisodeLevelRewards
        algorithmSpecificStats.episodeExperiences = episodeExperiences;
    end

    % CRITICAL FIX: Store transformer in agent for persistence across episodes
    if config.useTransformerState
        rlamAgent.transformerEncoder = transformerEncoder;
        rlamAgent.temporalBuffer = temporalBuffer;
    end

    algorithmSpecificStats.trainedAgent = rlamAgent;  % Return trained agent for reuse
    algorithmSpecificStats.config = config;  % Return config for reference

    if isfield(config, 'verbose') && config.verbose
        disp('=================================================================');
        fprintf('✓ RLAM-PSO completed with final fitness: %.3f\n', globalBestFitness);
        disp('=================================================================');
    end
end



