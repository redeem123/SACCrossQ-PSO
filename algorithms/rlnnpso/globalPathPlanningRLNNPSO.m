function [globalPath, convergenceHistory, algorithmSpecificStats] = globalPathPlanningRLNNPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, fixedW, fixedC1, fixedC2, pretrainedNetworkPath)
    % RL-NNPSO: Reinforcement Learning based Neural-Guided PSO
    % Uses TD3 (Twin Delayed DDPG) for velocity guidance with particle-level rewards
    %
    % Key Features:
    % - 3-feature state (iteration, diversity, stagnation) → 15-D sin-encoded (like RLAM-PSO)
    % - 15-D action space (velocity corrections)
    % - Particle-level rewards (not global)
    % - Experience replay with TD3 training
    % - Adaptive blending (RL guidance + standard PSO)
    % - Diversity-aware training and stagnation recovery
    %
    % Additional input (optional):
    %   pretrainedNetworkPath: Path to pre-trained TD3 networks (default: empty, random init)

    if nargin < 14
        pretrainedNetworkPath = '';  % Default: no pre-trained networks
    end

    if ~isempty(pretrainedNetworkPath)
        disp('Starting RL-NNPSO with PRE-TRAINED TD3 networks...');
    else
        disp('Starting RL-NNPSO with RANDOM TD3 networks...');
    end

    numWaypoints = 5;
    dims = 3 * numWaypoints;  % 15 dimensions

    firstFeasibleIteration = -1;
    initialFeasibleCount = 0;
    convergenceHistory = zeros(maxIterations + 1, 1);  % Preallocate (+1 for initial value)
    convergenceIdx = 1;  % Track current index

    % Initialize RL networks (TD3: Actor + 2 Critics + Targets)
    rlNetwork = initializeRLNetwork(dims, pretrainedNetworkPath);

    % Store initial exploration noise for stability (prevents unbounded growth)
    initialExplorationNoise = rlNetwork.explorationNoise;

    % PAPER METHODOLOGY: ALWAYS train online during execution
    % Pretrained networks provide initialization from past experience, then continue learning
    % This matches RLAM-PSO paper: "online...adaptation method" that uses both
    % "current situation" (online training) AND "past experience" (pretraining)

    % PSO particle initialization
    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = Inf(popSize, 1);
    particles.prevBestFitness = Inf(popSize, 1);  % For reward calculation
    particles.fitness = Inf(popSize, 1);
    particles.lastImprovement = ones(popSize, 1);
    particles.prevFitness = Inf(popSize, 1);  % For reward calculation

    % Global best tracking
    globalBestComponents = struct();
    globalBestComponents.pathLength = 0;
    globalBestComponents.collisionPenalty = 0;
    globalBestComponents.turningPenalty = 0;
    globalBestComponents.climbingPenalty = 0;
    globalBestComponents.heightPenalty = 0;
    globalBestComponents.terrainPenalty = 0;
    globalBestComponents.obstaclePenalty = 0;
    globalBestPosition = zeros(1, dims);
    globalBestFitness = Inf;
    lastImprovementIteration = 0;  % Track last iteration with global improvement

    % PSO parameters
    current_w = fixedW;
    current_c1 = fixedC1;
    current_c2 = fixedC2;

    % Training metrics
    avgReward = 0;
    avgGamma = 0;

    % =========================================================================
    % PARTICLE INITIALIZATION
    % =========================================================================
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
        particles.prevBestFitness(i) = particles.fitness(i);
        particles.prevFitness(i) = particles.fitness(i);

        if particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestPosition = particles.cartesianPositions(i,:);
            globalBestComponents = fitnessComponents;
        end
    end

    if initialFeasibleCount ~= 0
        firstFeasibleIteration = 0;
    end
    disp(['Initial feasible particles: ', num2str(initialFeasibleCount), '/', num2str(popSize)]);

    convergenceHistory(convergenceIdx) = globalBestFitness;
    convergenceIdx = convergenceIdx + 1;

    % =========================================================================
    % MAIN RL-NNPSO LOOP
    % =========================================================================
    for iter = 1:maxIterations
        % Calculate environment state for feature extraction
        environmentState = calculateEnvironmentState(particles, iter, maxIterations, globalBestPosition);
        environmentState.convergenceHistory = convergenceHistory;

        % Calculate swarm diversity (before particle updates)
        % Normalize by dimension-specific map sizes
        positionStd = std(particles.cartesianPositions, 0, 1);
        % Create normalization vector: [mapSize(1), mapSize(2), mapSize(3)] repeated for each waypoint
        normVector = repmat(mapSize, 1, numWaypoints);
        normalizedStd = positionStd ./ normVector;
        swarmDiversity = mean(normalizedStd);

        iterRewards = [];
        iterGammas = [];

        % ---------------------------------------------------------------------
        % FOR EACH PARTICLE: RL-GUIDED PSO UPDATE
        % ---------------------------------------------------------------------
        for i = 1:popSize
            % Save previous values for reward calculation
            particles.prevBestFitness(i) = particles.bestFitness(i);
            particles.prevFitness(i) = particles.fitness(i);

            % 1. EXTRACT PARTICLE-SPECIFIC STATE (sin-encoded to 15-D)
            % State: [distToGlobalBest, velocityMagnitude, personalStagnation] → sin-encoded
            state = extractRLNNPSOState(particles, iter, maxIterations, lastImprovementIteration, i, globalBestPosition, mapSize);

            % 2. GET RL VELOCITY GUIDANCE
            % LIKE RLAM-PSO: RL always participates (no warmup, no conditional gating)

            % Normalize state
            normalizedState = (state - rlNetwork.inputMean) ./ (rlNetwork.inputStd + 1e-8);
            normalizedState(~isfinite(normalizedState)) = 0;
            normalizedState = max(-3, min(3, normalizedState));

            % Get action from actor network
            [action, ~, ~, ~] = forwardPassActor(rlNetwork.actor, normalizedState);

            % Ornstein-Uhlenbeck exploration noise (correlated)
            dt = 1.0;
            rlNetwork.ou_state = rlNetwork.ou_state - rlNetwork.ou_theta * rlNetwork.ou_state * dt + ...
                                 rlNetwork.ou_sigma * sqrt(dt) * randn(size(rlNetwork.ou_state));

            % Scale OU noise by current exploration noise level
            noise = rlNetwork.ou_state * rlNetwork.explorationNoise;
            noisyAction = max(-1, min(1, action + noise));

            % Convert to velocity guidance
            maxVelocityCorrection = 3.0;
            velocityGuidance = noisyAction * maxVelocityCorrection;

            % SIMPLIFIED GAMMA: Like RLAM-PSO, RL always participates with fixed blending
            % Fixed 50% RL influence (like RLAM always applies its parameters)
            gamma = 0.5;
            alpha = 1.0 - gamma;

            % PSO components
            r1 = rand(1, dims);
            r2 = rand(1, dims);
            cognitive = current_c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));
            social = current_c2 * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));

            % BLENDED VELOCITY UPDATE (RL + PSO)
            particles.velocities(i,:) = current_w * particles.velocities(i,:) + ...
                                        alpha * (cognitive + social) + ...
                                        gamma * velocityGuidance';

            iterGammas = [iterGammas; gamma];

            % 3. VELOCITY CLAMPING
            maxVelocity = 0.15 * (mapSize(1) + mapSize(2) + mapSize(3)) / 3;
            for j = 1:numWaypoints
                idx = (j-1)*3 + 1;
                segmentVel = particles.velocities(i, idx:idx+2);
                velMag = norm(segmentVel);

                if velMag > maxVelocity
                    particles.velocities(i, idx:idx+2) = segmentVel * (maxVelocity / velMag);
                end
            end

            % 4. UPDATE POSITION
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

            % 5. EVALUATE NEW FITNESS
            [particles.fitness(i), fitnessComponents] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

            % 6. CALCULATE NEXT STATE (after position update)
            nextState = extractRLNNPSOState(particles, iter, maxIterations, lastImprovementIteration, i, globalBestPosition, mapSize);

            % 7. CALCULATE REWARD (fitness-based with proper personal best tracking)
            reward = calculateRLNNPSOReward(particles.prevFitness(i), particles.fitness(i), ...
                particles.prevBestFitness(i), particles.bestFitness(i), ...
                particles.lastImprovement(i), swarmDiversity);

            iterRewards = [iterRewards; reward];

            % 8. STORE EXPERIENCE IN REPLAY BUFFER
            isDone = (iter == maxIterations);  % Mark terminal state
            rlNetwork = addToReplayBuffer(rlNetwork, state, noisyAction, reward, nextState, isDone);

            % 9. UPDATE PERSONAL AND GLOBAL BESTS
            if particles.fitness(i) < particles.bestFitness(i)
                particles.bestFitness(i) = particles.fitness(i);
                particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
                particles.lastImprovement(i) = 0;

                if particles.fitness(i) < globalBestFitness
                    globalBestFitness = particles.fitness(i);
                    globalBestPosition = particles.cartesianPositions(i,:);
                    globalBestComponents = fitnessComponents;
                    lastImprovementIteration = iter;  % Update last improvement iteration
                end
            else
                particles.lastImprovement(i) = particles.lastImprovement(i) + 1;
            end
        end

        convergenceHistory(convergenceIdx) = globalBestFitness;
        convergenceIdx = convergenceIdx + 1;

        % ---------------------------------------------------------------------
        % TD3 NETWORK TRAINING
        % ALWAYS train online (matches paper methodology)
        % Pretrained networks continue learning during execution
        % ---------------------------------------------------------------------
        trainNetwork = length(rlNetwork.replayBuffer) >= rlNetwork.batchSize;

        if trainNetwork
            numTrainSteps = 2;  % Gradient steps per iteration (reduced for performance)

            for trainStep = 1:numTrainSteps
                % Sample minibatch
                batch = sampleReplayBatch(rlNetwork, rlNetwork.batchSize);

                % Normalize states
                normalizedStates = (batch.states - rlNetwork.inputMean) ./ (rlNetwork.inputStd + 1e-8);
                normalizedStates(~isfinite(normalizedStates)) = 0;

                normalizedNextStates = (batch.nextStates - rlNetwork.inputMean) ./ (rlNetwork.inputStd + 1e-8);
                normalizedNextStates(~isfinite(normalizedNextStates)) = 0;

                % TD3: Compute target Q-values with target policy smoothing
                [targetActions, ~, ~, ~] = forwardPassActor(rlNetwork.targetActor, normalizedNextStates);

                % Add noise to target actions (TD3 smoothing)
                targetNoise = randn(size(targetActions)) * rlNetwork.targetNoiseStd;
                targetNoise = max(-rlNetwork.targetNoiseClip, min(rlNetwork.targetNoiseClip, targetNoise));
                targetActions = max(-1, min(1, targetActions + targetNoise));

                % Compute target Q-values using MINIMUM of two critics
                [targetQ1, ~, ~, ~] = forwardPassCritic(rlNetwork.targetCritic1, normalizedNextStates, targetActions);
                [targetQ2, ~, ~, ~] = forwardPassCritic(rlNetwork.targetCritic2, normalizedNextStates, targetActions);
                targetQ = min(targetQ1, targetQ2);

                % TD target: r + γ * min(Q1', Q2')
                yTarget = batch.rewards + rlNetwork.gamma * targetQ .* (1 - batch.dones);

                % UPDATE CRITIC 1
                [currentQ1, h1_c1, h2_c1, h3_c1] = forwardPassCritic(rlNetwork.critic1, normalizedStates, batch.actions);
                criticLoss1 = mean((currentQ1 - yTarget).^2);

                % Backprop through critic1 (simplified inline)
                rlNetwork.critic1 = updateCriticSimple(rlNetwork.critic1, normalizedStates, batch.actions, ...
                    currentQ1, yTarget, h1_c1, h2_c1, h3_c1, rlNetwork.criticLR);

                % UPDATE CRITIC 2
                [currentQ2, h1_c2, h2_c2, h3_c2] = forwardPassCritic(rlNetwork.critic2, normalizedStates, batch.actions);
                criticLoss2 = mean((currentQ2 - yTarget).^2);

                rlNetwork.critic2 = updateCriticSimple(rlNetwork.critic2, normalizedStates, batch.actions, ...
                    currentQ2, yTarget, h1_c2, h2_c2, h3_c2, rlNetwork.criticLR);

                % TD3: DELAYED ACTOR UPDATE (every 2 critic updates)
                if mod(trainStep, rlNetwork.policyDelay) == 0
                    [actorActions, h1_a, h2_a, h3_a] = forwardPassActor(rlNetwork.actor, normalizedStates);

                    % Actor loss: -Q(s, μ(s))
                    [qValues, ~, ~, ~] = forwardPassCritic(rlNetwork.critic1, normalizedStates, actorActions);
                    actorLoss = -mean(qValues);

                    % Update actor (simplified inline)
                    rlNetwork.actor = updateActorSimple(rlNetwork.actor, rlNetwork.critic1, ...
                        normalizedStates, actorActions, h1_a, h2_a, h3_a, rlNetwork.actorLR);

                    % SOFT UPDATE TARGET NETWORKS
                    rlNetwork = softUpdateTargets(rlNetwork);
                end

                % Store losses
                rlNetwork.critic1LossHistory = [rlNetwork.critic1LossHistory, criticLoss1];
                rlNetwork.critic2LossHistory = [rlNetwork.critic2LossHistory, criticLoss2];
            end

            % Update normalization stats
            rlNetwork = updateNormalizationStatsOnline(rlNetwork, batch.states);
        end

        % Decay exploration noise
        rlNetwork.explorationNoise = max(rlNetwork.minNoise, ...
            rlNetwork.explorationNoise * rlNetwork.noiseDecay);

        % Calculate metrics
        if ~isempty(iterRewards)
            avgReward = mean(iterRewards);
            rlNetwork.rewardHistory = [rlNetwork.rewardHistory, avgReward];
        end
        if ~isempty(iterGammas)
            avgGamma = mean(iterGammas);
        end

        % Check for first feasible
        if firstFeasibleIteration == -1
            for i = 1:popSize
                if isPathFeasible(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
                    firstFeasibleIteration = iter;
                    break;
                end
            end
        end

        if firstFeasibleIteration == iter || (firstFeasibleIteration == 0 && firstFeasibleIteration+1 == iter)
            disp(['First feasible path found at iteration: ', num2str(firstFeasibleIteration)]);
        end

        % Progress reporting
        if mod(iter, 50) == 0 || iter == 1 || iter == maxIterations
            bufferSize = length(rlNetwork.replayBuffer);
            fprintf('RL-NNPSO Iter %d/%d | Fit: %.3f | Rew: %.2f | Gam: %.3f | Div: %.3f | Buf: %d | Noise: %.4f\n', ...
                iter, maxIterations, globalBestFitness, avgReward, avgGamma, swarmDiversity, bufferSize, rlNetwork.explorationNoise);
        end
    end

    % =========================================================================
    % CONSTRUCT FINAL PATH
    % =========================================================================
    globalPath = constructPathFromPSO(globalBestPosition, startPoint, goalPoint, numWaypoints);

    % Store algorithm statistics
    if ~exist('globalBestComponents', 'var')
        [~, globalBestComponents] = evaluatePathFitness(globalBestPosition, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
    end
    algorithmSpecificStats.actualBestFitness = globalBestFitness;
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, globalBestFitness, globalBestComponents);

    disp(['RL-NNPSO completed with final fitness: ', num2str(globalBestFitness)]);
    disp(['Total experiences collected: ', num2str(length(rlNetwork.replayBuffer))]);
end
