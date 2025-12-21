function [globalPath, convergenceHistory, algorithmSpecificStats] = globalPathPlanningDQNPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, w_initial, c1_initial, c2_initial, paramMode, pretrainedModel)
    % DQN-PSO (Deep Q-Network Enhanced PSO)
    % Reference: MDPI Applied Sciences, November 2024
    % Uses Q-learning to select from discrete parameter sets

    if nargin < 14
        paramMode = '5subgroup';
    end
    if nargin < 15
        pretrainedModel = [];
    end

    disp(['Starting DQN-PSO (Deep Q-Network Enhanced PSO) - Mode: ', paramMode]);

    numWaypoints = 5;
    dims = 3 * numWaypoints;

    firstFeasibleIteration = -1;
    initialFeasibleCount = 0;
    convergenceHistory = [];

    % Initialize particles
    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);

    globalBestPosition = zeros(1, dims);
    globalBestFitness = Inf;
    previousBestFitness = Inf;
    globalBestComponents = struct();
    globalBestComponents.pathLength = 0;
    globalBestComponents.collisionPenalty = 0;
    globalBestComponents.turningPenalty = 0;
    globalBestComponents.climbingPenalty = 0;
    globalBestComponents.heightPenalty = 0;
    globalBestComponents.terrainPenalty = 0;
    globalBestComponents.obstaclePenalty = 0;

    % DQN-PSO: Discrete parameter sets (5 states)
    parameterSets = [
        0.9, 2.5, 0.5;  % High exploration (high w, high c1, low c2)
        0.7, 2.0, 1.5;  % Balanced
        0.5, 1.5, 2.0;  % Moderate exploitation
        0.4, 1.0, 2.5;  % High exploitation (low w, low c1, high c2)
        0.6, 1.5, 1.5;  % Neutral
    ];
    numActions = size(parameterSets, 1);

    % Determine number of control entities based on paramMode
    switch paramMode
        case 'global'
            numEntities = 1;  % One set of parameters for all particles
            particlesPerEntity = popSize;
        case '5subgroup'
            numEntities = 5;  % Five groups
            particlesPerEntity = popSize / 5;
        case 'per-particle'
            numEntities = popSize;  % Each particle controlled independently
            particlesPerEntity = 1;
        otherwise
            error('Invalid paramMode: %s. Use ''global'', ''5subgroup'', or ''per-particle''', paramMode);
    end

    % Q-tables: One per entity (global=1, 5subgroup=5, per-particle=40)
    % Each Q-table is 5 states × 5 actions
    numStates = 5;
    Q = zeros(numStates, numActions, numEntities);

    % Q-learning parameters
    alpha = 0.1;     % Learning rate
    gamma = 0.9;     % Discount factor
    epsilon = 0.2;   % Exploration rate
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
                warning('Pretrained DQN-PSO model not found: %s. Continuing with random initialization.', modelPath);
            end
        elseif isstruct(pretrainedModel)
            modelStruct = pretrainedModel;
            pretrainedInfo.source = 'struct';
        end

        if ~isempty(modelStruct)
            if isfield(modelStruct, 'Q')
                loadedQ = modelStruct.Q;
                % Verify Q-table dimensions match current paramMode
                if ndims(loadedQ) == 2
                    % Old format: 2D Q-table, assume global mode
                    if numEntities == 1
                        Q(:,:,1) = loadedQ;
                    else
                        warning('Pretrained Q-table is 2D (global), but current mode is %s. Using replicated Q-table.', paramMode);
                        for e = 1:numEntities
                            Q(:,:,e) = loadedQ;
                        end
                    end
                elseif ndims(loadedQ) == 3 && size(loadedQ, 3) == numEntities
                    Q = loadedQ;
                else
                    warning('Pretrained Q-table dimensions mismatch. Expected [%d x %d x %d], got [%s]. Using random initialization.', ...
                        numStates, numActions, numEntities, mat2str(size(loadedQ)));
                end
            end
            if isfield(modelStruct, 'alpha')
                alpha = modelStruct.alpha;
            end
            if isfield(modelStruct, 'gamma')
                gamma = modelStruct.gamma;
            end
            if isfield(modelStruct, 'epsilon')
                epsilon = modelStruct.epsilon;
            end
            pretrainedInfo.used = true;
        end
    end

    % Current action and state per entity
    currentAction = 2 * ones(numEntities, 1); % Start with balanced parameters
    currentState = 3 * ones(numEntities, 1);  % Start with moderate state

    % Initialize parameter matrix: [popSize x 3] for [w, c1, c2]
    particleParams = zeros(popSize, 3);
    for e = 1:numEntities
        startIdx = (e - 1) * particlesPerEntity + 1;
        endIdx = e * particlesPerEntity;
        particleParams(startIdx:endIdx, 1) = parameterSets(currentAction(e), 1);  % w
        particleParams(startIdx:endIdx, 2) = parameterSets(currentAction(e), 2);  % c1
        particleParams(startIdx:endIdx, 3) = parameterSets(currentAction(e), 3);  % c2
    end

    % Vectorized initialization
    randMatrix = rand(popSize, dims);
    xIndices = 1:3:dims;
    yIndices = 2:3:dims;
    zIndices = 3:3:dims;

    particles.cartesianPositions(:, xIndices) = randMatrix(:, xIndices) * mapSize(1);
    particles.cartesianPositions(:, yIndices) = randMatrix(:, yIndices) * mapSize(2);
    particles.cartesianPositions(:, zIndices) = 10 + randMatrix(:, zIndices) * (mapSize(3) - 10);
    particles.velocities = (rand(popSize, dims) - 0.5) * 10;

    % Initial fitness evaluation
    % Initialize global best with the first particle to ensure valid structures even if all are Inf
    [particles.fitness(1), fitnessComponents] = evaluatePathFitness(particles.cartesianPositions(1,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
    particles.bestPositions(1,:) = particles.cartesianPositions(1,:);
    particles.bestFitness(1) = particles.fitness(1);
    
    globalBestFitness = particles.fitness(1);
    globalBestPosition = particles.cartesianPositions(1,:);
    globalBestComponents = fitnessComponents;
    
    if isPathFeasible(particles.cartesianPositions(1,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
        initialFeasibleCount = 1;
    end

    % Evaluate remaining particles
    for i = 2:popSize
        [particles.fitness(i), fitnessComponents] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
        particles.bestFitness(i) = particles.fitness(i);

        if particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestPosition = particles.cartesianPositions(i,:);
            globalBestComponents = fitnessComponents;
        end

        if isPathFeasible(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
            initialFeasibleCount = initialFeasibleCount + 1;
        end
    end

    if initialFeasibleCount ~= 0
        firstFeasibleIteration = 0;
    end
    disp(['Number of Initial feasible particles: ', num2str(initialFeasibleCount), '/', num2str(popSize)]);

    convergenceHistory = [convergenceHistory; globalBestFitness];
    previousBestFitness = globalBestFitness * ones(numEntities, 1);  % Track per-entity best fitness

    % Main DQN-PSO loop
    for iter = 1:maxIterations
        % Calculate state and update Q-tables every 10 iterations
        if mod(iter, 10) == 0
            % Update Q-learning per entity
            for e = 1:numEntities
                % Get particles belonging to this entity
                startIdx = (e - 1) * particlesPerEntity + 1;
                endIdx = e * particlesPerEntity;
                entityParticles = particles.cartesianPositions(startIdx:endIdx, :);
                entityFitness = particles.fitness(startIdx:endIdx);

                % Compute diversity for this entity's particles
                avgPosition = mean(entityParticles, 1);
                diversity = 0;
                for p = 1:particlesPerEntity
                    diversity = diversity + norm(entityParticles(p,:) - avgPosition);
                end
                diversity = diversity / particlesPerEntity;

                % Compute fitness improvement for this entity
                currentBestFitness = min(entityFitness);
                fitnessImprovement = previousBestFitness(e) - currentBestFitness;

                % Classify state (1-5)
                % State 1: High diversity + good improvement (exploration working)
                % State 2: High diversity + low improvement (need exploitation)
                % State 3: Moderate diversity (balanced)
                % State 4: Low diversity + good improvement (exploitation working)
                % State 5: Low diversity + low improvement (stagnation, need exploration)

                diversityThreshold = 10.0;
                improvementThreshold = 0.1;

                if diversity > diversityThreshold
                    if fitnessImprovement > improvementThreshold
                        nextState = 1; % High diversity, improving
                    else
                        nextState = 2; % High diversity, not improving
                    end
                elseif diversity > diversityThreshold/2
                    nextState = 3; % Moderate diversity
                else
                    if fitnessImprovement > improvementThreshold
                        nextState = 4; % Low diversity, improving
                    else
                        nextState = 5; % Low diversity, stagnating
                    end
                end

                % Calculate reward (fitness improvement)
                reward = fitnessImprovement;

                % Q-learning update for this entity's Q-table
                maxQNext = max(Q(nextState, :, e));
                Q(currentState(e), currentAction(e), e) = Q(currentState(e), currentAction(e), e) + ...
                    alpha * (reward + gamma * maxQNext - Q(currentState(e), currentAction(e), e));

                % Select next action (epsilon-greedy)
                if rand() < epsilon
                    % Explore: random action
                    nextAction = randi(numActions);
                else
                    % Exploit: best action for this entity
                    [~, nextAction] = max(Q(nextState, :, e));
                end

                % Update parameters for this entity's particles
                particleParams(startIdx:endIdx, 1) = parameterSets(nextAction, 1);  % w
                particleParams(startIdx:endIdx, 2) = parameterSets(nextAction, 2);  % c1
                particleParams(startIdx:endIdx, 3) = parameterSets(nextAction, 3);  % c2

                % Update state and action for next iteration
                currentState(e) = nextState;
                currentAction(e) = nextAction;
                previousBestFitness(e) = currentBestFitness;
            end
        end

        % Standard PSO update with learned parameters (per-particle)
        for i = 1:popSize
            r1 = rand(1, dims);
            r2 = rand(1, dims);

            % Get parameters for this particle
            w_i = particleParams(i, 1);
            c1_i = particleParams(i, 2);
            c2_i = particleParams(i, 3);

            particles.velocities(i,:) = w_i * particles.velocities(i,:) + ...
                                       c1_i * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) + ...
                                       c2_i * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));

            % Velocity clamping
            maxVelocity = 20.0;
            for d = 1:dims
                if abs(particles.velocities(i,d)) > maxVelocity
                    particles.velocities(i,d) = sign(particles.velocities(i,d)) * maxVelocity;
                end
            end

            % Position update
            particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);

            % Boundary handling
            for j = 1:numWaypoints
                idx = (j-1)*3 + 1;
                particles.cartesianPositions(i, idx) = max(0, min(particles.cartesianPositions(i, idx), mapSize(1)));
                particles.cartesianPositions(i, idx+1) = max(0, min(particles.cartesianPositions(i, idx+1), mapSize(2)));
                particles.cartesianPositions(i, idx+2) = max(10, min(particles.cartesianPositions(i, idx+2), mapSize(3)));
            end

            % Evaluate fitness
            [particles.fitness(i), fitnessComponents] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

            % Update personal best
            if particles.fitness(i) < particles.bestFitness(i)
                particles.bestFitness(i) = particles.fitness(i);
                particles.bestPositions(i,:) = particles.cartesianPositions(i,:);

                % Update global best
                if particles.fitness(i) < globalBestFitness
                    globalBestFitness = particles.fitness(i);
                    globalBestPosition = particles.cartesianPositions(i,:);
                    globalBestComponents = fitnessComponents;
                end
            end
        end

        convergenceHistory = [convergenceHistory; globalBestFitness];

        % Track first feasible solution
        if firstFeasibleIteration == -1
            for i = 1:popSize
                if isPathFeasible(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
                    firstFeasibleIteration = iter;
                    break;
                end
            end
        end

        if firstFeasibleIteration == -1
        elseif firstFeasibleIteration == iter || (firstFeasibleIteration == 0 && firstFeasibleIteration+1 == iter)
            disp(['First feasible path found at iteration: ', num2str(firstFeasibleIteration)]);
        end

        if mod(iter, 50) == 0
            % Show parameter diversity statistics
            w_mean = mean(particleParams(:,1));
            w_std = std(particleParams(:,1));
            c1_mean = mean(particleParams(:,2));
            c1_std = std(particleParams(:,2));
            c2_mean = mean(particleParams(:,3));
            c2_std = std(particleParams(:,3));

            if numEntities == 1
                % Global mode: show single state/action
                disp(sprintf('DQN-PSO iter %d/%d | Mode: %s | Fitness: %.2f | State: %d | Action: %d | w=%.3f c1=%.3f c2=%.3f', ...
                    iter, maxIterations, paramMode, globalBestFitness, currentState(1), currentAction(1), w_mean, c1_mean, c2_mean));
            else
                % Multi-entity: show stats across entities
                disp(sprintf('DQN-PSO iter %d/%d | Mode: %s | Fitness: %.2f | w: %.3f±%.3f | c1: %.3f±%.3f | c2: %.3f±%.3f', ...
                    iter, maxIterations, paramMode, globalBestFitness, w_mean, w_std, c1_mean, c1_std, c2_mean, c2_std));
            end
        end
    end

    % Construct final path
    globalPath = constructPathFromPSO(globalBestPosition, startPoint, goalPoint, numWaypoints, mapSize);

    % Apply terrain constraints
    for i = 1:size(globalPath, 1)
        x = globalPath(i,1);
        y = globalPath(i,2);

        [~, xIndex] = min(abs(terrainX(1,:) - x));
        [~, yIndex] = min(abs(terrainY(:,1) - y));

        minHeight = terrainGrid(yIndex, xIndex) + 8;
        globalPath(i,3) = max(globalPath(i,3), minHeight);
    end

    % Return statistics
    % ALWAYS re-evaluate the final path to ensure components match the reported fitness
    [~, globalBestComponents] = evaluatePathFitness(globalBestPosition, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

    algorithmSpecificStats.actualBestFitness = globalBestFitness;
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, globalBestFitness, globalBestComponents);
    algorithmSpecificStats.learnedQTable = Q;
    algorithmSpecificStats.qLearningState = struct( ...
        'alpha', alpha, ...
        'gamma', gamma, ...
        'epsilon', epsilon, ...
        'paramMode', paramMode, ...
        'pretrainedUsed', pretrainedInfo.used, ...
        'pretrainedSource', pretrainedInfo.source);

    disp(['DQN-PSO completed with final fitness: ', num2str(globalBestFitness)]);
end
