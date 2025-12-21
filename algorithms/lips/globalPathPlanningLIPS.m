function [globalPath, convergenceHistory, algorithmSpecificStats] = globalPathPlanningLIPS(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, w, c)
    % LIPS (Learning PSO with Heterogeneous Social Interactions)
    % Zhao et al., IEEE TEVC 2021
    % Three learning modes: Competition, Cooperation, Neutral

    disp('Starting LIPS (Learning PSO with Heterogeneous Interactions) path planning...');

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
    globalBestComponents = struct();
    globalBestComponents.pathLength = 0;
    globalBestComponents.collisionPenalty = 0;
    globalBestComponents.turningPenalty = 0;
    globalBestComponents.climbingPenalty = 0;
    globalBestComponents.heightPenalty = 0;
    globalBestComponents.terrainPenalty = 0;
    globalBestComponents.obstaclePenalty = 0;

    % LIPS-specific: Learning mode for each particle
    % Modes: 1=Competition, 2=Cooperation, 3=Neutral
    learningMode = ones(popSize, 1); % Initialize all to Competition
    modePerformance = zeros(popSize, 3); % Track performance of each mode
    modeUsageCount = ones(popSize, 3); % Count how many times each mode is used

    % Epsilon-greedy for mode selection
    epsilon = 0.1;

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
    for i = 1:popSize
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
    previousFitness = particles.fitness;

    % Main LIPS loop
    for iter = 1:maxIterations
        for i = 1:popSize
            % Select learning mode (epsilon-greedy)
            if rand() < epsilon
                % Explore: random mode
                learningMode(i) = randi(3);
            else
                % Exploit: best performing mode
                avgPerformance = modePerformance(i,:) ./ max(modeUsageCount(i,:), 1);
                [~, learningMode(i)] = max(avgPerformance);
            end

            mode = learningMode(i);
            modeUsageCount(i, mode) = modeUsageCount(i, mode) + 1;

            % Apply learning strategy based on mode
            if mode == 1
                % Competition mode: Learn from better particles
                betterIndices = find(particles.bestFitness < particles.bestFitness(i));
                if ~isempty(betterIndices)
                    exemplarIdx = betterIndices(randi(length(betterIndices)));
                    exemplar = particles.bestPositions(exemplarIdx, :);
                else
                    exemplar = globalBestPosition;
                end

                r1 = rand(1, dims);
                r2 = rand(1, dims);
                particles.velocities(i,:) = w * particles.velocities(i,:) + ...
                                           c * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) + ...
                                           c * r2 .* (exemplar - particles.cartesianPositions(i,:));

            elseif mode == 2
                % Cooperation mode: Learn from neighborhood average
                K = 5; % Neighborhood size
                neighbors = zeros(K, 1);
                for k = 1:K
                    neighbors(k) = mod(i - 1 + k, popSize) + 1;
                end
                avgNeighborPos = mean(particles.cartesianPositions(neighbors, :), 1);

                r = rand(1, dims);
                particles.velocities(i,:) = w * particles.velocities(i,:) + ...
                                           c * r .* (avgNeighborPos - particles.cartesianPositions(i,:));

            else
                % Neutral mode: No social learning (only cognitive)
                r = rand(1, dims);
                particles.velocities(i,:) = w * particles.velocities(i,:) + ...
                                           c * r .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));
            end

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

            % Update mode performance based on fitness improvement
            fitnessImprovement = previousFitness(i) - particles.fitness(i);
            if fitnessImprovement > 0
                modePerformance(i, mode) = modePerformance(i, mode) + fitnessImprovement;
            end

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

        previousFitness = particles.fitness;
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
            disp(['LIPS iteration: ', num2str(iter), '/', num2str(maxIterations), ' - Best fitness: ', num2str(globalBestFitness)]);
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
    if ~exist('globalBestComponents', 'var')
        [~, globalBestComponents] = evaluatePathFitness(globalBestPosition, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
    end
    algorithmSpecificStats.actualBestFitness = globalBestFitness;
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, globalBestFitness, globalBestComponents);

    disp(['LIPS completed with final fitness: ', num2str(globalBestFitness)]);
end
