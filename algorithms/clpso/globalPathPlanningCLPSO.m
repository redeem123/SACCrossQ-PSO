function [globalPath, convergenceHistory, algorithmSpecificStats] = globalPathPlanningCLPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, w, c)
    % CLPSO (Comprehensive Learning PSO) - Liang et al., IEEE TEVC 2006
    % Particles learn from different exemplars for different dimensions

    disp('Starting CLPSO (Comprehensive Learning PSO) path planning...');

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

    % CLPSO-specific: Learning exemplar for each particle and dimension
    learningExemplar = zeros(popSize, dims);
    lastUpdateIteration = zeros(popSize, 1);

    % Refreshing gap for each particle (m parameter)
    m = 7; % Standard CLPSO parameter

    % Learning probability Pc for each particle (PAPER FORMULA)
    % Pc(i) = 0.5 * (exp(5.0*i/(n-1)) - exp(0)) / (exp(5.0) - exp(0))
    Pc = zeros(popSize, 1);
    pc_indices = 5.0 * (0:popSize-1) / (popSize-1);
    Pc = 0.5 * (exp(pc_indices) - exp(0)) / (exp(5.0) - exp(0));
    Pc = Pc(:); % Ensure column vector

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

    % Initialize learning exemplars
    for i = 1:popSize
        for d = 1:dims
            if rand() < Pc(i)
                % Select random particle as exemplar (tournament selection based on fitness)
                idx1 = randi(popSize);
                idx2 = randi(popSize);
                if particles.bestFitness(idx1) < particles.bestFitness(idx2)
                    learningExemplar(i, d) = idx1;
                else
                    learningExemplar(i, d) = idx2;
                end
            else
                % Use own pbest
                learningExemplar(i, d) = i;
            end
        end
    end

    convergenceHistory = [convergenceHistory; globalBestFitness];

    % CLPSO inertia weight: linearly decreasing from 0.9 to 0.1 (comparison bounds)
    w_max = 0.9;
    w_min = 0.1;

    % Main CLPSO loop
    for iter = 1:maxIterations
        % Update inertia weight for this iteration (linearly decreasing)
        w_current = w_max - (w_max - w_min) * iter / maxIterations;

        for i = 1:popSize
            % Check if need to refresh learning exemplars (stagnation detection)
            if (iter - lastUpdateIteration(i)) > m
                % Refresh exemplars for this particle
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

            % CLPSO velocity update: learn from different exemplars per dimension
            exemplarPosition = zeros(1, dims);
            for d = 1:dims
                exemplarIdx = learningExemplar(i, d);
                exemplarPosition(d) = particles.bestPositions(exemplarIdx, d);
            end

            r = rand(1, dims);
            % Use w_current (linearly decreasing inertia weight)
            particles.velocities(i,:) = w_current * particles.velocities(i,:) + ...
                                       c * r .* (exemplarPosition - particles.cartesianPositions(i,:));

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
                lastUpdateIteration(i) = iter; % Reset stagnation counter

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
            disp(['CLPSO iteration: ', num2str(iter), '/', num2str(maxIterations), ...
                  ' - Best fitness: ', num2str(globalBestFitness), ...
                  ' - w: ', num2str(w_current, '%.3f')]);
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

    disp(['CLPSO completed with final fitness: ', num2str(globalBestFitness)]);
end
