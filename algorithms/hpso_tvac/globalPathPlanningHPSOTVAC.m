function [globalPath, convergenceHistory, algorithmSpecificStats] = globalPathPlanningHPSOTVAC(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, w, c1_initial, c2_initial)
    % HPSO-TVAC (Heterogeneous PSO with Time-Varying Acceleration Coefficients)
    % Nickabadi et al., 2011
    % Dynamically adjusts parameters and re-initializes particles to prevent premature convergence

    disp('Starting HPSO-TVAC path planning...');

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

    % HPSO-TVAC parameters
    successCount = 0;
    failureCount = 0;
    c1 = c1_initial;
    c2 = c2_initial;

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
    previousBestFitness = globalBestFitness;

    % Main HPSO-TVAC loop
    for iter = 1:maxIterations
        % Check for improvement or stagnation
        if globalBestFitness < previousBestFitness
            successCount = successCount + 1;
            failureCount = 0;
        else
            failureCount = failureCount + 1;
            successCount = 0;
        end

        % Adaptive parameter adjustment (TVAC mechanism)
        if successCount > 5
            % Increase exploitation (increase c1, decrease c2)
            c1 = c1 * 1.05;
            c2 = c2 * 0.95;
            successCount = 0;
        elseif failureCount > 5
            % Increase exploration (decrease c1, increase c2)
            c1 = c1 * 0.95;
            c2 = c2 * 1.05;
            failureCount = 0;
        end

        % Clamp acceleration coefficients
        c1 = max(1.5, min(c1, 2.5));
        c2 = max(1.5, min(c2, 2.5));

        previousBestFitness = globalBestFitness;

        for i = 1:popSize
            % Re-initialization strategy: if particle stagnates, re-initialize
            if particles.bestFitness(i) == particles.fitness(i) && rand() < 0.05
                % Re-initialize position
                randVec = rand(1, dims);
                particles.cartesianPositions(i, xIndices) = randVec(xIndices) * mapSize(1);
                particles.cartesianPositions(i, yIndices) = randVec(yIndices) * mapSize(2);
                particles.cartesianPositions(i, zIndices) = 10 + randVec(zIndices) * (mapSize(3) - 10);
                particles.velocities(i,:) = (rand(1, dims) - 0.5) * 10;
            else
                % Standard PSO velocity update with adaptive c1, c2
                r1 = rand(1, dims);
                r2 = rand(1, dims);

                particles.velocities(i,:) = w * particles.velocities(i,:) + ...
                                           c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) + ...
                                           c2 * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));

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
            disp(['HPSO-TVAC iteration: ', num2str(iter), '/', num2str(maxIterations), ' - Best fitness: ', num2str(globalBestFitness), ' - c1: ', num2str(c1), ' - c2: ', num2str(c2)]);
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

    disp(['HPSO-TVAC completed with final fitness: ', num2str(globalBestFitness)]);
end
