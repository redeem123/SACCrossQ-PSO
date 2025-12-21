function [globalPath, convergenceHistory, algorithmSpecificStats] = globalPathPlanningDMSPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, w, c1, c2)
    % DMS-PSO (Dynamic Multi-Swarm PSO) - Liang & Suganthan, 2005
    % Uses multiple sub-swarms with dynamic regrouping

    disp('Starting DMS-PSO (Dynamic Multi-Swarm PSO) path planning...');

    numWaypoints = 5;
    dims = 3 * numWaypoints;

    firstFeasibleIteration = -1;
    initialFeasibleCount = 0;
    convergenceHistory = [];

    % DMS-PSO parameters
    numSwarms = 4; % Number of sub-swarms
    swarmSize = floor(popSize / numSwarms);
    regroupPeriod = 5; % Regroup every R iterations

    % Initialize particles
    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);
    particles.swarmID = zeros(popSize, 1); % Which sub-swarm each particle belongs to

    % Sub-swarm best positions and fitness
    swarmBestPosition = zeros(numSwarms, dims);
    swarmBestFitness = Inf(numSwarms, 1);

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

    % Vectorized initialization
    randMatrix = rand(popSize, dims);
    xIndices = 1:3:dims;
    yIndices = 2:3:dims;
    zIndices = 3:3:dims;

    particles.cartesianPositions(:, xIndices) = randMatrix(:, xIndices) * mapSize(1);
    particles.cartesianPositions(:, yIndices) = randMatrix(:, yIndices) * mapSize(2);
    particles.cartesianPositions(:, zIndices) = 10 + randMatrix(:, zIndices) * (mapSize(3) - 10);
    particles.velocities = (rand(popSize, dims) - 0.5) * 10;

    % Assign particles to sub-swarms initially
    for i = 1:popSize
        particles.swarmID(i) = mod(i-1, numSwarms) + 1;
    end

    % Initial fitness evaluation
    for i = 1:popSize
        [particles.fitness(i), fitnessComponents] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
        particles.bestFitness(i) = particles.fitness(i);

        % Update swarm best
        swarmID = particles.swarmID(i);
        if particles.fitness(i) < swarmBestFitness(swarmID)
            swarmBestFitness(swarmID) = particles.fitness(i);
            swarmBestPosition(swarmID, :) = particles.cartesianPositions(i,:);
        end

        % Update global best
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

    % Main DMS-PSO loop
    for iter = 1:maxIterations
        % Dynamic regrouping: every R iterations
        if mod(iter, regroupPeriod) == 0
            % Sort particles by fitness
            [~, sortedIndices] = sort(particles.bestFitness);

            % Reassign to sub-swarms (round-robin of sorted particles)
            for i = 1:popSize
                particles.swarmID(sortedIndices(i)) = mod(i-1, numSwarms) + 1;
            end

            % Recalculate swarm bests after regrouping
            swarmBestFitness = Inf(numSwarms, 1);
            for i = 1:popSize
                swarmID = particles.swarmID(i);
                if particles.bestFitness(i) < swarmBestFitness(swarmID)
                    swarmBestFitness(swarmID) = particles.bestFitness(i);
                    swarmBestPosition(swarmID, :) = particles.bestPositions(i,:);
                end
            end
        end

        for i = 1:popSize
            % Get sub-swarm best for this particle
            swarmID = particles.swarmID(i);
            localBest = swarmBestPosition(swarmID, :);

            % PSO velocity update using sub-swarm best (not global best)
            r1 = rand(1, dims);
            r2 = rand(1, dims);

            particles.velocities(i,:) = w * particles.velocities(i,:) + ...
                                       c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) + ...
                                       c2 * r2 .* (localBest - particles.cartesianPositions(i,:));

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

                % Update swarm best
                if particles.fitness(i) < swarmBestFitness(swarmID)
                    swarmBestFitness(swarmID) = particles.fitness(i);
                    swarmBestPosition(swarmID, :) = particles.cartesianPositions(i,:);
                end

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
            disp(['DMS-PSO iteration: ', num2str(iter), '/', num2str(maxIterations), ' - Best fitness: ', num2str(globalBestFitness)]);
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

    disp(['DMS-PSO completed with final fitness: ', num2str(globalBestFitness)]);
end
