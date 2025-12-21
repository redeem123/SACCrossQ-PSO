function [globalPath, convergenceHistory, algorithmSpecificStats] = globalPathPlanningFIPS(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, w, c1, c2)
    % FIPS (Fully Informed Particle Swarm) - Mendes et al., 2004
    % Each particle is influenced by all its neighbors, not just global best

    disp('Starting FIPS (Fully Informed PSO) path planning...');

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

    % Vectorized initialization
    randMatrix = rand(popSize, dims);
    xIndices = 1:3:dims;
    yIndices = 2:3:dims;
    zIndices = 3:3:dims;

    particles.cartesianPositions(:, xIndices) = randMatrix(:, xIndices) * mapSize(1);
    particles.cartesianPositions(:, yIndices) = randMatrix(:, yIndices) * mapSize(2);
    particles.cartesianPositions(:, zIndices) = 10 + randMatrix(:, zIndices) * (mapSize(3) - 10);
    particles.velocities = (rand(popSize, dims) - 0.5) * 10;

    % Create ring topology (each particle connected to K neighbors)
    K = 3; % Number of neighbors (FIPS typically uses 3)
    topology = zeros(popSize, K);
    for i = 1:popSize
        for k = 1:K
            neighborIdx = mod(i - 1 + k, popSize) + 1;
            topology(i, k) = neighborIdx;
        end
    end

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

    % FIPS parameter (constriction coefficient)
    phi = c1 + c2; % Total acceleration coefficient
    chi = 2 / abs(2 - phi - sqrt(phi^2 - 4*phi)); % Constriction coefficient

    % Main FIPS loop
    for iter = 1:maxIterations
        for i = 1:popSize
            % FIPS: Average influence from all neighbors (including self)
            avgInfluence = zeros(1, dims);

            for k = 1:K
                neighborIdx = topology(i, k);
                r = rand(1, dims);
                avgInfluence = avgInfluence + r .* (particles.bestPositions(neighborIdx,:) - particles.cartesianPositions(i,:));
            end

            % Include self influence
            r_self = rand(1, dims);
            avgInfluence = avgInfluence + r_self .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));

            % Average over all neighbors (K + 1 including self)
            avgInfluence = avgInfluence / (K + 1);

            % FIPS velocity update with constriction
            particles.velocities(i,:) = chi * (particles.velocities(i,:) + phi * avgInfluence);

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
            disp(['FIPS iteration: ', num2str(iter), '/', num2str(maxIterations), ' - Best fitness: ', num2str(globalBestFitness)]);
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

    disp(['FIPS completed with final fitness: ', num2str(globalBestFitness)]);
end
