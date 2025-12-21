function [globalPath, convergenceHistory, parameterHistory, algorithmSpecificStats] = globalPathPlanningPSOTVAC(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, initialW, initialC1, initialC2)
    % PSO-TVAC: PSO with Time-Variant Acceleration Coefficients (c1, c2)
    % Acceleration coefficients vary with time; inertia decreases linearly 0.9 -> 0.1.
    disp('Starting PSO-TVAC (time-variant acceleration coefficients)...');

    numWaypoints = 5;
    dims = 3 * numWaypoints;

    firstFeasibleIteration = -1;
    initialFeasibleCount = 0;
    convergenceHistory = [];

    % Parameter tracking
    parameterHistory = struct();
    parameterHistory.w = [];
    parameterHistory.c1 = [];
    parameterHistory.c2 = [];

    % Track fitness components of the best solution
    globalBestComponents = struct();
    globalBestComponents.pathLength = 0;
    globalBestComponents.collisionPenalty = 0;
    globalBestComponents.turningPenalty = 0;
    globalBestComponents.climbingPenalty = 0;
    globalBestComponents.heightPenalty = 0;
    globalBestComponents.terrainPenalty = 0;
    globalBestComponents.obstaclePenalty = 0;

    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);
    particles.lastImprovement = ones(popSize, 1);
    particles.clusterAssignment = ones(popSize, 1);
    particles.stabilityFactor = ones(popSize, dims);

    globalBestPosition = zeros(1, dims);
    globalBestFitness = Inf;

    % TVAC ranges (typical bounds)
    % w is scheduled inside the adapter (0.9 -> 0.1)
    c1Min = 0.5; c1Max = 2.5;     % cognitive coef bounds
    c2Min = 0.5; c2Max = 2.5;     % social coef bounds

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

        if particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestPosition = particles.cartesianPositions(i,:);
            globalBestComponents = fitnessComponents;
        end
    end

    if initialFeasibleCount ~= 0
        firstFeasibleIteration = 0;
    end
    disp(['Number of Initial feasible particles: ', num2str(initialFeasibleCount), '/', num2str(popSize)]);

    % Store initial values
    convergenceHistory = [convergenceHistory; globalBestFitness];
    % Initial w (start of schedule)
    parameterHistory.w = [parameterHistory.w; 0.9];
    % For TVAC start: c1 at max, c2 at min
    parameterHistory.c1 = [parameterHistory.c1; c1Max];
    parameterHistory.c2 = [parameterHistory.c2; c2Min];

    % Main loop
    for iter = 1:maxIterations
        % Time-varying coefficients
        [current_w, current_c1, current_c2] = adaptParametersTVAC(0, 0, 0, ...
            iter, maxIterations, 0, c1Min, c1Max, c2Min, c2Max);

        % Track parameter values
        parameterHistory.w = [parameterHistory.w; current_w];
        parameterHistory.c1 = [parameterHistory.c1; current_c1];
        parameterHistory.c2 = [parameterHistory.c2; current_c2];

        for i = 1:popSize
            r1 = rand(1, dims);
            r2 = rand(1, dims);

            cognitiveComponent = current_c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));
            socialComponent = current_c2 * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));

            particles.velocities(i,:) = current_w * particles.velocities(i,:) + ...
                      cognitiveComponent + socialComponent;

            % Velocity clamping (relative to map scale)
            maxVelocity = 0.15 * (mapSize(1) + mapSize(2) + mapSize(3)) / 3;
            for j = 1:numWaypoints
                idx = (j-1)*3 + 1;
                segmentVel = particles.velocities(i, idx:idx+2);
                velMag = norm(segmentVel);
                if velMag > maxVelocity
                    particles.velocities(i, idx:idx+2) = segmentVel * (maxVelocity / velMag);
                end
            end

            % Position update
            particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);

            % Boundary + terrain height handling
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

            % Update personal and global bests
            if particles.fitness(i) < particles.bestFitness(i)
                particles.bestFitness(i) = particles.fitness(i);
                particles.bestPositions(i,:) = particles.cartesianPositions(i,:);

                if particles.fitness(i) < globalBestFitness
                    globalBestFitness = particles.fitness(i);
                    globalBestPosition = particles.cartesianPositions(i,:);
                    globalBestComponents = fitnessComponents;  
                end
            end
        end

        % Convergence trace
        convergenceHistory = [convergenceHistory; globalBestFitness];

        % First feasible reporting
        if firstFeasibleIteration == -1
            for i = 1:popSize
                if isPathFeasible(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
                    firstFeasibleIteration = iter;
                    break;
                end
            end
        end

        if mod(iter, 50) == 0 || iter == 1 || iter == maxIterations
            disp(['PSO-TVAC iteration: ', num2str(iter), '/', num2str(maxIterations), ...
                  ' - Best fitness: ', num2str(globalBestFitness), ...
                  ' - w=', num2str(current_w), ', c1=', num2str(current_c1), ', c2=', num2str(current_c2)]);
        end
    end

    % Construct final path
    globalPath = constructPathFromPSO(globalBestPosition, startPoint, goalPoint, numWaypoints, mapSize);

    % Ensure terrain constraint on final path
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
end
