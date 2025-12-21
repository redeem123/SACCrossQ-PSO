function [globalPath, convergenceHistory, parameterHistory, algorithmSpecificStats] = globalPathPlanningUAPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, cMin, cMax)
%GLOBALPATHPLANNINGUAPSO Implementation of the Unique Adaptive PSO (UAPSO)
%   Isiet & Gadala, Applied Soft Computing, 2019.
%   Each particle adapts its inertia weight and acceleration coefficients
%   through the evolutionary state feedback described in the paper.

    if nargin < 12
        cMax = 4.0;
    end
    if nargin < 11
        cMin = 0.0;
    end

    disp('Starting UAPSO global path planning...');

    numWaypoints = 5;
    dims = 3 * numWaypoints;

    % Tracking variables for diagnostics
    firstFeasibleIteration = -1;
    initialFeasibleCount = 0;

    % Particle containers
    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);

    latestFeasibleFitness = ones(popSize, 1); % f(x_f) in the paper

    % Global best bookkeeping
    globalBestPosition = zeros(1, dims);
    globalBestFitness = Inf;
    globalBestComponents = struct();

    % Random initialisation of positions and velocities
    randMatrix = rand(popSize, dims);
    xIdx = 1:3:dims;
    yIdx = 2:3:dims;
    zIdx = 3:3:dims;

    particles.cartesianPositions(:, xIdx) = randMatrix(:, xIdx) * mapSize(1);
    particles.cartesianPositions(:, yIdx) = randMatrix(:, yIdx) * mapSize(2);
    particles.cartesianPositions(:, zIdx) = 10 + randMatrix(:, zIdx) * (mapSize(3) - 10);
    particles.velocities = (rand(popSize, dims) - 0.5) * 10;

    % Evaluate initial population
    numFeasible = 0;
    for i = 1:popSize
        [particles.fitness(i), fitnessComponents] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        particles.bestFitness(i) = particles.fitness(i);
        particles.bestPositions(i,:) = particles.cartesianPositions(i,:);

        if particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestPosition = particles.cartesianPositions(i,:);
            globalBestComponents = fitnessComponents;
        end

        if isPathFeasible(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
            numFeasible = numFeasible + 1;
            latestFeasibleFitness(i) = max(abs(particles.fitness(i)), 1e-9);
        else
            latestFeasibleFitness(i) = max(abs(particles.fitness(i)), 1e-9);
        end
    end

    initialFeasibleCount = numFeasible;
    if numFeasible > 0
        firstFeasibleIteration = 0;
    end
    disp(['Initial feasible particles: ', num2str(numFeasible), '/', num2str(popSize)]);

    % Histories
    convergenceHistory = zeros(maxIterations + 1, 1);
    convergenceHistory(1) = globalBestFitness;
    avgInertiaHistory = zeros(maxIterations, 1);
    avgC1History = zeros(maxIterations, 1);
    avgC2History = zeros(maxIterations, 1);
    avgESHistory = zeros(maxIterations, 1);

    % Main optimisation loop
    for iter = 1:maxIterations
        inertiaValues = zeros(popSize, 1);
        c1Values = zeros(popSize, 1);
        c2Values = zeros(popSize, 1);
        esValues = zeros(popSize, 1);
        tvac = ((cMax - cMin) * (maxIterations - iter)) / maxIterations;

        for i = 1:popSize
            feasibleDen = max(abs(latestFeasibleFitness(i)), 1e-9);
            es = (particles.bestFitness(i) - globalBestFitness) / feasibleDen;
            es = max(0, min(1, es));

            inertia = es;
            if es <= 0.5
                c1 = tvac + cMin;
                c2 = cMax - tvac;
            else
                c1 = cMax - tvac;
                c2 = tvac + cMin;
            end
            c1 = min(max(c1, cMin), cMax);
            c2 = min(max(c2, cMin), cMax);

            r1 = rand(1, dims);
            r2 = rand(1, dims);

            cognitive = c1 .* r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));
            social = c2 .* r2 .* (globalBestPosition - particles.cartesianPositions(i,:));
            repulsion = (1 - es) .* (globalBestPosition - particles.bestPositions(i,:));

            particles.velocities(i,:) = inertia .* particles.velocities(i,:) + cognitive + social - repulsion;

            maxVelocity = 20.0;
            particles.velocities(i,:) = max(min(particles.velocities(i,:), maxVelocity), -maxVelocity);

            particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);

            for j = 1:numWaypoints
                idx = (j-1)*3 + 1;
                particles.cartesianPositions(i, idx) = min(max(particles.cartesianPositions(i, idx), 0), mapSize(1));
                particles.cartesianPositions(i, idx+1) = min(max(particles.cartesianPositions(i, idx+1), 0), mapSize(2));
                particles.cartesianPositions(i, idx+2) = min(max(particles.cartesianPositions(i, idx+2), 10), mapSize(3));
            end

            [particles.fitness(i), fitnessComponents] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

            isFeasible = isPathFeasible(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
            if isFeasible
                latestFeasibleFitness(i) = max(abs(particles.fitness(i)), 1e-9);
            end

            if particles.fitness(i) < particles.bestFitness(i)
                particles.bestFitness(i) = particles.fitness(i);
                particles.bestPositions(i,:) = particles.cartesianPositions(i,:);

                if particles.fitness(i) < globalBestFitness
                    globalBestFitness = particles.fitness(i);
                    globalBestPosition = particles.cartesianPositions(i,:);
                    globalBestComponents = fitnessComponents;
                end
            end

            inertiaValues(i) = inertia;
            c1Values(i) = c1;
            c2Values(i) = c2;
            esValues(i) = es;
        end

        convergenceHistory(iter + 1) = globalBestFitness;
        avgInertiaHistory(iter) = mean(inertiaValues);
        avgC1History(iter) = mean(c1Values);
        avgC2History(iter) = mean(c2Values);
        avgESHistory(iter) = mean(esValues);

        if firstFeasibleIteration == -1
            for i = 1:popSize
                if isPathFeasible(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
                    firstFeasibleIteration = iter;
                    break;
                end
            end
        end

        if mod(iter, 50) == 0
            disp(['UAPSO iteration ', num2str(iter), '/', num2str(maxIterations), ' - Best fitness: ', num2str(globalBestFitness)]);
        end
    end

    convergenceHistory = convergenceHistory(1:maxIterations+1);

    globalPath = constructPathFromPSO(globalBestPosition, startPoint, goalPoint, numWaypoints, mapSize);
    for k = 1:size(globalPath, 1)
        x = globalPath(k,1);
        y = globalPath(k,2);
        [~, xIndex] = min(abs(terrainX(1,:) - x));
        [~, yIndex] = min(abs(terrainY(:,1) - y));
        minHeight = terrainGrid(yIndex, xIndex) + 8;
        globalPath(k,3) = max(globalPath(k,3), minHeight);
    end

    if ~exist('globalBestComponents', 'var') || isempty(fieldnames(globalBestComponents))
        [~, globalBestComponents] = evaluatePathFitness(globalBestPosition, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
    end

    algorithmSpecificStats = struct();
    algorithmSpecificStats.actualBestFitness = globalBestFitness;
    algorithmSpecificStats.firstFeasibleIteration = firstFeasibleIteration;
    algorithmSpecificStats.initialFeasibleCount = initialFeasibleCount;
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, globalBestFitness, globalBestComponents);

    parameterHistory = struct();
    parameterHistory.avgInertia = avgInertiaHistory;
    parameterHistory.avgC1 = avgC1History;
    parameterHistory.avgC2 = avgC2History;
    parameterHistory.avgEvolutionaryState = avgESHistory;

    disp(['UAPSO completed with final fitness: ', num2str(globalBestFitness)]);
end
