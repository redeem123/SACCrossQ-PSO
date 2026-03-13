function [globalPath, convergenceHistory, parameterHistory, algorithmSpecificStats] = globalPathPlanningSAEPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, c_min, c_max, w_min, w_max, useOtherImprovements)
    % SAEPSO: Two-mode implementation
    % useOtherImprovements = true:  Full SAEPSO (Spherical, Tent map, OBL, EP, Acceleration)
    % useOtherImprovements = false: PSO with SAEPSO parameter control only (Cartesian)

    % Default to full version if not specified
    if nargin < 14
        useOtherImprovements = true;
    end

    if useOtherImprovements
        disp('Starting SAEPSO-FULL (All improvements enabled)...');
    else
        disp('Starting SAEPSO-ParamsOnly (Parameter control strategy only)...');
    end

    numWaypoints = 5;
    dims = 3 * numWaypoints; 
    
    convergenceHistory = [];
    parameterHistory = struct();
    parameterHistory.w = [];
    parameterHistory.c1 = [];
    parameterHistory.c2 = [];
    parameterHistory.w_samples = [];
    parameterHistory.c1_samples = [];
    parameterHistory.c2_samples = [];

    % SAEPSO parameters from paper
    psi1 = 2; psi2 = 100; % Tent map parameters
    psi_evolution = 3; % Evolutionary parameter
    psi2_acceleration = 0.01; % Local optima detection threshold

    % ========== MODE-SPECIFIC INITIALIZATION ==========
    if useOtherImprovements
        % ===== FULL SAEPSO MODE: Spherical coordinates + Tent map + OBL =====

        % Define spherical coordinate bounds as per paper
        maxSegmentLength = norm(goalPoint - startPoint) / numWaypoints * 2;
        lowerBounds = [];
        upperBounds = [];
        for i = 1:numWaypoints
            lowerBounds = [lowerBounds, 0, -pi/2, 0];
            upperBounds = [upperBounds, maxSegmentLength, pi/2, 2*pi];
        end

        % STAGE 1: Initialize using Tent Map with perturbation (Equation 39)
        disp('  Stage 1: Tent map initialization in spherical coordinates...');
        initialPopulation = zeros(popSize, dims);
        for i = 1:popSize
            for j = 1:dims
                gamma_t = rand(); % Initial seed
                r1 = rand(); r2 = rand(); r3 = rand();

                % Apply tent map with perturbation (Equation 39)
                if 0 < gamma_t && gamma_t < 0.5
                    gamma_next = psi1 * gamma_t + (2*r1 - 1)/psi2;
                elseif 0.5 < gamma_t && gamma_t < 1
                    gamma_next = psi1 * (1 - gamma_t) + (2*r2 - 1)/psi2;
                else
                    gamma_next = r3;
                end

                gamma_next = max(0, min(1, gamma_next));

                % Map to spherical bounds (Equation 40)
                initialPopulation(i, j) = lowerBounds(j) + ...
                    (upperBounds(j) - lowerBounds(j)) * gamma_next;
            end
        end

        % STAGE 2: Opposition-based learning (Equation 41)
        disp('  Stage 2: Opposition-based learning...');
        oppositePopulation = zeros(popSize, dims);
        for i = 1:popSize
            for j = 1:dims
                oppositePopulation(i, j) = lowerBounds(j) + upperBounds(j) - initialPopulation(i, j);
            end
        end

        % Combine and select best b individuals
        combinedPopulation = [initialPopulation; oppositePopulation];
        combinedFitness = zeros(2*popSize, 1);

        for i = 1:(2*popSize)
            cartesianPos = sphericalToCartesian(combinedPopulation(i, :), startPoint, numWaypoints);
            % CRITICAL FIX: Clamp to map bounds before fitness evaluation
            % This ensures fitness reflects the ACTUAL path that will be used (not out-of-bounds path)
            cartesianPos = clampWaypointsToMapBounds(cartesianPos, mapSize, numWaypoints);
            [combinedFitness(i), ~] = evaluatePathFitness(cartesianPos, startPoint, goalPoint, ...
                dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        end

        [~, sortedIndices] = sort(combinedFitness);
        selectedIndices = sortedIndices(1:popSize);

    else
        % ===== PARAMS-ONLY MODE: Cartesian coordinates + Random initialization =====

        % Define Cartesian coordinate bounds (standard PSO)
        lowerBounds = repmat([1, 1, 1], 1, numWaypoints);
        upperBounds = repmat(mapSize, 1, numWaypoints);

        disp('  Initialization: Random Cartesian coordinates...');

        % Random initialization in Cartesian space
        initialPopulation = zeros(popSize, dims);
        for i = 1:popSize
            for j = 1:dims
                initialPopulation(i, j) = lowerBounds(j) + (upperBounds(j) - lowerBounds(j)) * rand();
            end
        end

        % No OBL - evaluate only initial population
        combinedPopulation = initialPopulation;
        combinedFitness = zeros(popSize, 1);

        for i = 1:popSize
            [combinedFitness(i), ~] = evaluatePathFitness(combinedPopulation(i, :), startPoint, goalPoint, ...
                dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        end

        selectedIndices = 1:popSize;
    end

    % ========== COMMON INITIALIZATION ==========
    % Initialize particles structure
    particles = struct();
    particles.positions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);
    particles.fitnessHistory = zeros(popSize, 3); % For local optima detection

    if useOtherImprovements
        particles.eta = ones(popSize, dims); % For evolutionary programming (full mode only)
    end

    globalBestPosition = zeros(1, dims);
    globalBestFitness = Inf;

    % Initialize with selected solutions
    for i = 1:popSize
        idx = selectedIndices(i);
        particles.positions(i, :) = combinedPopulation(idx, :);
        particles.fitness(i) = combinedFitness(idx);
        particles.fitnessHistory(i, :) = particles.fitness(i);
        particles.bestPositions(i, :) = particles.positions(i, :);
        particles.bestFitness(i) = particles.fitness(i);

        % Initial velocities
        if useOtherImprovements
            % Spherical coordinates: smaller range
            particles.velocities(i, :) = (rand(1, dims) - 0.5) * 2.0;
        else
            % Cartesian coordinates: standard range
            particles.velocities(i, :) = (rand(1, dims) - 0.5) .* (upperBounds - lowerBounds) * 0.1;
        end

        if particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestPosition = particles.positions(i, :);
        end
    end
    
    avgFitnessHistory = zeros(3, 1);
    avgFitnessHistory(:) = sum(particles.fitness) / popSize;
    convergenceHistory = [convergenceHistory; globalBestFitness];
    
    % STAGE 3: Main SAEPSO loop
    disp('  Stage 3: Main optimization loop...');
    for iter = 1:maxIterations
        iterW = zeros(1, popSize);
        iterC1 = zeros(1, popSize);
        iterC2 = zeros(1, popSize);

        % Calculate average fitness F_avge (Equation 42)
        avgPopulationFitness = sum(particles.fitness) / popSize;
        avgFitnessHistory = [avgFitnessHistory(2:end); avgPopulationFitness];
        
        % Calculate adaptive sine factor (Equation 44)
        chi = 0.5 * sin((iter/maxIterations) * pi - pi/2) + 0.5;
        
        % FOR EACH PARTICLE (Algorithm 1: lines 5-25)
        for i = 1:popSize
            % Calculate adaptive fitness factor fap (Equation 42)
            if abs(avgPopulationFitness - globalBestFitness) > eps
                fap = (particles.fitness(i) - globalBestFitness) / (avgPopulationFitness - globalBestFitness);
            else
                fap = 0;
            end
            fap = max(0, min(1, fap));
            
            % Calculate adaptive parameters (Equations 43-44)
            w = w_max - (w_max - w_min) * (iter/maxIterations) * fap;
            c1 = c_max - (c_max - c_min) * chi * fap;
            c2 = c_min + (c_max - c_min) * chi * fap;

            iterW(i) = w;
            iterC1(i) = c1;
            iterC2(i) = c2;
            
            r1 = rand(1, dims);
            r2 = rand(1, dims);
            
            % Velocity update components
            cognitiveComponent = c1 .* r1 .* (particles.bestPositions(i,:) - particles.positions(i,:));
            socialComponent = c2 .* r2 .* (globalBestPosition - particles.positions(i,:));

            % Check for local optima and calculate acceleration (Equations 45-46)
            acceleration = zeros(1, dims);
            if useOtherImprovements && iter >= 3
                isTrapped = particles.fitnessHistory(i,1) > avgFitnessHistory(1) && ...
                           particles.fitnessHistory(i,2) > avgFitnessHistory(2) && ...
                           particles.fitnessHistory(i,3) > avgFitnessHistory(3) && ...
                           abs(particles.fitnessHistory(i,1) - particles.fitnessHistory(i,2)) < psi2_acceleration && ...
                           abs(particles.fitnessHistory(i,2) - particles.fitnessHistory(i,3)) < psi2_acceleration;

                if isTrapped
                    r4 = rand();
                    acceleration = r4 * (globalBestPosition - particles.positions(i,:)) * (iter/maxIterations) * fap;
                end
            end

            % Velocity update (Equation 47)
            particles.velocities(i,:) = w * particles.velocities(i,:) + cognitiveComponent + socialComponent + acceleration;

            % Velocity clamping
            for j = 1:dims
                maxVel = (upperBounds(j) - lowerBounds(j)) * 0.2; % 20% of range
                particles.velocities(i,j) = max(-maxVel, min(maxVel, particles.velocities(i,j)));
            end

            % Position update
            particles.positions(i,:) = particles.positions(i,:) + particles.velocities(i,:);

            % Apply bounds
            for j = 1:dims
                particles.positions(i,j) = max(lowerBounds(j), min(upperBounds(j), particles.positions(i,j)));
            end

            % ========== EVOLUTIONARY PROGRAMMING (FULL MODE ONLY) ==========
            if useOtherImprovements
                % Update η (Equation 48) - Standard ES/EP formula
                % τ' = 1/√(2n) for global learning rate (smaller, affects all dimensions)
                % τ = 1/√(2√n) for individual learning rate (larger, per-dimension exploration)
                for j = 1:dims
                    N0 = randn();
                    Ni = randn();
                    % FIXED: Swapped coefficients to match standard ES/EP theory
                    % Global term (N0) uses smaller coefficient (τ' = 0.408)
                    % Individual term (Ni) uses larger coefficient (τ = 0.537)
                    particles.eta(i,j) = particles.eta(i,j) * exp((sqrt(2*psi_evolution))^(-1) * N0 + ...
                        (sqrt(2*sqrt(psi_evolution)))^(-1) * Ni);
                end

                % Generate mutations
                r_gauss = randn(1, dims);
                u_gauss = particles.positions(i,:) + particles.eta(i,:) .* r_gauss;

                r_cauchy = tan(pi * (rand(1, dims) - 0.5));
                u_cauchy = particles.positions(i,:) + particles.eta(i,:) .* r_cauchy;

                % Apply bounds to mutations
                for j = 1:dims
                    u_gauss(j) = max(lowerBounds(j), min(upperBounds(j), u_gauss(j)));
                    u_cauchy(j) = max(lowerBounds(j), min(upperBounds(j), u_cauchy(j)));
                end

                % Evaluate all three candidates (spherical mode uses conversion)
                cartesianCurrent = sphericalToCartesian(particles.positions(i,:), startPoint, numWaypoints);
                cartesianGauss = sphericalToCartesian(u_gauss, startPoint, numWaypoints);
                cartesianCauchy = sphericalToCartesian(u_cauchy, startPoint, numWaypoints);

                % CRITICAL FIX: Clamp to map bounds before fitness evaluation
                cartesianCurrent = clampWaypointsToMapBounds(cartesianCurrent, mapSize, numWaypoints);
                cartesianGauss = clampWaypointsToMapBounds(cartesianGauss, mapSize, numWaypoints);
                cartesianCauchy = clampWaypointsToMapBounds(cartesianCauchy, mapSize, numWaypoints);

                [fitness_current, ~] = evaluatePathFitness(cartesianCurrent, startPoint, goalPoint, ...
                    dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
                [fitness_gauss, ~] = evaluatePathFitness(cartesianGauss, startPoint, goalPoint, ...
                    dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
                [fitness_cauchy, ~] = evaluatePathFitness(cartesianCauchy, startPoint, goalPoint, ...
                    dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

                % Select best among parent, Gaussian, and Cauchy
                if fitness_gauss < fitness_current && fitness_gauss <= fitness_cauchy
                    particles.positions(i,:) = u_gauss;
                    particles.fitness(i) = fitness_gauss;
                elseif fitness_cauchy < fitness_current && fitness_cauchy < fitness_gauss
                    particles.positions(i,:) = u_cauchy;
                    particles.fitness(i) = fitness_cauchy;
                else
                    particles.fitness(i) = fitness_current;
                end

            else
                % ========== PARAMS-ONLY MODE: Just evaluate current position ==========
                [particles.fitness(i), ~] = evaluatePathFitness(particles.positions(i,:), startPoint, goalPoint, ...
                    dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
            end
            
            % Update fitness history
            particles.fitnessHistory(i,:) = [particles.fitnessHistory(i,2:end), particles.fitness(i)];
            
            % Update personal best (lines 20-21)
            if particles.fitness(i) < particles.bestFitness(i)
                particles.bestFitness(i) = particles.fitness(i);
                particles.bestPositions(i,:) = particles.positions(i,:);
            end
            
            % Update global best (lines 22-23)
            if particles.fitness(i) < globalBestFitness
                globalBestFitness = particles.fitness(i);
                globalBestPosition = particles.positions(i,:);
            end
        end % End of particle loop
        
        convergenceHistory = [convergenceHistory; globalBestFitness];
        
        % Store the actual particle-wise parameter values and their means.
        parameterHistory.w = [parameterHistory.w; mean(iterW)];
        parameterHistory.c1 = [parameterHistory.c1; mean(iterC1)];
        parameterHistory.c2 = [parameterHistory.c2; mean(iterC2)];
        parameterHistory.w_samples = [parameterHistory.w_samples; iterW];
        parameterHistory.c1_samples = [parameterHistory.c1_samples; iterC1];
        parameterHistory.c2_samples = [parameterHistory.c2_samples; iterC2];
        
        if mod(iter, 50) == 0
            disp(['Iteration: ', num2str(iter), '/', num2str(maxIterations), ...
                  ' - Best fitness: ', num2str(globalBestFitness)]);
        end
    end
    
    % ========== FINAL PATH CONSTRUCTION ==========
    if useOtherImprovements
        % Full mode: Convert from spherical to Cartesian
        cartesianBest = sphericalToCartesian(globalBestPosition, startPoint, numWaypoints);
        disp(['SAEPSO-FULL completed with final fitness: ', num2str(globalBestFitness)]);
    else
        % Params-only mode: Already in Cartesian
        cartesianBest = globalBestPosition;
        disp(['SAEPSO-ParamsOnly completed with final fitness: ', num2str(globalBestFitness)]);
    end

    % CRITICAL FIX: Clamp cartesianBest BEFORE path construction to ensure consistency
    % This ensures the returned path matches the fitness evaluation constraints (Z >= 10)
    cartesianBest = clampWaypointsToMapBounds(cartesianBest, mapSize, numWaypoints);

    globalPath = constructPathFromPSO(cartesianBest, startPoint, goalPoint, numWaypoints, mapSize);

    % Store statistics with fitness components
    [finalFitness, globalBestComponents] = evaluatePathFitness(cartesianBest, startPoint, goalPoint, ...
        dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

    algorithmSpecificStats = struct();
    algorithmSpecificStats.actualBestFitness = finalFitness;  % FIXED: Use recalculated fitness, not optimization loop fitness
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, finalFitness, globalBestComponents);
end

function clampedPos = clampWaypointsToMapBounds(cartesianPos, mapSize, numWaypoints)
    % Clamp waypoints to map boundaries
    % CRITICAL FIX: Ensures fitness is evaluated on actual achievable path (not out-of-bounds)
    %
    % This solves the problem where:
    % - Spherical coords create waypoints at Z=524 (way out of bounds)
    % - Fitness evaluated on high-flying path (unrealistic low terrain penalty)
    % - Final path clamped to Z=100 goes through terrain
    %
    % Now we clamp BEFORE fitness evaluation, so optimization sees real constraints

    clampedPos = cartesianPos;
    for i = 1:numWaypoints
        idx = (i-1)*3 + 1;
        % Clamp X coordinate to [0, mapSize(1)]
        clampedPos(idx) = max(0, min(clampedPos(idx), mapSize(1)));
        % Clamp Y coordinate to [0, mapSize(2)]
        clampedPos(idx+1) = max(0, min(clampedPos(idx+1), mapSize(2)));
        % Clamp Z coordinate to [10, mapSize(3)]
        clampedPos(idx+2) = max(10, min(clampedPos(idx+2), mapSize(3)));
    end
end
