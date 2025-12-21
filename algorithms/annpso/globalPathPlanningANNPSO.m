function [globalPath, convergence, annStats, algorithmSpecificStats] = globalPathPlanningANNPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, w, c1, c2, hiddenNeurons)
    % ANN-PSO: Fully consistent with Benhalem & Lones (2020) methodology
    
    fprintf('Starting ANN-PSO (Fully Consistent with Benhalem & Lones 2020)...\n');
    
    % Algorithm parameters
    numWaypoints = 5;
    dimensions = 3 * numWaypoints;  % Problem dimensionality
    numInformants = 5;  % Should be configurable parameter 'nf' from paper
    
    % Define bounds for path planning  
    lowerBounds = repmat([0, 0, 5], 1, numWaypoints);
    upperBounds = repmat(mapSize, 1, numWaypoints);
    
    % Initialize swarm
    positions = zeros(popSize, dimensions);
    velocities = zeros(popSize, dimensions);
    personalBest = zeros(popSize, dimensions);
    personalBestFitness = inf(popSize, 1);
    
    % Initialize ANN for each particle (PAPER ALGORITHM 2, lines 7-8)
    particleANNs = cell(popSize, 1);
    
    for i = 1:popSize
        % Random initialization within bounds
        positions(i, :) = lowerBounds + (upperBounds - lowerBounds) .* rand(1, dimensions);
        velocities(i, :) = (rand(1, dimensions) - 0.5) * 2.5; % Paper mentions velocity initialization
        personalBest(i, :) = positions(i, :);
        
        % Initialize ANN (PAPER: anni â†? random neural network, fitness(anni) â†? 0)
        particleANNs{i} = initializeConsistentANN(dimensions, hiddenNeurons);
    end
    
    % Assign informants randomly (PAPER ALGORITHM 1, line 9: "Randomly select nf particles as informants")
    informants = cell(popSize, 1);
    for i = 1:popSize
        availableParticles = setdiff(1:popSize, i); % Exclude self
        if length(availableParticles) >= numInformants
            selectedInformants = availableParticles(randperm(length(availableParticles), numInformants));
        else
            selectedInformants = availableParticles; % Use all available if not enough
        end
        informants{i} = selectedInformants;
    end
    
    % Initialize global best tracking
    globalBestFitness = inf;
    globalBestPosition = zeros(1, dimensions);
    
    % Tracking variables
    convergence = zeros(maxIterations, 1);
    annStats = struct();
    annStats.successHistory = [];
    annStats.mutationEvents = 0;
    annStats.annEvolutionHistory = [];

    % CRITICAL FIX: Track fitness from PREVIOUS iteration for proper improvement calculation
    previousFitness = inf(popSize, 1);

    % Initial fitness evaluation
    for i = 1:popSize
        previousFitness(i) = evaluatePathFitness(positions(i, :), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        personalBestFitness(i) = previousFitness(i);
    end

    % Main ANN-PSO loop (PAPER ALGORITHM 2, main do-while loop)
    for iter = 1:maxIterations
        iterationStartTime = tic;

        % PAPER ALGORITHM 2, lines 12-24: Update bests
        for i = 1:popSize
            % Evaluate current position
            currentFitness = evaluatePathFitness(positions(i, :), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
            
            % Update personal best (PAPER: pbest i â†? best previous location of i)
            if currentFitness < personalBestFitness(i)
                personalBestFitness(i) = currentFitness;
                personalBest(i, :) = positions(i, :);
            end
            
            % Update global best (PAPER: if fitness(xi) < fitness(best) then best â†? xi)
            if currentFitness < globalBestFitness
                globalBestFitness = currentFitness;
                globalBestPosition = positions(i, :);
            end
            
            % ANN-guided gbest selection (PAPER lines 19-24)
            selectedGbestIdx = selectGbestWithANN(particleANNs{i}, personalBest, informants{i});
            
            % PAPER ALGORITHM 2, lines 26-31: Update velocities and positions
            r1 = rand(1, dimensions);
            r2 = rand(1, dimensions);

            % Standard PSO velocity update with ANN-selected gbest
            velocities(i, :) = w * velocities(i, :) + ...
                              c1 * r1 .* (personalBest(i, :) - positions(i, :)) + ...
                              c2 * r2 .* (personalBest(selectedGbestIdx, :) - positions(i, :));

            % Velocity clamping (common in PSO implementations)
            maxVelocity = 0.2 * (upperBounds - lowerBounds);
            velocities(i, :) = max(-maxVelocity, min(maxVelocity, velocities(i, :)));

            % Update position (PAPER: xid â†? xid + vid)
            positions(i, :) = positions(i, :) + velocities(i, :);

            % Ensure bounds
            positions(i, :) = max(lowerBounds, min(upperBounds, positions(i, :)));
        end

        % PAPER ALGORITHM 2, lines 33-36: Evaluate ANNs
        % CRITICAL FIX: Calculate improvement based on fitness BEFORE vs AFTER this iteration
        for i = 1:popSize
            % Evaluate fitness AFTER movement
            newFitness = evaluatePathFitness(positions(i, :), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

            % PAPER: change â†? improvement in fitness(xi)
            % improvement = fitness_before_iteration - fitness_after_iteration
            improvement = previousFitness(i) - newFitness; % Positive if improved (minimization)

            % PAPER: anni.success â†? anni.success + change
            particleANNs{i}.success = particleANNs{i}.success + improvement;

            % Update previousFitness for next iteration
            previousFitness(i) = newFitness;
        end
        
        % PAPER ALGORITHM 2, lines 37-42: Evolve ANNs
        evolutionEvents = 0;
        for i = 1:popSize
            particleInformants = informants{i};
            
            % PAPER: annbest â†? most successful ann among informants
            bestSuccess = particleANNs{i}.success;
            bestInformantIdx = i;
            
            for j = 1:length(particleInformants)
                informantIdx = particleInformants(j);
                if particleANNs{informantIdx}.success > bestSuccess
                    bestSuccess = particleANNs{informantIdx}.success;
                    bestInformantIdx = informantIdx;
                end
            end
            
            % PAPER: if annbest.success > anni.success then anni â†?mutate(annbest)
            if bestInformantIdx ~= i
                particleANNs{i} = mutateConsistentANN(particleANNs{bestInformantIdx});
                evolutionEvents = evolutionEvents + 1;
            end
        end
        
        % Store convergence and statistics
        convergence(iter) = globalBestFitness;
        annStats.mutationEvents = annStats.mutationEvents + evolutionEvents;
        
        % Store success values for analysis
        successValues = zeros(popSize, 1);
        for i = 1:popSize
            successValues(i) = particleANNs{i}.success;
        end
        annStats.successHistory(iter, :) = successValues;
        
        iterationTime = toc(iterationStartTime);
        
        if mod(iter, 10) == 0
            avgSuccess = mean(successValues);
            maxSuccess = max(successValues);
            fprintf('  ANN-PSO Iter %d: Best=%.4f, AvgSuccess=%.3f, MaxSuccess=%.3f, Mutations=%d, Time=%.4fs\n', ...
                iter, convergence(iter), avgSuccess, maxSuccess, evolutionEvents, iterationTime);
        end
    end
    
    % Find best solution and convert to waypoint format
    [~, bestIdx] = min(personalBestFitness);
    bestPosition = personalBest(bestIdx, :);
    
    % Convert position vector to waypoint matrix
    waypoints = zeros(numWaypoints, 3);
    for i = 1:numWaypoints
        idx = (i-1)*3 + 1;
        waypoints(i,:) = bestPosition(idx:idx+2);
    end
    
    % Create full path with start and goal
    globalPath = [startPoint; waypoints; goalPoint];
    
    % Algorithm-specific statistics
    algorithmSpecificStats = struct();
    algorithmSpecificStats.actualBestFitness = globalBestFitness;
    algorithmSpecificStats.totalMutationEvents = annStats.mutationEvents;
    algorithmSpecificStats.finalAvgSuccess = mean(successValues);
    algorithmSpecificStats.finalMaxSuccess = max(successValues);
    algorithmSpecificStats.finalSuccessStd = std(successValues);
    
    % Add fitness components
    [~, globalBestComponents] = evaluatePathFitness(bestPosition, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, globalBestFitness, globalBestComponents);
    
    fprintf('ANN-PSO completed. Best: %.6f, Total mutations: %d\n', globalBestFitness, annStats.mutationEvents);
end

