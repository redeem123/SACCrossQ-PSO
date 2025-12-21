function [globalPath, convergenceHistory, algorithmSpecificStats] = globalPathPlanningPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, w, c1, c2)
    % Traditional PSO with tracking - completely traditional implementation
    
    disp('Starting Traditional PSO path planning with tracking...');
    
    numWaypoints = 5;
    dims = 3 * numWaypoints;
    
    firstFeasibleIteration = -1;
    initialFeasibleCount = 0;
    convergenceHistory = [];
    
    % Initialize particles using traditional random initialization
    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);
    
    globalBestPosition = zeros(1, dims);
    globalBestFitness = Inf;
    
    % Initialize globalBestComponents
    globalBestComponents = struct();
    globalBestComponents.pathLength = 0;
    globalBestComponents.collisionPenalty = 0;
    globalBestComponents.turningPenalty = 0;
    globalBestComponents.climbingPenalty = 0;
    globalBestComponents.heightPenalty = 0;
    globalBestComponents.terrainPenalty = 0;
    globalBestComponents.obstaclePenalty = 0;

    % Vectorized initialization for better performance
    % Generate all random values at once
    randMatrix = rand(popSize, dims);
    
    % Pre-allocate coordinate indices
    xIndices = 1:3:dims;
    yIndices = 2:3:dims;
    zIndices = 3:3:dims;
    
    % Vectorized position initialization
    particles.cartesianPositions(:, xIndices) = randMatrix(:, xIndices) * mapSize(1);      % X coordinates
    particles.cartesianPositions(:, yIndices) = randMatrix(:, yIndices) * mapSize(2);      % Y coordinates  
    particles.cartesianPositions(:, zIndices) = 10 + randMatrix(:, zIndices) * (mapSize(3) - 10); % Z coordinates
    
    % Vectorized velocity initialization
    particles.velocities = (rand(popSize, dims) - 0.5) * 10; % [-5, 5] range
    
    % MISSING FOR LOOP ADDED HERE
    for i = 1:popSize
        % Evaluate fitness
        [particles.fitness(i), fitnessComponents] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        
        % Initialize personal best
        particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
        particles.bestFitness(i) = particles.fitness(i);
        
        % Update global best
        if particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestPosition = particles.cartesianPositions(i,:);
            globalBestComponents = fitnessComponents;
        end
        
        % Check feasibility for tracking
        if isPathFeasible(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
            initialFeasibleCount = initialFeasibleCount + 1;
        end
    end
    
    if initialFeasibleCount ~= 0
        firstFeasibleIteration = 0;  
    end 
    disp(['Number of Initial feasible particles: ', num2str(initialFeasibleCount), '/', num2str(popSize)]);

    % Store initial best fitness
    convergenceHistory = [convergenceHistory; globalBestFitness];

    % Main Traditional PSO loop
    for iter = 1:maxIterations
        for i = 1:popSize
            % Traditional PSO velocity update equation
            r1 = rand(1, dims);  % Random vector for cognitive component
            r2 = rand(1, dims);  % Random vector for social component
            
            % Standard PSO velocity update: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
            particles.velocities(i,:) = w * particles.velocities(i,:) + ...
                                       c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) + ...
                                       c2 * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));
            
            % Traditional velocity clamping (simple maximum velocity)
            maxVelocity = 20.0;  % Simple maximum velocity limit
            for d = 1:dims
                if abs(particles.velocities(i,d)) > maxVelocity
                    particles.velocities(i,d) = sign(particles.velocities(i,d)) * maxVelocity;
                end
            end
            
            % Traditional position update: x = x + v
            particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);
            
            % Traditional boundary handling (simple clamping)
            for j = 1:numWaypoints
                idx = (j-1)*3 + 1;
                % Clamp X coordinate
                particles.cartesianPositions(i, idx) = max(0, min(particles.cartesianPositions(i, idx), mapSize(1)));
                % Clamp Y coordinate  
                particles.cartesianPositions(i, idx+1) = max(0, min(particles.cartesianPositions(i, idx+1), mapSize(2)));
                % Clamp Z coordinate (minimum height of 10)
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

        % Store convergence data
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
        
        % Progress reporting
        if mod(iter, 50) == 0
            disp(['Traditional PSO iteration: ', num2str(iter), '/', num2str(maxIterations), ' - Best fitness: ', num2str(globalBestFitness)]);
        end
    end
  
    % Construct final path with mapSize constraints
    globalPath = constructPathFromPSO(globalBestPosition, startPoint, goalPoint, numWaypoints, mapSize);

    % Apply simple terrain constraints to final path (minimal post-processing)
    for i = 1:size(globalPath, 1)
        x = globalPath(i,1);
        y = globalPath(i,2);
        
        [~, xIndex] = min(abs(terrainX(1,:) - x));
        [~, yIndex] = min(abs(terrainY(:,1) - y));
        
        % Simple terrain clearance (just ensure minimum height above terrain)
        minHeight = terrainGrid(yIndex, xIndex) + 8;
        globalPath(i,3) = max(globalPath(i,3), minHeight);
    end
    
    % Return algorithm statistics
    if ~exist('globalBestComponents', 'var')    
        [~, globalBestComponents] = evaluatePathFitness(globalBestPosition, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints); 
    end  
    algorithmSpecificStats.actualBestFitness = globalBestFitness;
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, globalBestFitness, globalBestComponents);
    
    disp(['Traditional PSO completed with final fitness: ', num2str(globalBestFitness)]);
    
end

%% 4. SVPSO (Spherical-Vector PSO)

