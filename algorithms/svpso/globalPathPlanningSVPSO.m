function [globalPath, sphericalStats, vectorStats, convergenceHistory, algorithmSpecificStats] = globalPathPlanningSVPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, w, c1, c2)
    % True Spherical-Vector PSO: Operating entirely in spherical coordinates
    % Reuses SAEPSO's spherical coordinate functions for proper implementation
    
    disp('Starting True Spherical-Vector PSO with spherical coordinates...');
    
    numWaypoints = 5;
    sphericalDims = 3 * numWaypoints; % (Ï?, Î¾, Ï†) for each waypoint
    
    firstFeasibleIteration = -1;
    initialFeasibleCount = 0;
    convergenceHistory = [];
    
    % Initialize globalBestComponents
    globalBestComponents = struct();
    globalBestComponents.pathLength = 0;
    globalBestComponents.collisionPenalty = 0;
    globalBestComponents.turningPenalty = 0;
    globalBestComponents.climbingPenalty = 0;
    globalBestComponents.heightPenalty = 0;
    globalBestComponents.terrainPenalty = 0;
    globalBestComponents.obstaclePenalty = 0;
    
    % Define spherical coordinate bounds (reusing SAEPSO approach)
    maxSegmentLength = norm(goalPoint - startPoint) / numWaypoints * 2;
    sphericalLowerBounds = [];
    sphericalUpperBounds = [];
    for i = 1:numWaypoints
        sphericalLowerBounds = [sphericalLowerBounds, 0, -pi/2, 0]; % [Ï?_min, Î¾_min, Ï†_min]
        sphericalUpperBounds = [sphericalUpperBounds, maxSegmentLength, pi/2, 2*pi]; % [Ï?_max, Î¾_max, Ï†_max]
    end
    
    % Initialize particles in spherical coordinate space
    particles = struct();
    particles.sphericalPositions = zeros(popSize, sphericalDims);
    particles.sphericalVelocities = zeros(popSize, sphericalDims);
    particles.cartesianPositions = zeros(popSize, sphericalDims); % For fitness evaluation
    particles.bestSphericalPositions = zeros(popSize, sphericalDims);
    particles.bestCartesianPositions = zeros(popSize, sphericalDims);
    particles.bestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);
    
    globalBestSphericalPosition = zeros(1, sphericalDims);
    globalBestCartesianPosition = zeros(1, sphericalDims);
    globalBestFitness = Inf;
    
    % Tracking for spherical statistics
    sphericalStats = [];
    vectorStats = [];
    
    % Initialize particles using improved initialization (like SAEPSO stage 1)
    fprintf('  Initializing particles in spherical coordinates...\n');
    
    for i = 1:popSize
        % Initialize in spherical coordinates with tent map approach
        for j = 1:sphericalDims
            % Use tent map perturbation for better initialization
            gamma_t = rand();
            r1 = rand(); r2 = rand(); r3 = rand();
            
            % Apply tent map with perturbation (from SAEPSO Eq. 39)
            if 0 < gamma_t && gamma_t < 0.5
                gamma_next = 2 * gamma_t + (2*r1 - 1)/100;
            elseif 0.5 < gamma_t && gamma_t < 1
                gamma_next = 2 * (1 - gamma_t) + (2*r2 - 1)/100;
            else
                gamma_next = r3;
            end
            
            gamma_next = max(0, min(1, gamma_next));
            
            % Convert to spherical coordinate bounds
            particles.sphericalPositions(i, j) = sphericalLowerBounds(j) + ...
                (sphericalUpperBounds(j) - sphericalLowerBounds(j)) * gamma_next;
        end
        
        % Convert spherical to cartesian for fitness evaluation
        particles.cartesianPositions(i,:) = sphericalToCartesian(particles.sphericalPositions(i,:), startPoint, numWaypoints);
        
        % Apply terrain and boundary constraints to cartesian coordinates
        particles.cartesianPositions(i,:) = applyTerrainConstraints(particles.cartesianPositions(i,:), terrainGrid, terrainX, terrainY, mapSize, numWaypoints);
        
        % Check feasibility
        if isPathFeasible(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
            initialFeasibleCount = initialFeasibleCount + 1;
        end
        
        % Initialize spherical velocities (small random values)
        particles.sphericalVelocities(i,:) = (rand(1, sphericalDims) - 0.5) * 0.1;
        
        % Evaluate fitness
        [particles.fitness(i), fitnessComponents] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        
        % Initialize personal best
        particles.bestSphericalPositions(i,:) = particles.sphericalPositions(i,:);
        particles.bestCartesianPositions(i,:) = particles.cartesianPositions(i,:);
        particles.bestFitness(i) = particles.fitness(i);
        
        % Update global best
        if particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestSphericalPosition = particles.sphericalPositions(i,:);
            globalBestCartesianPosition = particles.cartesianPositions(i,:);
            globalBestComponents = fitnessComponents;
        end
    end
    
    if initialFeasibleCount ~= 0
        firstFeasibleIteration = 0;
    end
    disp(['Number of Initial feasible particles: ', num2str(initialFeasibleCount), '/', num2str(popSize)]);
    
    convergenceHistory = [convergenceHistory; globalBestFitness];
    
    % Main Spherical-Vector PSO loop
    fprintf('  Running main spherical PSO optimization...\n');
    
    for iter = 1:maxIterations
        for i = 1:popSize
            % Standard PSO update equations in spherical coordinate space
            r1 = rand(1, sphericalDims);
            r2 = rand(1, sphericalDims);
            
            % Spherical velocity update (PSO equations applied in spherical space)
            cognitiveComponent = c1 * r1 .* (particles.bestSphericalPositions(i,:) - particles.sphericalPositions(i,:));
            socialComponent = c2 * r2 .* (globalBestSphericalPosition - particles.sphericalPositions(i,:));
            
            particles.sphericalVelocities(i,:) = w * particles.sphericalVelocities(i,:) + ...
                                                cognitiveComponent + socialComponent;
            
            % Velocity clamping in spherical space
            maxSphericalVelocity = 0.1; % Conservative limit for spherical coordinates
            for j = 1:sphericalDims
                if abs(particles.sphericalVelocities(i,j)) > maxSphericalVelocity
                    particles.sphericalVelocities(i,j) = sign(particles.sphericalVelocities(i,j)) * maxSphericalVelocity;
                end
            end
            
            % Update spherical position
            particles.sphericalPositions(i,:) = particles.sphericalPositions(i,:) + particles.sphericalVelocities(i,:);
            
            % Apply spherical coordinate bounds
            for j = 1:sphericalDims
                particles.sphericalPositions(i,j) = max(sphericalLowerBounds(j), ...
                    min(sphericalUpperBounds(j), particles.sphericalPositions(i,j)));
            end
            
            % Convert to cartesian coordinates for fitness evaluation
            particles.cartesianPositions(i,:) = sphericalToCartesian(particles.sphericalPositions(i,:), startPoint, numWaypoints);
            
            % Apply terrain and boundary constraints
            particles.cartesianPositions(i,:) = applyTerrainConstraints(particles.cartesianPositions(i,:), terrainGrid, terrainX, terrainY, mapSize, numWaypoints);
            
            % Evaluate fitness
            [particles.fitness(i), fitnessComponents] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
            
            % Update personal best
            if particles.fitness(i) < particles.bestFitness(i)
                particles.bestFitness(i) = particles.fitness(i);
                particles.bestSphericalPositions(i,:) = particles.sphericalPositions(i,:);
                particles.bestCartesianPositions(i,:) = particles.cartesianPositions(i,:);
                
                % Update global best
                if particles.fitness(i) < globalBestFitness
                    globalBestFitness = particles.fitness(i);
                    globalBestSphericalPosition = particles.sphericalPositions(i,:);
                    globalBestCartesianPosition = particles.cartesianPositions(i,:);
                    globalBestComponents = fitnessComponents;
                end
            end
        end
        
        convergenceHistory = [convergenceHistory; globalBestFitness];
        
        % Calculate spherical-specific statistics
        sphericalConvergence = calculateSphericalConvergenceMetrics(particles.sphericalPositions, globalBestSphericalPosition);
        sphericalStats = [sphericalStats; sphericalConvergence];
        
        % Calculate vector magnitude statistics in spherical space
        avgVectorMagnitude = mean(sqrt(sum(particles.sphericalVelocities.^2, 2)));
        vectorStats = [vectorStats; avgVectorMagnitude];
        
        % Check for first feasible solution
        if firstFeasibleIteration == -1
            for i = 1:popSize
                if isPathFeasible(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints)
                    firstFeasibleIteration = iter;
                    break;
                end
            end
        end
        
        if firstFeasibleIteration == iter || (firstFeasibleIteration == 0 && firstFeasibleIteration+1 == iter)
            disp(['First feasible path found at iteration: ', num2str(firstFeasibleIteration)]);
        end
        
        % Progress reporting
        if mod(iter, 50) == 0
            avgSphericalRadius = mean(particles.sphericalPositions(:, 1:3:end), 'all');
            disp(['True SVPSO iteration: ', num2str(iter), '/', num2str(maxIterations), ...
                  ' - Best fitness: ', num2str(globalBestFitness), ...
                  ' - Spherical convergence: ', num2str(sphericalConvergence), ...
                  ' - Avg radius: ', num2str(avgSphericalRadius)]);
        end
    end
    
    % Construct final path from best spherical solution
    globalPath = constructPathFromPSO(globalBestCartesianPosition, startPoint, goalPoint, numWaypoints);
    
    % Store algorithm statistics
    if ~exist('globalBestComponents', 'var')
        [~, globalBestComponents] = evaluatePathFitness(globalBestCartesianPosition, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
    end
    algorithmSpecificStats.actualBestFitness = globalBestFitness;
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, globalBestFitness, globalBestComponents);
    
    % Add spherical-specific statistics
    algorithmSpecificStats.finalSphericalConvergence = sphericalConvergence;
    algorithmSpecificStats.avgSphericalRadius = mean(globalBestSphericalPosition(1:3:end));
    algorithmSpecificStats.avgClimbingAngle = mean(globalBestSphericalPosition(2:3:end));
    algorithmSpecificStats.avgTurningAngle = mean(globalBestSphericalPosition(3:3:end));
    
    disp(['True Spherical-Vector PSO completed with final fitness: ', num2str(globalBestFitness)]);
    disp(['  Final spherical convergence: ', num2str(sphericalConvergence)]);
    disp(['  Average spherical radius: ', num2str(algorithmSpecificStats.avgSphericalRadius)]);
end

