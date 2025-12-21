function [globalPath, convergenceHistory, algorithmSpecificStats] = globalPathPlanningNeuralGuidedPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, fixedW, fixedC1, fixedC2)
    % Neural-Guided PSO (NNPSO) with Elite Network for UAV Path Planning
    % Uses 14-feature adaptive neural network guidance with trajectory-based learning
    % Key innovation: Unified convergence-stagnation index for exploration boost
    disp('Starting Neural-Guided PSO (NNPSO) path planning...');
    
    numWaypoints = 5;
    dims = 3 * numWaypoints;
    
    firstFeasibleIteration = -1;
    initialFeasibleCount = 0;
    convergenceHistory = [];
    
    eliteNetwork = initializeEliteNetwork(dims);
    eliteParticles = [];
    
    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);
    particles.lastImprovement = ones(popSize, 1);
    particles.clusterAssignment = ones(popSize, 1);
    particles.stabilityFactor = ones(popSize, dims);

    % Initialize globalBestComponents
    globalBestComponents = struct();
    globalBestComponents.pathLength = 0;
    globalBestComponents.collisionPenalty = 0;
    globalBestComponents.turningPenalty = 0;
    globalBestComponents.climbingPenalty = 0;
    globalBestComponents.heightPenalty = 0;
    globalBestComponents.terrainPenalty = 0;
    globalBestComponents.obstaclePenalty = 0;
    globalBestPosition = zeros(1, dims);
    globalBestFitness = Inf;
    
    current_w = fixedW;
    current_c1 = fixedC1;
    current_c2 = fixedC2;
    
    % Initialize particles (same structure as other algorithms)
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

    convergenceHistory = [convergenceHistory; globalBestFitness];

    % Initialize trajectory data collection for neural network training
    trajectoryData = struct();
    trajectoryData.positions = [];
    trajectoryData.velocities = [];
    trajectoryData.prevFitness = [];
    trajectoryData.newFitness = [];
    trajectoryData.wasStagnant = [];
    trajectoryData.stagnationDuration = [];

    % Track gamma for logging
    currentGamma = 0.0;

    % Main loop with fixed parameters
    for iter = 1:maxIterations
        environmentState = calculateEnvironmentState(particles, iter, maxIterations, globalBestPosition);
        environmentState.convergenceHistory = convergenceHistory;  % Add convergence history for feature extraction

        for i = 1:popSize
            c1_final = current_c1; 
            c2_final = current_c2; 
            
           % FIXED: Changed > to >= for consistency with training requirement
           if ~isempty(eliteNetwork.W1) && size(eliteParticles, 1) >= 10 && ~isempty(eliteNetwork.lossHistory)
                r1 = rand(1, dims);
                r2 = rand(1, dims);

                cognitiveComponent = c1_final * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));
                socialComponent = c2_final * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));

                % FIXED: Pass mapSize parameter
                nnGuidance = neuralNetworkGuidedUpdate(particles, i, eliteNetwork, environmentState, dims, globalBestPosition, globalBestFitness, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize);

                % IMPROVED: Adaptive gamma calculation with DECAY schedule
                % 1. Network Confidence: Based on loss improvement (normalized scale)
                recentLoss = mean(eliteNetwork.lossHistory(max(1,end-9):end));

                % Map loss to confidence: loss range [0.5, 5.0] → confidence [1.0, 0.0]
                minLoss = 0.5;   % Excellent network performance
                maxLoss = 5.0;   % Poor network performance
                normalizedLoss = (recentLoss - minLoss) / (maxLoss - minLoss);
                networkConfidence = max(0.0, min(1.0, 1.0 - normalizedLoss));

                % 2. Data Sufficiency: Scale with elite sample size
                dataSufficiency = min(1.0, size(eliteParticles, 1) / 50);

                % 3. DECAYING schedule: High exploration early, convergence later
                % Exponential decay: 1.0 → 0.135 over iterations
                explorationPhase = exp(-2.0 * iter / maxIterations);

                % 4. Combine all factors with max gamma = 0.4 (reduced from 1.0)
                % Early: gamma ≈ 0.4 (if network good), Late: gamma ≈ 0.05
                gamma = 0.4 * networkConfidence * dataSufficiency * explorationPhase;

                % Store for logging
                currentGamma = gamma;

                % IMPROVED: Residual learning - NN adds correction to PSO velocity
                % PSO provides base velocity, NN provides learned correction
                psoVelocity = current_w * particles.velocities(i,:) + ...
                              cognitiveComponent + socialComponent;

                particles.velocities(i,:) = psoVelocity + gamma * nnGuidance;
            else
                r1 = rand(1, dims);
                r2 = rand(1, dims);
                
                cognitiveComponent = c1_final * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));
                socialComponent = c2_final * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));
                
                particles.velocities(i,:) = current_w * particles.velocities(i,:) + ...
                          cognitiveComponent + socialComponent; 

            end
            
            maxVelocity = 0.15 * (mapSize(1) + mapSize(2) + mapSize(3)) / 3;
            for j = 1:numWaypoints
                idx = (j-1)*3 + 1;
                segmentVel = particles.velocities(i, idx:idx+2);
                velMag = norm(segmentVel);
                
                if velMag > maxVelocity
                    particles.velocities(i, idx:idx+2) = segmentVel * (maxVelocity / velMag);
                end
            end
            
            particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);
            
            for j = 1:numWaypoints
                idx = (j-1)*3 + 1;
                waypoint = particles.cartesianPositions(i, idx:idx+2);
                
                waypoint = max([1, 1, 1], min(waypoint, mapSize));

                terrainHeight = getTerrainHeight(waypoint(1), waypoint(2), terrainGrid, terrainX, terrainY);
                waypoint(3) = max(waypoint(3), terrainHeight + 10);
                
                particles.cartesianPositions(i, idx:idx+2) = waypoint;
            end
            
            [particles.fitness(i), fitnessComponents] = evaluatePathFitness(particles.cartesianPositions(i,:), startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

            % Update personal and global bests
            if particles.fitness(i) < particles.bestFitness(i)
                % Store previous best fitness before updating
                prevBestFitness = particles.bestFitness(i);

                particles.bestFitness(i) = particles.fitness(i);
                particles.bestPositions(i,:) = particles.cartesianPositions(i,:);

                % === TRAJECTORY DATA COLLECTION for Neural Learning ===
                % Check if this is a significant improvement (with safety check for division)
                if abs(prevBestFitness) > 1e-6
                    improvementRatio = (prevBestFitness - particles.fitness(i)) / prevBestFitness;
                else
                    improvementRatio = 0;  % Avoid division by zero
                end

                if improvementRatio > 0.05  % More than 5% improvement
                    % Record if particle was stagnant before this improvement (escape trajectory)
                    wasStagnant = particles.lastImprovement(i) > 5;
                    stagnationDuration = particles.lastImprovement(i);

                    % Store trajectory data (the velocity that led to this improvement)
                    trajectoryData.positions = [trajectoryData.positions; particles.cartesianPositions(i,:)];
                    trajectoryData.velocities = [trajectoryData.velocities; particles.velocities(i,:)];
                    trajectoryData.prevFitness = [trajectoryData.prevFitness; prevBestFitness];
                    trajectoryData.newFitness = [trajectoryData.newFitness; particles.fitness(i)];
                    trajectoryData.wasStagnant = [trajectoryData.wasStagnant; wasStagnant];
                    trajectoryData.stagnationDuration = [trajectoryData.stagnationDuration; stagnationDuration];
                end

                particles.lastImprovement(i) = 0;

                eliteParticles = updateEliteParticles(eliteParticles, particles.cartesianPositions(i,:), particles.fitness(i));

                if particles.fitness(i) < globalBestFitness
                    globalBestFitness = particles.fitness(i);
                    globalBestPosition = particles.cartesianPositions(i,:);
                    globalBestComponents = fitnessComponents;
                end
            else
                particles.lastImprovement(i) = particles.lastImprovement(i) + 1;
            end
        end
        
        convergenceHistory = [convergenceHistory; globalBestFitness];

        % Train network every 10 iterations (increased from 20) with data augmentation
        if mod(iter, 10) == 0 && size(eliteParticles, 1) > 10
            % Add data augmentation: create synthetic samples with 1% Gaussian noise
            % FIXED: Reduced from 0.05 to 0.01 to minimize contradictory training signals
            augmentedElites = [eliteParticles; addNoiseToElites(eliteParticles, 0.01)];

            % Pass both elite particles, augmented data, and trajectory data
            % FIXED: Pass mapSize parameter
            eliteNetwork = trainEliteNetworkProperly(eliteNetwork, augmentedElites, trajectoryData, environmentState, globalBestPosition, globalBestFitness, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize);
        end
        
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
        
        if mod(iter, 50) == 0 || iter == 1 || iter == maxIterations
            diversity = environmentState.diversity;
            disp(['NNPSO iteration: ', num2str(iter), '/', num2str(maxIterations), ...
                  ' - Best fitness: ', num2str(globalBestFitness), ...
                  ' - Diversity: ', num2str(diversity), ...
                  ' - Gamma (NN influence): ', num2str(currentGamma)]);
        end
    end
    
    globalPath = constructPathFromPSO(globalBestPosition, startPoint, goalPoint, numWaypoints);

    % Store the actual best fitness and add fitness components
    if ~exist('globalBestComponents', 'var')     
        [~, globalBestComponents] = evaluatePathFitness(globalBestPosition, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints); 
    end  
    algorithmSpecificStats.actualBestFitness = globalBestFitness;
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, globalBestFitness, globalBestComponents);

    disp(['NNPSO completed with final fitness: ', num2str(globalBestFitness)]);
end


