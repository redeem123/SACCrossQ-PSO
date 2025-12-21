function [globalPath, convergenceHistory, algorithmSpecificStats] = globalPathPlanningNeuralGuidedPSO_Ablated(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, popSize, maxIterations, fixedW, fixedC1, fixedC2, config)
    % Ablated version of Neural-Guided PSO based on configuration

    tic;
    disp(['Starting Ablated Neural-Guided PSO: ', config.name]);
    
    numWaypoints = 5;
    dims = 3 * numWaypoints;
    
    convergenceHistory = [];
    neuralInfluenceHistory = [];
    
    % Initialize network based on configuration
    if config.useNeuralGuidance
        eliteNetwork = initializeAblatedNetwork(dims, config);
    else
        eliteNetwork = [];
    end
    eliteParticles = [];
    
    % Initialize particles (same as original)
    particles = initializeParticles(popSize, dims, numWaypoints, mapSize, ...
        terrainGrid, terrainX, terrainY, startPoint, goalPoint, dangerZones);
    
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
    globalBestComponents.duplicatePenalty = 0;
    
    % Find initial best
    for i = 1:popSize
        if particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestPosition = particles.cartesianPositions(i,:);
            [~, globalBestComponents] = evaluatePathFitness(particles.cartesianPositions(i,:), ...
                startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
        end
    end
    
    convergenceHistory = [convergenceHistory; globalBestFitness];
    
    % Main PSO loop
    for iter = 1:maxIterations
        environmentState = calculateEnvironmentState(particles, iter, maxIterations, globalBestPosition);
        
        for i = 1:popSize
            % Calculate standard PSO components
            r1 = rand(1, dims);
            r2 = rand(1, dims);
            
            cognitiveComponent = fixedC1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:));
            socialComponent = fixedC2 * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));
            
            % Apply neural guidance based on configuration
            if config.useNeuralGuidance && size(eliteParticles, 1) > 5
                % Get neural guidance based on feature set
                nnGuidance = getAblatedNeuralGuidance(particles, i, eliteNetwork, ...
                    environmentState, dims, config, globalBestPosition, globalBestFitness, ...
                    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY);
                
                % Calculate influence based on configuration
                gamma = calculateAblatedInfluence(eliteNetwork, eliteParticles, ...
                    particles, i, iter, maxIterations, config);
                
                alpha = 1.0 - gamma;
                
                % Combined velocity update
                particles.velocities(i,:) = fixedW * particles.velocities(i,:) + ...
                    alpha * (cognitiveComponent + socialComponent) + ...
                    gamma * nnGuidance;
                
                neuralInfluenceHistory = [neuralInfluenceHistory, gamma];
            else
                % Standard PSO update
                particles.velocities(i,:) = fixedW * particles.velocities(i,:) + ...
                    cognitiveComponent + socialComponent;
            end
            
            % Velocity clamping and position update (same as original)
            particles = updateParticlePosition(particles, i, mapSize, terrainGrid, ...
                terrainX, terrainY, numWaypoints);
            
            % Evaluate fitness
            [particles.fitness(i), fitnessComponents] = evaluatePathFitness(...
                particles.cartesianPositions(i,:), startPoint, goalPoint, ...
                dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
            
            % Update personal and global bests
            if particles.fitness(i) < particles.bestFitness(i)
                particles.bestFitness(i) = particles.fitness(i);
                particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
                
                % Update elite particles for neural network
                if config.useNeuralGuidance
                    eliteParticles = updateEliteParticles(eliteParticles, ...
                        particles.cartesianPositions(i,:), particles.fitness(i));
                end
                
                if particles.fitness(i) < globalBestFitness
                    globalBestFitness = particles.fitness(i);
                    globalBestPosition = particles.cartesianPositions(i,:);
                    globalBestComponents = fitnessComponents;
                end
            end
        end
        
        convergenceHistory = [convergenceHistory; globalBestFitness];
        
        % Train network based on configuration
        if config.useNeuralGuidance && strcmp(config.trainingStrategy, 'online')
            if mod(iter, 20) == 0 && size(eliteParticles, 1) > 10
                eliteNetwork = trainAblatedNetwork(eliteNetwork, eliteParticles, ...
                    environmentState, config, globalBestPosition, globalBestFitness, ...
                    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY);
            end
        elseif config.useNeuralGuidance && strcmp(config.trainingStrategy, 'batch')
            % Train only at the end
            if iter == maxIterations && size(eliteParticles, 1) > 10
                eliteNetwork = trainAblatedNetwork(eliteNetwork, eliteParticles, ...
                    environmentState, config, globalBestPosition, globalBestFitness, ...
                    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY);
            end
        end
    end
    
    % Construct final path
    globalPath = constructPathFromPSO(globalBestPosition, startPoint, goalPoint, numWaypoints);
    
    % Make sure we have valid fitness components
    if ~isfield(globalBestComponents, 'pathLength') || globalBestComponents.pathLength == 0
        [~, globalBestComponents] = evaluatePathFitness(globalBestPosition, ...
            startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
    end
    
    % Prepare statistics
    algorithmSpecificStats = struct();
    algorithmSpecificStats.actualBestFitness = globalBestFitness;
    algorithmSpecificStats.executionTime = toc;
    
    if ~isempty(neuralInfluenceHistory)
        algorithmSpecificStats.avgNeuralInfluence = mean(neuralInfluenceHistory);
    else
        algorithmSpecificStats.avgNeuralInfluence = 0;
    end
    
    algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, ...
        globalBestFitness, globalBestComponents);
end

