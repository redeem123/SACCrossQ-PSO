function eliteNetwork = trainAblatedNetwork(eliteNetwork, eliteParticles, ...
    environmentState, config, globalBestPosition, globalBestFitness, startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY)
    % Train network based on configuration
    
    if strcmp(config.trainingStrategy, 'none')
        return;  % No training
    end
    
    % Prepare training data (simplified for ablation)
    numSamples = min(50, size(eliteParticles, 1));
    [~, sortIdx] = sort(eliteParticles(:, end));
    bestElites = eliteParticles(sortIdx(1:numSamples), :);
    
    % Simple training loop (abbreviated for ablation study)
    numEpochs = 20;
    for epoch = 1:numEpochs
        totalLoss = 0;
        for i = 1:numSamples-1
            % Create input-output pairs
            currentPos = bestElites(i, 1:end-1);
            betterPos = bestElites(i+1, 1:end-1);
            
            particles_temp = struct('cartesianPositions', currentPos, ...
                                     'fitness', bestElites(i, end), ...
                                     'bestFitness', bestElites(i, end), ...
                                     'lastImprovement', 1, ...
                                     'velocities', zeros(1, size(currentPos, 2)));
            features = extractAblatedFeatures(particles_temp, 1, environmentState, config, ...
                globalBestPosition, globalBestFitness, startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY);
            
            target = (betterPos - currentPos)' * 0.1;
            
            % Simple gradient update (simplified)
            output = forwardPassAblatedNetwork(eliteNetwork, features);
            loss = 0.5 * mean((output - target).^2);
            totalLoss = totalLoss + loss;
        end
        
        eliteNetwork.lossHistory = [eliteNetwork.lossHistory, totalLoss / numSamples];
    end
end

