function nnGuidance = getAblatedNeuralGuidance(particles, particleIdx, eliteNetwork, ...
    environmentState, dims, config, globalBestPosition, globalBestFitness, startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY)
    % Get neural guidance based on feature configuration

    % Extract features based on configuration
    features = extractAblatedFeatures(particles, particleIdx, environmentState, config, globalBestPosition, globalBestFitness, startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY);
    
    % Ensure features match network input size
    if length(features) ~= eliteNetwork.inputSize
        % Pad or truncate if necessary
        if length(features) < eliteNetwork.inputSize
            features = [features; zeros(eliteNetwork.inputSize - length(features), 1)];
        else
            features = features(1:eliteNetwork.inputSize);
        end
    end
    
    % Normalize features - ensure dimensions match
    if length(eliteNetwork.inputMean) ~= length(features)
        % Reinitialize normalization parameters if needed
        eliteNetwork.inputMean = zeros(size(features));
        eliteNetwork.inputStd = ones(size(features));
    end
    
    features = (features - eliteNetwork.inputMean) ./ (eliteNetwork.inputStd + 1e-8);
    features(~isfinite(features)) = 0;
    features = max(-3, min(3, features));
    
    % Forward pass through network
    output = forwardPassAblatedNetwork(eliteNetwork, features);
    
    % Scale output
    nnGuidance = output' * eliteNetwork.outputScale * 0.5;  % Conservative scaling
    
    % Ensure correct dimensions
    if length(nnGuidance) < dims
        nnGuidance = [nnGuidance, zeros(1, dims - length(nnGuidance))];
    else
        nnGuidance = nnGuidance(1:dims);
    end
    
    nnGuidance(~isfinite(nnGuidance)) = 0;
    nnGuidance = max(-10, min(10, nnGuidance));
end

