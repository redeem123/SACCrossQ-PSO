function eliteNetwork = initializeAblatedNetwork(dims, config)
    % Initialize network based on ablation configuration
    
    eliteNetwork = struct();
    
    % Determine input size based on feature set
    switch config.featureSet
        case 'full'
            eliteNetwork.inputSize = 14;  % All advanced features
        case 'without_stagnation'
            eliteNetwork.inputSize = 10;  % First 10 features only (no convergence-stagnation)
        case 'basic'
            eliteNetwork.inputSize = 10;
        case 'position'
            eliteNetwork.inputSize = 15;  % 5 waypoints × 3 dimensions
        case 'environment'
            eliteNetwork.inputSize = 5;   % Environment state features
        case 'none'
            eliteNetwork.inputSize = 1;   % Placeholder for no NN
        otherwise
            eliteNetwork.inputSize = 14;  % Default to full
    end
    
    eliteNetwork.outputSize = dims;
    
    % Set architecture based on configuration
    switch config.networkArchitecture
        case 'deep'
            eliteNetwork.hiddenSize1 = 64;
            eliteNetwork.hiddenSize2 = 32;
            eliteNetwork.numLayers = 3;
            
        case 'shallow'
            eliteNetwork.hiddenSize1 = 32;
            eliteNetwork.numLayers = 2;
            
        case 'linear'
            eliteNetwork.numLayers = 1;
            
        otherwise
            eliteNetwork.hiddenSize1 = 64;
            eliteNetwork.hiddenSize2 = 32;
            eliteNetwork.numLayers = 3;
    end
    
    % Initialize weights based on architecture
    if eliteNetwork.numLayers >= 1
        scale = sqrt(2.0 / eliteNetwork.inputSize);
        eliteNetwork.W1 = randn(eliteNetwork.outputSize, eliteNetwork.inputSize) * scale;
        eliteNetwork.b1 = zeros(eliteNetwork.outputSize, 1);
    end
    
    if eliteNetwork.numLayers >= 2
        eliteNetwork.W1 = randn(eliteNetwork.hiddenSize1, eliteNetwork.inputSize) * scale;
        eliteNetwork.b1 = zeros(eliteNetwork.hiddenSize1, 1);
        
        scale = sqrt(2.0 / eliteNetwork.hiddenSize1);
        eliteNetwork.W2 = randn(eliteNetwork.outputSize, eliteNetwork.hiddenSize1) * scale;
        eliteNetwork.b2 = zeros(eliteNetwork.outputSize, 1);
    end
    
    if eliteNetwork.numLayers >= 3
        eliteNetwork.W2 = randn(eliteNetwork.hiddenSize2, eliteNetwork.hiddenSize1) * scale;
        eliteNetwork.b2 = zeros(eliteNetwork.hiddenSize2, 1);
        
        scale = sqrt(2.0 / eliteNetwork.hiddenSize2);
        eliteNetwork.W3 = randn(eliteNetwork.outputSize, eliteNetwork.hiddenSize2) * scale;
        eliteNetwork.b3 = zeros(eliteNetwork.outputSize, 1);
    end
    
    % Training parameters
    eliteNetwork.learningRate = 0.001;
    eliteNetwork.lossHistory = [];
    eliteNetwork.inputMean = zeros(eliteNetwork.inputSize, 1);
    eliteNetwork.inputStd = ones(eliteNetwork.inputSize, 1);
    eliteNetwork.outputScale = 10.0;
    
end

