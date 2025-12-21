function actorNet = createActorNetwork(config)
    % Create SAC-CrossQ Actor Network with Batch Normalization
    %
    % Key features:
    %   - BatchNorm after each hidden layer (CrossQ innovation)
    %   - Gaussian policy with mean and log_std outputs
    %   - Tanh squashing for bounded actions
    %   - Larger architecture for 120D action space
    %
    % Architecture:
    %   Input (45D) → FC(128) → BatchNorm → ReLU
    %              → FC(128) → BatchNorm → ReLU
    %              → FC(128) → BatchNorm → ReLU
    %              → [Mean FC(120), LogStd FC(120)]
    %
    % Based on:
    %   - CrossQ (ICLR 2024): BatchNorm dramatically improves sample efficiency
    %   - SAC: Stochastic Gaussian policy for exploration
    %
    % Inputs:
    %   config: APEX-PSO configuration
    %
    % Outputs:
    %   actorNet: dlnetwork actor network

    stateSize = config.stateSize;
    actionSize = config.actionSize;
    hiddenLayers = config.actorHiddenLayers;

    % Build layer array
    layers = [];

    % Input layer
    layers = [layers
        featureInputLayer(stateSize, 'Name', 'state_input', 'Normalization', 'none')];

    % Hidden layers with BatchNorm (CrossQ innovation)
    for i = 1:length(hiddenLayers)
        hiddenSize = hiddenLayers(i);
        layerName = sprintf('fc%d', i);
        bnName = sprintf('bn%d', i);
        reluName = sprintf('relu%d', i);

        layers = [layers
            fullyConnectedLayer(hiddenSize, 'Name', layerName)
            batchNormalizationLayer('Name', bnName, ...
                'Epsilon', config.batchNormEpsilon)
            reluLayer('Name', reluName)];
    end

    % Output: Gaussian policy (mean and log_std concatenated)
    % SAC outputs both mean and log_std
    % Output size: actionSize*2 (mean and log_std concatenated)

    layers = [layers
        fullyConnectedLayer(actionSize * 2, 'Name', 'output_fc')];

    % Create layer graph for proper network construction
    lgraph = layerGraph(layers);

    % Create network
    actorNet = dlnetwork(lgraph);

    % Initialize weights using Xavier/He initialization
    actorNet = initializeWeights(actorNet);
end

function net = initializeWeights(net)
    % Initialize network weights with proper scaling
    % Use Xavier/He initialization for better convergence

    % Access learnables table
    learnables = net.Learnables;

    for i = 1:height(learnables)
        paramName = learnables.Parameter{i};
        layerName = learnables.Layer{i};

        if strcmp(paramName, 'Weights')
            % Get current weights to determine size
            weights = learnables.Value{i};
            [outputSize, inputSize] = size(weights);

            % He initialization (good for ReLU)
            scale = sqrt(2.0 / inputSize);
            newWeights = dlarray(randn(outputSize, inputSize, 'single') * scale);
            net.Learnables.Value{i} = newWeights;

        elseif strcmp(paramName, 'Bias')
            % Initialize biases to zero
            bias = learnables.Value{i};
            newBias = dlarray(zeros(size(bias), 'single'));
            net.Learnables.Value{i} = newBias;
        end
    end
end
