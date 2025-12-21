function criticNet = createCriticNetwork(config)
    % Create SAC-CrossQ Critic Network with Batch Normalization
    %
    % Key features:
    %   - Concatenates state and action as input
    %   - BatchNorm after each hidden layer (CrossQ innovation)
    %   - Outputs single Q-value estimate
    %   - Two critics will be instantiated (SAC approach)
    %
    % Architecture:
    %   [State(45D), Action(120D)] → FC(128) → BatchNorm → ReLU
    %                                → FC(128) → BatchNorm → ReLU
    %                                → FC(64) → BatchNorm → ReLU
    %                                → FC(1)  [Q-value]
    %
    % Based on:
    %   - CrossQ (ICLR 2024): BatchNorm for sample efficiency
    %   - SAC: Twin critics for stable learning
    %
    % Inputs:
    %   config: APEX-PSO configuration
    %
    % Outputs:
    %   criticNet: dlnetwork critic network

    stateSize = config.stateSize;
    actionSize = config.actionSize;
    inputSize = stateSize + actionSize;  % Concatenated state-action input
    hiddenLayers = config.criticHiddenLayers;

    % Build layer array
    layers = [];

    % Input layer (state and action concatenated)
    layers = [layers
        featureInputLayer(inputSize, 'Name', 'state_action_input', 'Normalization', 'none')];

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

    % Output layer: Single Q-value (no activation)
    layers = [layers
        fullyConnectedLayer(1, 'Name', 'q_value')];

    % Create layer graph for proper network construction
    lgraph = layerGraph(layers);

    % Create network
    criticNet = dlnetwork(lgraph);

    % Initialize weights
    criticNet = initializeWeights(criticNet);
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
