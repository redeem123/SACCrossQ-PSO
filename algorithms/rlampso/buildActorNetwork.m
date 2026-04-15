function actorNet = buildActorNetwork(stateSize, actionSize, verbose)
    % Build Actor Network using MATLAB Deep Learning Toolbox
    %
    % Replaces custom weight matrices with modern dlnetwork architecture
    % Matches paper architecture: state → 64 → 64 → 64 → action (tanh)
    %
    % Inputs:
    %   stateSize: Dimension of state space (15 for baseline, 64 for transformer)
    %   actionSize: Dimension of action space (default: 20 for 5 subgroups × 4 params)
    %   verbose: Print network summary (default: false)
    %
    % Outputs:
    %   actorNet: dlnetwork object ready for training
    %
    % Architecture:
    %   Input(stateSize) → FC(64, LeakyReLU) → FC(64, LeakyReLU) →
    %   FC(64, LeakyReLU) → FC(actionSize, Tanh) → Output
    %
    % Usage:
    %   actorNet = buildActorNetwork(15, 20);  % Baseline
    %   actorNet = buildActorNetwork(64, 20);  % Transformer state

    if nargin < 1
        stateSize = 15;  % Default: baseline state
    end

    if nargin < 2
        actionSize = 20;  % Default: 5 subgroups × 4 parameters
    end

    if nargin < 3
        verbose = false;  % Default: silent
    end

    % Define layer architecture
    layers = [
        featureInputLayer(stateSize, 'Name', 'state_input', ...
                         'Normalization', 'none')

        % Hidden Layer 1: stateSize → 64
        fullyConnectedLayer(64, 'Name', 'fc1', ...
                           'WeightsInitializer', 'he', ...
                           'BiasInitializer', 'zeros')
        leakyReluLayer(0.01, 'Name', 'leaky_relu1')

        % Hidden Layer 2: 64 → 64
        fullyConnectedLayer(64, 'Name', 'fc2', ...
                           'WeightsInitializer', 'he', ...
                           'BiasInitializer', 'zeros')
        leakyReluLayer(0.01, 'Name', 'leaky_relu2')

        % Hidden Layer 3: 64 → 64
        fullyConnectedLayer(64, 'Name', 'fc3', ...
                           'WeightsInitializer', 'he', ...
                           'BiasInitializer', 'zeros')
        leakyReluLayer(0.01, 'Name', 'leaky_relu3')

        % Output Layer: 64 → actionSize (with tanh activation)
        fullyConnectedLayer(actionSize, 'Name', 'fc4', ...
                           'WeightsInitializer', 'glorot', ...
                           'BiasInitializer', 'zeros')
        tanhLayer('Name', 'tanh_output')
    ];

    % Create layer graph
    lgraph = layerGraph(layers);

    % Convert to dlnetwork
    actorNet = dlnetwork(lgraph);

    % Display network summary (only if verbose)
    if verbose
        fprintf('\n=== Actor Network (DL Toolbox) ===\n');
        fprintf('State Size: %d\n', stateSize);
        fprintf('Action Size: %d\n', actionSize);
        fprintf('Architecture: %d → 64 → 64 → 64 → %d (tanh)\n', stateSize, actionSize);
        fprintf('==================================\n\n');
    end
end
