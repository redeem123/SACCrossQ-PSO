function criticNet = buildCriticNetwork(stateSize, actionSize, verbose)
    % Build Critic Network using MATLAB Deep Learning Toolbox
    %
    % Replaces custom weight matrices with modern dlnetwork architecture
    % Matches paper architecture: [state; action] → 64 → 64 → 32 → 32 → 16 → 1
    %
    % Inputs:
    %   stateSize: Dimension of state space (15 for baseline, 64 for transformer)
    %   actionSize: Dimension of action space (default: 20)
    %   verbose: Print network summary (default: false)
    %
    % Outputs:
    %   criticNet: dlnetwork object for Q-value estimation
    %
    % Architecture:
    %   Input([state; action]) → FC(64, LeakyReLU) → FC(64, LeakyReLU) →
    %   FC(32, LeakyReLU) → FC(32, LeakyReLU) → FC(16, LeakyReLU) → FC(1) → Output
    %
    % Note: For TD3, create two critics using this function
    %
    % Usage:
    %   critic1 = buildCriticNetwork(15, 20);  % Baseline
    %   critic2 = buildCriticNetwork(64, 20);  % Transformer state

    if nargin < 1
        stateSize = 15;  % Default: baseline state
    end

    if nargin < 2
        actionSize = 20;  % Default: 5 subgroups × 4 parameters
    end

    if nargin < 3
        verbose = false;  % Default: silent
    end

    % Total input size: state + action concatenated
    inputSize = stateSize + actionSize;

    % Define layer architecture
    layers = [
        featureInputLayer(inputSize, 'Name', 'state_action_input', ...
                         'Normalization', 'none')

        % Hidden Layer 1: (state+action) → 64
        fullyConnectedLayer(64, 'Name', 'fc1', ...
                           'WeightsInitializer', 'he', ...
                           'BiasInitializer', 'zeros')
        leakyReluLayer(0.01, 'Name', 'leaky_relu1')

        % Hidden Layer 2: 64 → 64
        fullyConnectedLayer(64, 'Name', 'fc2', ...
                           'WeightsInitializer', 'he', ...
                           'BiasInitializer', 'zeros')
        leakyReluLayer(0.01, 'Name', 'leaky_relu2')

        % Hidden Layer 3: 64 → 32
        fullyConnectedLayer(32, 'Name', 'fc3', ...
                           'WeightsInitializer', 'he', ...
                           'BiasInitializer', 'zeros')
        leakyReluLayer(0.01, 'Name', 'leaky_relu3')

        % Hidden Layer 4: 32 → 32
        fullyConnectedLayer(32, 'Name', 'fc4', ...
                           'WeightsInitializer', 'he', ...
                           'BiasInitializer', 'zeros')
        leakyReluLayer(0.01, 'Name', 'leaky_relu4')

        % Hidden Layer 5: 32 → 16
        fullyConnectedLayer(16, 'Name', 'fc5', ...
                           'WeightsInitializer', 'he', ...
                           'BiasInitializer', 'zeros')
        leakyReluLayer(0.01, 'Name', 'leaky_relu5')

        % Output Layer: 16 → 1 (Q-value, no activation)
        % Use smaller weight initialization for output stability
        fullyConnectedLayer(1, 'Name', 'q_value_output', ...
                           'WeightsInitializer', 'narrow-normal', ...
                           'BiasInitializer', 'zeros')
    ];

    % Create layer graph
    lgraph = layerGraph(layers);

    % Convert to dlnetwork
    criticNet = dlnetwork(lgraph);

    % Display network summary (only if verbose)
    if verbose
        fprintf('\n=== Critic Network (DL Toolbox) ===\n');
        fprintf('State Size: %d\n', stateSize);
        fprintf('Action Size: %d\n', actionSize);
        fprintf('Input Size: %d (state + action)\n', inputSize);
        fprintf('Architecture: %d → 64 → 64 → 32 → 32 → 16 → 1\n', inputSize);
        fprintf('===================================\n\n');
    end
end
