function loraAgent = initializeLoRAAgent(baseAgent, loraRank, frozenLayers)
    % Initialize LoRA (Low-Rank Adaptation) Agent for Transfer Learning
    %
    % Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
    %
    % LoRA adds low-rank decomposition matrices to frozen pretrained weights:
    %   W' = W + BA, where B: (d × r), A: (r × k), r << min(d, k)
    %
    % Inputs:
    %   baseAgent: Pretrained agent with frozen base weights
    %   loraRank: Rank of LoRA matrices (default: 8)
    %   frozenLayers: Cell array of layer names to freeze (default: all except output)
    %
    % Outputs:
    %   loraAgent: Agent with LoRA parameters added

    if nargin < 2, loraRank = 8; end
    if nargin < 3
        % Default: Freeze all hidden layers, train only LoRA
        frozenLayers = {'W1', 'W2', 'W3', 'W4', 'W5', 'W6'};
    end

    % Copy base agent
    loraAgent = baseAgent;
    loraAgent.loraRank = loraRank;
    loraAgent.frozenLayers = frozenLayers;
    loraAgent.useLoRA = true;

    fprintf('=== Initializing LoRA Agent ===\n');
    fprintf('LoRA Rank: %d\n', loraRank);
    fprintf('Frozen Layers: %s\n', strjoin(frozenLayers, ', '));

    % === ADD LORA MATRICES TO ACTOR ===
    fprintf('\nAdding LoRA to Actor Network...\n');
    loraAgent.actor = addLoRAToNetwork(loraAgent.actor, loraRank, 'actor');

    % === ADD LORA TO CRITICS ===
    fprintf('Adding LoRA to Critic Networks...\n');

    if isfield(loraAgent, 'critic1')
        % TD3 has twin critics
        loraAgent.critic1 = addLoRAToNetwork(loraAgent.critic1, loraRank, 'critic1');
        loraAgent.critic2 = addLoRAToNetwork(loraAgent.critic2, loraRank, 'critic2');
    else
        % DDPG has single critic
        loraAgent.critic = addLoRAToNetwork(loraAgent.critic, loraRank, 'critic');
    end

    % === FREEZE BASE WEIGHTS ===
    fprintf('\nFreezing base network weights...\n');
    loraAgent.frozenWeights = extractBaseWeights(baseAgent);

    fprintf('\n=== LoRA Initialization Complete ===\n');
    fprintf('Trainable Parameters: %d (LoRA only)\n', countLoRAParameters(loraAgent));
    fprintf('Frozen Parameters: %d (base weights)\n', countBaseParameters(baseAgent));
    fprintf('===================================\n\n');
end

function net = addLoRAToNetwork(net, rank, networkName)
    % Add LoRA matrices to network layers
    %
    % For each weight matrix W (d × k), add:
    %   - LoRA_A: (r × k) initialized with Gaussian
    %   - LoRA_B: (d × r) initialized with zeros
    %   - Scaling factor alpha (default: rank)

    loraAlpha = rank;  % Scaling factor
    loraFields = {};

    % Identify weight matrices to augment
    fields = fieldnames(net);
    for i = 1:length(fields)
        field = fields{i};

        % Only add LoRA to weight matrices (not biases)
        if startsWith(field, 'W') && isnumeric(net.(field))
            [d, k] = size(net.(field));

            % Initialize LoRA matrices
            % LoRA_A: (r × k) - Gaussian initialization
            loraA_name = [field, '_LoRA_A'];
            net.(loraA_name) = randn(rank, k) * 0.01;

            % LoRA_B: (d × r) - Zero initialization (important!)
            loraB_name = [field, '_LoRA_B'];
            net.(loraB_name) = zeros(d, rank);

            % Scaling
            loraScale_name = [field, '_LoRA_scale'];
            net.(loraScale_name) = loraAlpha / rank;

            loraFields{end+1} = field;

            fprintf('  %s.%s: (%d × %d) -> LoRA rank %d (%d params)\n', ...
                    networkName, field, d, k, rank, rank*(d+k));
        end
    end

    net.loraFields = loraFields;
    net.loraRank = rank;
    net.loraAlpha = loraAlpha;
end

function frozenWeights = extractBaseWeights(agent)
    % Extract and store base weights for freezing
    frozenWeights = struct();

    % Actor weights
    frozenWeights.actor = extractNetworkWeights(agent.actor);

    % Critic weights
    if isfield(agent, 'critic1')
        frozenWeights.critic1 = extractNetworkWeights(agent.critic1);
        frozenWeights.critic2 = extractNetworkWeights(agent.critic2);
    else
        frozenWeights.critic = extractNetworkWeights(agent.critic);
    end
end

function weights = extractNetworkWeights(net)
    % Extract all weight and bias parameters
    weights = struct();
    fields = fieldnames(net);

    for i = 1:length(fields)
        field = fields{i};
        if isnumeric(net.(field)) && ~contains(field, 'LoRA')
            weights.(field) = net.(field);
        end
    end
end

function count = countLoRAParameters(agent)
    % Count total trainable LoRA parameters
    count = 0;

    % Actor LoRA params
    count = count + countNetworkLoRAParams(agent.actor);

    % Critic LoRA params
    if isfield(agent, 'critic1')
        count = count + countNetworkLoRAParams(agent.critic1);
        count = count + countNetworkLoRAParams(agent.critic2);
    else
        count = count + countNetworkLoRAParams(agent.critic);
    end
end

function count = countNetworkLoRAParams(net)
    % Count LoRA parameters in a single network
    count = 0;
    fields = fieldnames(net);

    for i = 1:length(fields)
        field = fields{i};
        if contains(field, 'LoRA_A') || contains(field, 'LoRA_B')
            count = count + numel(net.(field));
        end
    end
end

function count = countBaseParameters(agent)
    % Count frozen base parameters
    count = 0;

    % Actor params
    count = count + countNetworkBaseParams(agent.actor);

    % Critic params
    if isfield(agent, 'critic1')
        count = count + countNetworkBaseParams(agent.critic1);
        count = count + countNetworkBaseParams(agent.critic2);
    else
        count = count + countNetworkBaseParams(agent.critic);
    end
end

function count = countNetworkBaseParams(net)
    % Count base parameters (weights and biases, excluding LoRA)
    count = 0;
    fields = fieldnames(net);

    for i = 1:length(fields)
        field = fields{i};
        if isnumeric(net.(field)) && ~contains(field, 'LoRA') && ...
           (startsWith(field, 'W') || startsWith(field, 'b'))
            count = count + numel(net.(field));
        end
    end
end
