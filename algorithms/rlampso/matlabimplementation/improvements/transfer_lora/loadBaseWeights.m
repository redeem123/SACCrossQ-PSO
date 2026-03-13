function agent = loadBaseWeights(filepath, stateSize, useTD3, usePER)
    % Load pretrained base weights for transfer learning
    %
    % Inputs:
    %   filepath: Path to saved base weights
    %   stateSize: State dimension (15 for sin-encoding, 64 for transformer)
    %   useTD3: Use TD3 instead of DDPG (default: true)
    %   usePER: Use Prioritized Experience Replay (default: true)
    %
    % Outputs:
    %   agent: Agent with loaded base weights

    if nargin < 3, useTD3 = true; end
    if nargin < 4, usePER = true; end

    fprintf('Loading base weights from: %s\n', filepath);

    if ~exist(filepath, 'file')
        error('Base weights file not found: %s', filepath);
    end

    % Load saved weights
    loaded = load(filepath);
    config = loaded.config;

    fprintf('  Saved state size: %d\n', config.stateSize);
    fprintf('  Saved action size: %d\n', config.actionSize);

    % Check compatibility
    actionSize = config.actionSize;

    if config.stateSize ~= stateSize
        warning('State size mismatch! Saved: %d, Requested: %d\nAttempting to adapt...', ...
                config.stateSize, stateSize);

        % Need to reinitialize input layer with new state size
        % This is a limitation - full transfer requires matching state dimensions
        % Or implement adapter layers
    end

    % Initialize agent with appropriate algorithm
    if useTD3
        agent = initializeTD3Agent(stateSize, actionSize, usePER, '');
    else
        agent = initializePaperExactDDPGAgent('');
        agent.usePER = usePER;
        if usePER
            alpha = 0.6;
            beta_start = 0.4;
            beta_frames = 100000;
            agent.replayBuffer = PrioritizedReplayBuffer(agent.bufferSize, alpha, beta_start, beta_frames);
        end
    end

    % Load actor weights
    if config.stateSize == stateSize
        % Direct loading - dimensions match
        agent.actor = loadNetworkWeights(agent.actor, loaded.baseActor);
        agent.targetActor = agent.actor;
        fprintf('✓ Actor weights loaded\n');
    else
        % Partial loading - keep input layer random, load hidden layers
        agent.actor = loadNetworkWeightsPartial(agent.actor, loaded.baseActor);
        agent.targetActor = agent.actor;
        fprintf('⚠ Actor weights partially loaded (state dimension mismatch)\n');
    end

    % Load critic weights
    if useTD3 && isfield(loaded, 'baseCritic1')
        % Load twin critics
        agent.critic1 = loadNetworkWeights(agent.critic1, loaded.baseCritic1);
        agent.critic2 = loadNetworkWeights(agent.critic2, loaded.baseCritic2);
        agent.targetCritic1 = agent.critic1;
        agent.targetCritic2 = agent.critic2;
        fprintf('✓ Twin critic weights loaded\n');

    elseif ~useTD3 && isfield(loaded, 'baseCritic')
        % Load single critic
        agent.critic = loadNetworkWeights(agent.critic, loaded.baseCritic);
        agent.targetCritic = agent.critic;
        fprintf('✓ Critic weights loaded\n');

    else
        warning('Critic type mismatch or missing. Using random initialization.');
    end

    % Display training info if available
    if isfield(config, 'trainingInfo')
        info = config.trainingInfo;
        fprintf('\nPretrained Model Info:\n');

        if isfield(info, 'numEpisodes')
            fprintf('  Episodes trained: %d\n', info.numEpisodes);
        end
        if isfield(info, 'finalAvgReward')
            fprintf('  Final avg reward: %.2f\n', info.finalAvgReward);
        end
        if isfield(info, 'trainingTime')
            fprintf('  Training time: %.1f minutes\n', info.trainingTime / 60);
        end
    end

    fprintf('\n✓ Base weights loaded successfully\n');
    fprintf('  Ready for LoRA fine-tuning or continued training\n\n');
end

function targetNet = loadNetworkWeights(targetNet, sourceNet)
    % Load weights from source to target network
    fields = fieldnames(sourceNet);

    for i = 1:length(fields)
        field = fields{i};

        % Only load weight and bias parameters
        if isnumeric(sourceNet.(field)) && ...
           (startsWith(field, 'W') || startsWith(field, 'b'))

            if isfield(targetNet, field)
                % Check dimension compatibility
                if isequal(size(targetNet.(field)), size(sourceNet.(field)))
                    targetNet.(field) = sourceNet.(field);
                else
                    warning('Dimension mismatch for %s. Skipping...', field);
                end
            end
        end
    end
end

function targetNet = loadNetworkWeightsPartial(targetNet, sourceNet)
    % Partial weight loading - skip input layer if dimensions mismatch
    fields = fieldnames(sourceNet);

    for i = 1:length(fields)
        field = fields{i};

        % Skip input layer (W1, b1) if dimensions don't match
        if (strcmp(field, 'W1') || strcmp(field, 'b1'))
            if ~isequal(size(targetNet.(field)), size(sourceNet.(field)))
                fprintf('  Skipping %s (dimension mismatch)\n', field);
                continue;
            end
        end

        % Load other layers
        if isnumeric(sourceNet.(field)) && ...
           (startsWith(field, 'W') || startsWith(field, 'b'))

            if isfield(targetNet, field)
                if isequal(size(targetNet.(field)), size(sourceNet.(field)))
                    targetNet.(field) = sourceNet.(field);
                else
                    warning('Dimension mismatch for %s. Skipping...', field);
                end
            end
        end
    end
end
