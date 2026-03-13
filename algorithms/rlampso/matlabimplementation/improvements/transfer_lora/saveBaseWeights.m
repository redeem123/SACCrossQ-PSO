function saveBaseWeights(agent, filepath, trainingInfo)
    % Save pretrained base weights for transfer learning
    %
    % Inputs:
    %   agent: Trained agent with base networks
    %   filepath: Path to save weights (e.g., 'models/pretrained_base.mat')
    %   trainingInfo: (optional) Training metadata

    fprintf('Saving base weights to: %s\n', filepath);

    % Ensure directory exists
    [saveDir, ~, ~] = fileparts(filepath);
    if ~isempty(saveDir) && ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    % Extract base networks (move from GPU if needed)
    if agent.useGPU
        baseActor = moveNetworkToCPU(agent.actor);

        if isfield(agent, 'critic1')
            baseCritic1 = moveNetworkToCPU(agent.critic1);
            baseCritic2 = moveNetworkToCPU(agent.critic2);
        else
            baseCritic = moveNetworkToCPU(agent.critic);
        end
    else
        baseActor = agent.actor;

        if isfield(agent, 'critic1')
            baseCritic1 = agent.critic1;
            baseCritic2 = agent.critic2;
        else
            baseCritic = agent.critic;
        end
    end

    % Remove LoRA parameters if present
    baseActor = removeLoRAParameters(baseActor);

    if isfield(agent, 'critic1')
        baseCritic1 = removeLoRAParameters(baseCritic1);
        baseCritic2 = removeLoRAParameters(baseCritic2);
    else
        baseCritic = removeLoRAParameters(baseCritic);
    end

    % Save configuration
    config = struct();
    config.stateSize = agent.stateSize;
    config.actionSize = agent.actionSize;
    config.numSubgroups = agent.numSubgroups;
    config.useTD3 = isfield(agent, 'critic1');
    config.savedAt = datetime('now');

    % Save training info if provided
    if nargin >= 3
        config.trainingInfo = trainingInfo;
    end

    % Save to file
    if isfield(agent, 'critic1')
        % TD3
        save(filepath, 'baseActor', 'baseCritic1', 'baseCritic2', 'config', '-v7.3');
        fprintf('✓ TD3 base weights saved (Actor + Twin Critics)\n');
    else
        % DDPG
        save(filepath, 'baseActor', 'baseCritic', 'config', '-v7.3');
        fprintf('✓ DDPG base weights saved (Actor + Critic)\n');
    end

    fprintf('  State size: %d\n', config.stateSize);
    fprintf('  Action size: %d\n', config.actionSize);

    % Count parameters
    actorParams = countNetworkParameters(baseActor);
    if isfield(agent, 'critic1')
        criticParams = countNetworkParameters(baseCritic1) + countNetworkParameters(baseCritic2);
        fprintf('  Total parameters: %d (Actor: %d, Critics: %d)\n', ...
                actorParams + criticParams, actorParams, criticParams);
    else
        criticParams = countNetworkParameters(baseCritic);
        fprintf('  Total parameters: %d (Actor: %d, Critic: %d)\n', ...
                actorParams + criticParams, actorParams, criticParams);
    end
end

function net = removeLoRAParameters(net)
    % Remove LoRA-specific fields from network
    fields = fieldnames(net);
    loraFields = {};

    for i = 1:length(fields)
        field = fields{i};
        if contains(field, 'LoRA') || strcmp(field, 'loraFields') || ...
           strcmp(field, 'loraRank') || strcmp(field, 'loraAlpha')
            loraFields{end+1} = field;
        end
    end

    % Remove LoRA fields
    for i = 1:length(loraFields)
        net = rmfield(net, loraFields{i});
    end
end

function net = moveNetworkToCPU(net)
    % Move all network weights from GPU to CPU
    fields = fieldnames(net);

    for i = 1:length(fields)
        field = fields{i};
        if isa(net.(field), 'gpuArray')
            net.(field) = gather(net.(field));
        end
    end
end

function count = countNetworkParameters(net)
    % Count total parameters in network
    count = 0;
    fields = fieldnames(net);

    for i = 1:length(fields)
        field = fields{i};
        if isnumeric(net.(field)) && (startsWith(field, 'W') || startsWith(field, 'b'))
            count = count + numel(net.(field));
        end
    end
end
