function rlamAgent = initializePaperExactDDPGAgent(pretrainedNetworkPath)
    % Initialize DDPG agent exactly as specified in paper
    %
    % Inputs:
    %   pretrainedNetworkPath (optional): Path to pre-trained networks .mat file
    %                                     If not provided, initializes randomly

    rlamAgent = struct();

    % PAPER: State dimensions (3 basic x 5 sin-encoding = 15)
    rlamAgent.stateSize = 15;

    % PAPER: Action dimensions (5 subgroups x 4 parameters = 20)
    rlamAgent.actionSize = 20;
    rlamAgent.numSubgroups = 5;

    % DDPG hyperparameters
    rlamAgent.gamma = 0.99;
    rlamAgent.tau = 0.001;
    rlamAgent.actorLR = 0.0001;
    rlamAgent.criticLR = 0.001;
    rlamAgent.batchSize = 32;
    rlamAgent.bufferSize = 10000;

    % GPU settings
    rlamAgent.useGPU = canUseGPU();
    if rlamAgent.useGPU
        fprintf('GPU detected and will be used for training acceleration\n');
    else
        fprintf('GPU not available or not enabled, using CPU\n');
    end

    % Check if pre-trained networks should be loaded
    if nargin >= 1 && ~isempty(pretrainedNetworkPath) && exist(pretrainedNetworkPath, 'file')
        % Load pre-trained networks
        fprintf('Loading pre-trained networks from: %s\n', pretrainedNetworkPath);
        loaded = load(pretrainedNetworkPath);

        rlamAgent.actor = loaded.trainedActor;
        rlamAgent.critic = loaded.trainedCritic;
        rlamAgent.targetActor = loaded.trainedTargetActor;
        rlamAgent.targetCritic = loaded.trainedTargetCritic;

        % Move loaded networks to GPU if available
        if rlamAgent.useGPU
            rlamAgent.actor = moveNetworkToGPU(rlamAgent.actor);
            rlamAgent.critic = moveNetworkToGPU(rlamAgent.critic);
            rlamAgent.targetActor = moveNetworkToGPU(rlamAgent.targetActor);
            rlamAgent.targetCritic = moveNetworkToGPU(rlamAgent.targetCritic);
            fprintf('Loaded networks moved to GPU\n');
        end

        disp('Pre-trained DDPG Agent loaded successfully!');

        % Display training info if available
        if isfield(loaded, 'trainingInfo')
            fprintf('  Training episodes: %d\n', loaded.trainingInfo.numEpisodes);
            fprintf('  Final avg reward: %.2f\n', loaded.trainingInfo.finalAvgReward);
        end
    else
        % PAPER: Initialize networks randomly with exact architecture (Section 4.1.6)
        if nargin >= 1 && ~isempty(pretrainedNetworkPath)
            warning('Pre-trained network file not found: %s\nInitializing randomly instead.', pretrainedNetworkPath);
        end

        rlamAgent.actor = initializePaperActorNetwork(rlamAgent.stateSize, rlamAgent.actionSize);
        rlamAgent.critic = initializePaperCriticNetwork(rlamAgent.stateSize, rlamAgent.actionSize);
        rlamAgent.targetActor = rlamAgent.actor;
        rlamAgent.targetCritic = rlamAgent.critic;

        % Move networks to GPU if available
        if rlamAgent.useGPU
            rlamAgent.actor = moveNetworkToGPU(rlamAgent.actor);
            rlamAgent.critic = moveNetworkToGPU(rlamAgent.critic);
            rlamAgent.targetActor = moveNetworkToGPU(rlamAgent.targetActor);
            rlamAgent.targetCritic = moveNetworkToGPU(rlamAgent.targetCritic);
            fprintf('Networks moved to GPU\n');
        end

        disp('Paper-Exact DDPG Agent initialized randomly: 15-state, 20-action, exact architecture');
    end

    % Experience replay buffer
    rlamAgent.replayBuffer = [];
    rlamAgent.bufferIndex = 1;

    % Training statistics
    rlamAgent.trainStep = 0;
    rlamAgent.lastState = [];
    rlamAgent.lastAction = [];
end
