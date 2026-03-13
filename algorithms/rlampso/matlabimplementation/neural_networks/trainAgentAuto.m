function agent = trainAgentAuto(agent, nextState, action, reward)
    % DL Toolbox Agent Training (REQUIRED)
    %
    % Uses MATLAB Deep Learning Toolbox for training with automatic differentiation
    % Now supports multiple gradient steps per call for better GPU utilization
    %
    % Inputs:
    %   agent: DL Toolbox agent structure
    %   nextState: Next state for training (not used directly)
    %   action: Action taken (not used, kept for compatibility)
    %   reward: Reward received (not used directly)
    %
    % Outputs:
    %   agent: Updated agent with trained networks
    %
    % Usage (in PSO training loop):
    %   agent = trainAgentAuto(agent, nextState, [], reward);

    % Check if enough experiences in buffer
    bufferSize = getBufferSize(agent);
    if bufferSize < agent.batchSize
        return;  % Not enough data to train yet
    end

    % ===== MULTIPLE GRADIENT STEPS FOR GPU UTILIZATION (2025 Optimization) =====
    % Perform multiple gradient updates per call to maximize GPU usage
    % Research-backed: UTD ratio of 2-8 is optimal for TD3
    numGradientSteps = 1;  % Default: 1 step (backward compatible)
    if isfield(agent, 'gradientStepsPerTrainingCall')
        numGradientSteps = agent.gradientStepsPerTrainingCall;
    end

    % Perform multiple training steps
    for step = 1:numGradientSteps
        % Use DL Toolbox training (automatic differentiation)
        if isfield(agent, 'useTD3') && agent.useTD3
            agent = trainDLToolboxTD3(agent);
        else
            agent = trainDLToolboxDDPG(agent);
        end
    end
end

function bufferSize = getBufferSize(agent)
    % Get current replay buffer size
    if isfield(agent, 'usePER') && agent.usePER
        if isfield(agent.replayBuffer, 'size')
            bufferSize = agent.replayBuffer.size();
        else
            bufferSize = length(agent.replayBuffer);
        end
    else
        bufferSize = length(agent.replayBuffer);
    end
end
