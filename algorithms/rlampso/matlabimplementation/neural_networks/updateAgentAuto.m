function agent = updateAgentAuto(agent, state, action, reward)
    % Automatic Agent Update with Fallback
    %
    % Intelligently chooses between DL Toolbox and custom training
    % based on agent type
    %
    % Inputs:
    %   agent: Agent structure (DL Toolbox or custom)
    %   state: Current state (for next iteration's training)
    %   action: Action taken (stored for training)
    %   reward: Reward received (stored for training)
    %
    % Outputs:
    %   agent: Updated agent with trained networks
    %
    % Usage:
    %   agent = updateAgentAuto(agent, nextState, [], reward);

    % Store experience in replay buffer
    if ~isempty(agent.lastState)
        experience = struct();
        experience.state = agent.lastState;
        experience.action = agent.lastAction;
        experience.reward = reward;
        experience.nextState = state;
        experience.done = false;

        % Add to replay buffer
        if agent.usePER
            % Prioritized replay
            agent.replayBuffer.add(experience);
        else
            % Uniform replay
            if length(agent.replayBuffer) < agent.bufferSize
                agent.replayBuffer = [agent.replayBuffer, experience];
            else
                agent.replayBuffer(agent.bufferIndex) = experience;
                agent.bufferIndex = mod(agent.bufferIndex, agent.bufferSize) + 1;
            end
        end
    end

    % Train networks if enough experiences
    bufferSize = getBufferSize(agent);
    if bufferSize >= agent.batchSize
        % Choose training function based on implementation type
        if isfield(agent, 'useDLToolbox') && agent.useDLToolbox
            % Use DL Toolbox training (automatic differentiation)
            if isfield(agent, 'useTD3') && agent.useTD3
                agent = trainDLToolboxTD3(agent);
            else
                agent = trainDLToolboxDDPG(agent);
            end
        else
            % Use custom training (manual gradients)
            if isfield(agent, 'useTD3') && agent.useTD3
                agent = trainTD3Networks(agent);
            elseif isfield(agent, 'critic2')
                % Has twin critics, must be TD3
                agent = trainTD3Networks(agent);
            else
                agent = trainPaperExactNetworks(agent);
            end
        end
    end
end

function bufferSize = getBufferSize(agent)
    % Get current replay buffer size
    if agent.usePER
        bufferSize = agent.replayBuffer.size();
    else
        bufferSize = length(agent.replayBuffer);
    end
end
