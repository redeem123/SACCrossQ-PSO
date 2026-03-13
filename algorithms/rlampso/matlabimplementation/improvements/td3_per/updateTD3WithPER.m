function td3Agent = updateTD3WithPER(td3Agent, state, action, reward)
    % Update TD3 agent with experience and train networks
    % Replaces updatePaperExactDDPG.m with TD3+PER improvements
    %
    % Inputs:
    %   td3Agent: TD3 agent structure
    %   state: Current state
    %   action: Action taken (can be empty if using stored lastAction)
    %   reward: Reward received
    %
    % Outputs:
    %   td3Agent: Updated agent

    % Store experience in replay buffer
    if ~isempty(td3Agent.lastState)
        % Create experience struct
        experience = struct();
        experience.state = td3Agent.lastState;
        experience.action = td3Agent.lastAction;
        experience.reward = reward;
        experience.nextState = state;
        experience.done = false;

        % Add to replay buffer
        if td3Agent.usePER
            % Add to prioritized replay buffer
            % Initially use max priority (will be updated after training)
            td3Agent.replayBuffer.add(experience);
        else
            % Add to standard replay buffer
            if length(td3Agent.replayBuffer) < td3Agent.bufferSize
                td3Agent.replayBuffer = [td3Agent.replayBuffer, experience];
            else
                % Circular buffer
                td3Agent.replayBuffer(td3Agent.bufferIndex) = experience;
                td3Agent.bufferIndex = mod(td3Agent.bufferIndex, td3Agent.bufferSize) + 1;
            end
        end
    end

    % Train networks if enough experiences collected
    minExperiences = td3Agent.batchSize;

    if td3Agent.usePER
        bufferSize = td3Agent.replayBuffer.length();
    else
        bufferSize = length(td3Agent.replayBuffer);
    end

    if bufferSize >= minExperiences
        % Train TD3 networks - auto-detect implementation
        if isfield(td3Agent, 'useDLToolbox') && td3Agent.useDLToolbox
            % Use DL Toolbox training (dlnetwork + automatic differentiation)
            td3Agent = trainDLToolboxTD3(td3Agent);
        else
            % Use custom training (manual gradients)
            td3Agent = trainTD3Networks(td3Agent);
        end
    end
end
