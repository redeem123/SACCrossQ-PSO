function rlamAgent = updatePaperExactDDPG(rlamAgent, state, action, reward)
    % PAPER Algorithm 1: Update DDPG with experience replay
    
    % Store experience
    if ~isempty(rlamAgent.lastState)
        experience = struct();
        experience.state = rlamAgent.lastState;
        experience.action = rlamAgent.lastAction;
        experience.reward = reward;
        experience.nextState = state;
        experience.done = false;
        
        % Add to replay buffer
        if length(rlamAgent.replayBuffer) < rlamAgent.bufferSize
            rlamAgent.replayBuffer = [rlamAgent.replayBuffer, experience];
        else
            rlamAgent.replayBuffer(rlamAgent.bufferIndex) = experience;
            rlamAgent.bufferIndex = mod(rlamAgent.bufferIndex, rlamAgent.bufferSize) + 1;
        end
    end
    
    % Train if enough experiences
    if length(rlamAgent.replayBuffer) >= rlamAgent.batchSize
        rlamAgent = trainPaperExactNetworks(rlamAgent);
    end
    
    rlamAgent.trainStep = rlamAgent.trainStep + 1;
    
    % Soft update target networks
    if mod(rlamAgent.trainStep, 1) == 0
        rlamAgent = softUpdatePaperTargets(rlamAgent);
    end
end

