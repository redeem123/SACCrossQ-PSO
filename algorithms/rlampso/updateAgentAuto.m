function agent = updateAgentAuto(agent, state, action, reward)
    % Store experience and train DDPG (paper Algorithm 1 steps 8-10)
    %
    % Inputs:
    %   agent:  DL Toolbox agent structure
    %   state:  Next state (s_{t+1})
    %   action: Unused, kept for interface compatibility
    %   reward: Reward r_t

    % Store experience in uniform replay buffer (Paper step 8)
    if ~isempty(agent.lastState)
        exp = struct('state', agent.lastState, ...
                     'action', agent.lastAction, ...
                     'reward', reward, ...
                     'nextState', state, ...
                     'done', false);

        if length(agent.replayBuffer) < agent.bufferSize
            agent.replayBuffer = [agent.replayBuffer, exp];
        else
            agent.replayBuffer(agent.bufferIndex) = exp;
            agent.bufferIndex = mod(agent.bufferIndex, agent.bufferSize) + 1;
        end
    end

    % Train DDPG if enough experiences (Paper steps 9-10)
    if length(agent.replayBuffer) >= agent.batchSize
        agent = trainDLToolboxDDPG(agent);
    end
end
