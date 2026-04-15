function agent = trainAgentAuto(agent, nextState, action, reward)
    % Train DDPG agent (DL Toolbox, paper Algorithm 1 steps 9-10)
    %
    % Inputs:
    %   agent:     DL Toolbox agent structure
    %   nextState: Next state (stored but not used directly here)
    %   action:    Unused, kept for interface compatibility
    %   reward:    Reward (stored but not used directly here)

    if length(agent.replayBuffer) < agent.batchSize
        return;  % Not enough data yet
    end

    agent = trainDLToolboxDDPG(agent);
end
