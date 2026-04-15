function [subgroupParams, rlamAgent] = getPaperExactAction(rlamAgent, state, iter)
    % Get action from actor network and convert to PSO parameters
    %
    % Paper Eq. 14: a_t = Actor(s_t) + N(0, 0.5)
    % Paper Eq. 11: Convert 20D action → 5×[w, c1, c2]

    % Forward pass through actor (DL Toolbox dlnetwork)
    action = predictAction(rlamAgent.actor, state, false);

    % Exploration noise (Paper Eq. 14)
    if isfield(rlamAgent, 'explorationNoise')
        noiseStd = rlamAgent.explorationNoise;
    else
        noiseStd = 0.0;  % Inference mode: no noise
    end

    % No noise during inference (no replay buffer = test mode)
    if ~isfield(rlamAgent, 'replayBuffer')
        noiseStd = 0.0;
    end

    noisyAction = action + randn(size(action)) * noiseStd;
    noisyAction = max(-1.0, min(1.0, noisyAction));

    % Convert to PSO parameters (Paper Eq. 11)
    subgroupParams = convertActionToParameters(noisyAction);

    % Store for experience replay
    rlamAgent.lastState  = state;
    rlamAgent.lastAction = action;
end
