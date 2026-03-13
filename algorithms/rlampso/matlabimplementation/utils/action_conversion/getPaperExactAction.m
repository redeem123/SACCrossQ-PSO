function [subgroupParams, rlamAgent] = getPaperExactAction(rlamAgent, state, iter)
    % PAPER: Get 20-dimensional action and convert using Equation 11

    % PAPER: Forward pass through actor network
    % Auto-detect DL Toolbox vs custom implementation
    if isa(rlamAgent.actor, 'dlnetwork')
        % Use DL Toolbox prediction
        action = predictAction(rlamAgent.actor, state, false);
    else
        % Use custom forward pass
        action = forwardPassPaperActor(rlamAgent.actor, state);
    end
    
    % PAPER Equation 14: Add exploration noise
    % Note: Noise level is constant during episode, decays BETWEEN episodes
    if isfield(rlamAgent, 'explorationNoise')
        currentNoise = rlamAgent.explorationNoise;
    else
        % FIXED: Default to 0 for testing/inference (not 0.1!)
        % High noise during testing degrades performance
        currentNoise = 0.0;
    end

    % FIXED: For testing/inference, use minimal noise (deterministic policy)
    % Check if we're in test mode (no training fields present)
    isTestMode = ~isfield(rlamAgent, 'replayBuffer');
    if isTestMode
        currentNoise = 0.0;  % Zero noise for deterministic testing
    else
        % During training: clamp to at most 5% noise for stability
        currentNoise = min(currentNoise, 0.05);
    end

    explorationNoise = randn(size(action)) * currentNoise;
    noisyAction = action + explorationNoise;

    % DEBUG: Log raw action before clipping (sample first 3 values)
    if mod(iter, 100) == 0 && nargin >= 3
        fprintf('[ACTION DEBUG iter %d] Raw action[1:3]: [%.4f, %.4f, %.4f], noise: %.4f\n', ...
                iter, action(1), action(2), action(3), currentNoise);
    end

    % FIXED: Clip actions to valid range [-1, 1] (matching Python TF2_DDPG_Basic.py:147)
    % This ensures network only sees valid action values during training
    % Prevents TD error spikes and training instability from out-of-range actions
    beforeClip = noisyAction;
    noisyAction = max(-1.0, min(1.0, noisyAction));

    % DEBUG: Check if clipping occurred
    if mod(iter, 100) == 0 && nargin >= 3
        numClipped = sum(abs(beforeClip) > 1.0);
        if numClipped > 0
            fprintf('[ACTION DEBUG iter %d] WARNING: %d/%d actions clipped (network saturating)\n', ...
                    iter, numClipped, length(beforeClip));
        end
    end

    % PAPER Equation 11: Convert 20D action to 5×3 PSO parameters
    subgroupParams = convertActionToParameters(noisyAction);
    
    % Store for experience replay
    rlamAgent.lastState = state;
    rlamAgent.lastAction = action;
end

