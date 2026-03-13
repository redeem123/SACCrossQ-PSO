function reward = calculateContinuousReward(fitnessImprovement, config)
    % Calculate Continuous Reward Signal for RLAMPSO
    % FIXED: Replaces binary reward with magnitude-aware signal
    %
    % The paper uses binary reward (+1/-1), but this is suboptimal for DDPG learning.
    % This function provides better reward shaping for continuous learning.
    %
    % Inputs:
    %   fitnessImprovement: deltaFitness = previousBest - currentBest
    %                       Positive values = improvement
    %   config: Configuration struct (optional, for reward type selection)
    %
    % Outputs:
    %   reward: Continuous reward signal, typically in [-1, 1]
    %
    % Reward Shaping Options:
    %   1. Binary (original paper): +1 if improvement, -1 otherwise
    %   2. Magnitude-aware: -log(1 + |improvement|) for improvements
    %   3. Scaled penalty: abs(improvement) * scale factor
    %   4. Log scale: sign(improvement) * log(1 + abs(improvement))

    % Default: Use magnitude-aware reward if config not specified
    if nargin < 2 || isempty(config)
        rewardType = 'magnitude_aware';
    elseif isfield(config, 'rewardType')
        rewardType = config.rewardType;
    else
        rewardType = 'magnitude_aware';
    end

    switch lower(rewardType)
        case 'binary'
            % PAPER EXACT: Binary reward (original but suboptimal)
            epsilon = 1e-6;
            if fitnessImprovement > epsilon
                reward = 1.0;      % Improvement
            else
                reward = -1.0;     % Stagnation
            end

        case 'magnitude_aware'
            % FIXED: Sparse reward - only reward improvements, no penalty for stagnation
            %
            % CRITICAL INSIGHT: PSO naturally has 80%+ iterations with no improvement
            % (this is normal swarm behavior). Penalizing stagnation teaches the network
            % that "all my actions are bad", leading to policy collapse and saturation.
            %
            % NEW APPROACH: Sparse rewards
            %   - Improvement: positive reward (log scale)
            %   - Stagnation: ZERO reward (not -0.1!)
            % This allows network to learn "these actions are neutral, not bad"
            epsilon = 1e-6;

            if fitnessImprovement > epsilon
                % Improvement: POSITIVE reward (lower fitness is better)
                % Use log scale to reward both small and large improvements
                %
                % Scaling for UAV path planning (typical fitness: 1800-2600):
                %   Improvement = 0.1 → reward ~ 0.05
                %   Improvement = 1.0 → reward ~ 0.30
                %   Improvement = 10  → reward ~ 0.70
                %   Improvement = 100 → reward ~ 1.00
                %
                % Formula: reward = log10(1 + improvement) / 2.0
                %   - Dividing by 2.0 scales log10(101) ≈ 2.0 to reward = 1.0
                reward = log10(1.0 + fitnessImprovement) / 2.0;
                % Clip to [0, 1] range for safety
                reward = max(0.0, min(1.0, reward));

                % DEBUG: Log positive rewards
                if mod(randi(100), 20) == 0  % Print ~5% of the time
                    fprintf('[REWARD DEBUG] Improvement: %.4f → Reward: +%.4f\n', fitnessImprovement, reward);
                end
            else
                % FIXED: Zero reward for stagnation (was -0.1)
                % Stagnation is NORMAL in PSO, not a failure!
                reward = 0.0;

                % DEBUG: Log zero rewards occasionally
                if mod(randi(1000), 10) == 0  % Print ~1% of the time
                    fprintf('[REWARD DEBUG] Stagnation (normal): %.4f → Reward: 0.0\n', fitnessImprovement);
                end
            end

        case 'scaled_penalty'
            % Proportional penalty for stagnation
            epsilon = 1e-6;

            if fitnessImprovement > epsilon
                % Improvement: reward based on magnitude
                reward = min(1.0, fitnessImprovement);
            else
                % Stagnation: scaled penalty
                reward = -min(1.0, abs(fitnessImprovement) * 0.01);
            end

        case 'log_scale'
            % Logarithmic scaling for both improvements and stagnation
            if fitnessImprovement > 1e-6
                % Improvement: log scale helps differentiate
                reward = log(1.0 + fitnessImprovement);
                reward = min(1.0, reward);
            else
                % Stagnation: negative log penalty
                reward = -log(1.0 + abs(fitnessImprovement));
                reward = max(-1.0, reward);
            end

        otherwise
            % Default to magnitude-aware if unknown type
            error('Unknown reward type: %s. Use "binary", "magnitude_aware", "scaled_penalty", or "log_scale"', rewardType);
    end

    % Final clipping to ensure reward is in [-1, 1]
    reward = max(-1.0, min(1.0, reward));
end
