function reward = calculateRLNNPSOReward(prevFitness, newFitness, prevBestFitness, newBestFitness, stagnationDuration, swarmDiversity)
    % Simplified reward function for RL-NNPSO (3-feature state approach)
    % Focuses on fitness improvement, diversity, and stagnation escape
    %
    % Inputs:
    %   prevFitness: Previous particle fitness
    %   newFitness: New particle fitness
    %   prevBestFitness: Previous personal best fitness
    %   newBestFitness: New personal best fitness
    %   stagnationDuration: Iterations without improvement
    %   swarmDiversity: Current swarm diversity [0,1]
    %
    % Output:
    %   reward: Scalar reward value

    reward = 0.0;
    epsilon = 1e-6;

    % =========================================================================
    % 1. FITNESS IMPROVEMENT REWARD (Primary signal)
    % =========================================================================
    % Current fitness improvement
    if abs(prevFitness) > epsilon
        if newFitness < prevFitness  % Improvement (minimization)
            fitnessImprovement = (prevFitness - newFitness) / abs(prevFitness);
            reward = reward + 20.0 * fitnessImprovement;  % Strong positive reward
        else  % Degradation
            fitnessDegradation = (newFitness - prevFitness) / abs(prevFitness);
            reward = reward - 1.0 * min(fitnessDegradation, 1.0);  % Small penalty
        end
    end

    % Personal best improvement (highest priority)
    if abs(prevBestFitness) > epsilon
        if newBestFitness < prevBestFitness  % New personal best
            bestImprovement = (prevBestFitness - newBestFitness) / abs(prevBestFitness);
            reward = reward + 30.0 * bestImprovement;  % Very strong reward
        end
    end

    % =========================================================================
    % 2. DIVERSITY BONUS
    % =========================================================================
    % Reward maintaining good diversity
    if swarmDiversity > 0.2  % Good diversity
        reward = reward + 3.0;  % Strong reward for maintaining diversity
    elseif swarmDiversity < 0.1  % Low diversity - mild encouragement
        reward = reward + 1.0;  % Smaller bonus to encourage exploration
    end

    % =========================================================================
    % 3. STAGNATION ESCAPE REWARD
    % =========================================================================
    if stagnationDuration > 10  % Particle was stuck
        if newFitness < prevFitness
            % Reward escaping stagnation (scaled by stagnation duration)
            stagnationBonus = 5.0 * min(stagnationDuration / 20, 1.0);
            reward = reward + stagnationBonus;
        end
    end

    % =========================================================================
    % 4. NO IMPROVEMENT PENALTY (Gentle, diversity-aware)
    % =========================================================================
    if newFitness >= prevFitness && newBestFitness >= prevBestFitness
        % Scale penalty by diversity - less penalty when diversity is low
        penaltyScale = max(0.3, swarmDiversity);
        reward = reward - 0.5 * penaltyScale;
    end

    % =========================================================================
    % 5. CLIP REWARD (Prevent extreme values)
    % =========================================================================
    reward = max(-10.0, min(50.0, reward));

    % =========================================================================
    % 6. NUMERICAL SAFETY
    % =========================================================================
    if ~isfinite(reward)
        reward = -1.0;  % Default penalty for numerical errors
    end
end
