function [reward, components] = calculateMultiObjectiveReward(currentFitness, previousFitness, ...
    currentDiversity, diversityHistory, stateHistory, config)
    % Calculate multi-objective reward for APEX-PSO
    %
    % Combines multiple objectives for better learning:
    %   1. Fitness improvement (primary goal)
    %   2. Diversity maintenance (avoid premature convergence)
    %   3. Convergence speed (reward fast improvements)
    %   4. Curiosity bonus (encourage exploration of new states)
    %
    % Based on research:
    %   - Multi-objective RL for better exploration-exploitation balance
    %   - Curiosity-driven exploration (intrinsic motivation)
    %
    % Inputs:
    %   currentFitness: Current global best fitness
    %   previousFitness: Previous global best fitness
    %   currentDiversity: Current population diversity metric
    %   diversityHistory: Array of recent diversity values
    %   stateHistory: Array of recent states (for curiosity)
    %   config: APEX-PSO configuration
    %
    % Outputs:
    %   reward: Total reward (scalar)
    %   components: Struct with individual reward components

    epsilon = 1e-6;

    % ===== 1. FITNESS IMPROVEMENT REWARD (Primary) =====
    % Positive if fitness improves (minimization), negative if no improvement
    
    % Define threshold for "invalid" fitness (since we now clamp Inf to 1e9)
    INVALID_FITNESS_THRESHOLD = 1e8;

    % Handle invalid/valid transitions
    isPreviousInvalid = (previousFitness >= INVALID_FITNESS_THRESHOLD);
    isCurrentInvalid = (currentFitness >= INVALID_FITNESS_THRESHOLD);

    if isPreviousInvalid && isCurrentInvalid
        fitnessImprovement = 0;
        fitnessReward = -0.5; % Treat as stagnation
    elseif isPreviousInvalid && ~isCurrentInvalid
        % First valid solution found! Huge reward.
        fitnessImprovement = 10000; 
        fitnessReward = 5.0; % High reward
    elseif ~isPreviousInvalid && isCurrentInvalid
        % Lost valid solution (should rarely happen with elitism). Huge penalty.
        fitnessImprovement = -10000;
        fitnessReward = -5.0;
    else
        % Normal case
        fitnessImprovement = previousFitness - currentFitness;

        if fitnessImprovement > epsilon
            % Improvement: Use log-scale for large improvements
            fitnessReward = sign(fitnessImprovement) * log10(1.0 + abs(fitnessImprovement));
        elseif abs(fitnessImprovement) <= epsilon
            % Stagnation: Small penalty
            fitnessReward = -0.5;
        else
            % Worsening: Penalty proportional to degradation
            fitnessReward = -abs(fitnessImprovement) / (abs(previousFitness) + epsilon);
        end
    end

    % ===== 2. DIVERSITY REWARD =====
    % Maintain healthy diversity to avoid premature convergence
    % Diversity typically decreases over time, but too low is bad

    if length(diversityHistory) >= 2
        % Calculate diversity trend
        diversityTrend = diversityHistory(end) - diversityHistory(end-1);

        % Penalty if diversity drops too low
        if currentDiversity < config.minDiversityThreshold
            diversityReward = -1.0;  % Strong penalty for loss of diversity
        elseif diversityTrend < -epsilon
            % Diversity decreasing: small penalty (natural, but watch it)
            diversityReward = -0.2 * abs(diversityTrend);
        else
            % Diversity maintained or increasing: small bonus
            diversityReward = 0.1;
        end
    else
        diversityReward = 0.0;  % Not enough history
    end

    % ===== 3. CONVERGENCE SPEED REWARD =====
    % Bonus for making improvements quickly (early in the search)
    % Helps reward efficient exploration

    if fitnessImprovement > epsilon
        % Calculate improvement rate (improvement per iteration)
        % Larger improvements get higher rewards
        improvementMagnitude = abs(fitnessImprovement) / (previousFitness + epsilon);
        convergenceReward = improvementMagnitude * 2.0;  % Amplify for large jumps
    else
        convergenceReward = 0.0;
    end

    % ===== 4. CURIOSITY BONUS (Intrinsic Motivation) =====
    % Reward exploring novel states to encourage exploration
    % Based on state novelty detection

    if config.useCuriosityBonus && ~isempty(stateHistory) && size(stateHistory, 1) >= 2
        % Calculate novelty: distance to most recent states
        currentState = stateHistory(end, :);
        recentStates = stateHistory(max(1, end-10):end-1, :);  % Last 10 states

        % Compute minimum distance to recent states (novelty measure)
        distances = sqrt(sum((recentStates - currentState).^2, 2));
        minDistance = min(distances);

        % Normalize novelty (0 to 1)
        novelty = tanh(minDistance);  % Squash to [0, 1]

        % Curiosity bonus: reward novel states
        curiosityReward = novelty * 0.5;
    else
        curiosityReward = 0.0;
    end

    % ===== COMBINE REWARDS =====
    % Weighted sum of all components

    reward = config.rewardWeightFitness * fitnessReward + ...
             config.rewardWeightDiversity * diversityReward + ...
             config.rewardWeightConvergence * convergenceReward + ...
             config.rewardWeightCuriosity * curiosityReward;

    % Store components for logging/analysis
    components = struct();
    components.fitness = fitnessReward;
    components.diversity = diversityReward;
    components.convergence = convergenceReward;
    components.curiosity = curiosityReward;
    components.total = reward;
    components.fitnessImprovement = fitnessImprovement;
    components.currentDiversity = currentDiversity;

    % Optional: Clip reward to prevent extreme values
    reward = max(-10, min(10, reward));
end
