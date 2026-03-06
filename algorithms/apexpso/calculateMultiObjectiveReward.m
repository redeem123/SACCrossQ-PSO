function [reward, components] = calculateMultiObjectiveReward(currentFitness, previousFitness, ...
    currentDiversity, diversityHistory, stateHistory, config, rewardContext, currentState)
    % Adaptive multi-objective reward for APEXPSO.
    %
    % Reward terms:
    %   1) Relative fitness improvement (scale-aware)
    %   2) Diversity tracking to a scheduled target
    %   3) Stagnation pressure / escape bonus
    %   4) Curiosity novelty from recent state history

    if nargin < 7 || isempty(rewardContext)
        rewardContext = struct();
    end
    if nargin < 8
        currentState = [];
    end

    epsilon = 1e-8;
    INVALID_FITNESS_THRESHOLD = 1e8;

    progress = 0;
    if isfield(rewardContext, 'iter') && isfield(rewardContext, 'maxIterations') ...
            && rewardContext.maxIterations > 0
        progress = min(1, max(0, rewardContext.iter / rewardContext.maxIterations));
    end

    % ===== 1. FITNESS IMPROVEMENT =====
    isPreviousInvalid = (previousFitness >= INVALID_FITNESS_THRESHOLD) || ~isfinite(previousFitness);
    isCurrentInvalid = (currentFitness >= INVALID_FITNESS_THRESHOLD) || ~isfinite(currentFitness);

    if isPreviousInvalid && isCurrentInvalid
        relImprovement = 0;
        fitnessReward = -0.4;
    elseif isPreviousInvalid && ~isCurrentInvalid
        relImprovement = 1;
        fitnessReward = 2.5;
    elseif ~isPreviousInvalid && isCurrentInvalid
        relImprovement = -1;
        fitnessReward = -2.5;
    else
        relImprovement = (previousFitness - currentFitness) / (abs(previousFitness) + epsilon);
        improvementScale = 0.02;
        if isfield(config, 'rewardImprovementScale')
            improvementScale = max(config.rewardImprovementScale, 1e-6);
        end
        fitnessReward = tanh(relImprovement / improvementScale);
        if relImprovement <= 0
            fitnessReward = fitnessReward - 0.15;
        end
    end

    % ===== 2. DIVERSITY TRACKING =====
    diversityTargetMax = 0.30;
    diversityTargetMin = 0.03;
    diversityTargetPower = 1.2;
    if isfield(config, 'diversityTarget')
        dt = config.diversityTarget;
        if isfield(dt, 'max'), diversityTargetMax = dt.max; end
        if isfield(dt, 'min'), diversityTargetMin = dt.min; end
        if isfield(dt, 'power'), diversityTargetPower = dt.power; end
    end
    targetDiversity = diversityTargetMin + ...
        (diversityTargetMax - diversityTargetMin) * ((1 - progress) ^ diversityTargetPower);
    diversityError = (currentDiversity - targetDiversity) / (targetDiversity + epsilon);
    diversityReward = tanh(diversityError);

    % Penalize sharp diversity collapse.
    if numel(diversityHistory) >= 2
        trend = diversityHistory(end) - diversityHistory(end-1);
        if trend < -1e-6
            diversityReward = diversityReward - 0.25 * min(1, abs(trend) / (targetDiversity + epsilon));
        end
    end

    % ===== 3. STAGNATION PRESSURE =====
    stagnationIters = 0;
    if isfield(rewardContext, 'iter') && isfield(rewardContext, 'lastImprovementIteration')
        stagnationIters = max(0, rewardContext.iter - rewardContext.lastImprovementIteration);
    end
    stagnationTau = 35;
    if isfield(config, 'stagnationTimeConstant')
        stagnationTau = max(1, config.stagnationTimeConstant);
    end

    stagnationRatio = 1 - exp(-stagnationIters / stagnationTau);
    if relImprovement > 0
        stagnationReward = (1 - stagnationRatio);       % improvement after stall is rewarded
    else
        stagnationReward = -0.30 * stagnationRatio;     % prolonged stall is penalized
    end

    % ===== 4. CURIOSITY / NOVELTY =====
    curiosityReward = 0;
    if config.useCuriosityBonus && ~isempty(stateHistory) && ~isempty(currentState)
        recentStates = stateHistory(max(1, end-12):end, :);
        diffs = recentStates - repmat(currentState(:)', size(recentStates, 1), 1);
        distances = sqrt(sum(diffs.^2, 2));
        novelty = tanh(min(distances));
        curiosityReward = novelty;
    end

    % ===== COMBINE =====
    reward = config.rewardWeightFitness * fitnessReward + ...
             config.rewardWeightDiversity * diversityReward + ...
             config.rewardWeightStagnation * stagnationReward + ...
             config.rewardWeightCuriosity * curiosityReward;

    rewardClip = 10;
    if isfield(config, 'rewardClip')
        rewardClip = max(1, config.rewardClip);
    end
    reward = max(-rewardClip, min(rewardClip, reward));

    components = struct();
    components.fitness = fitnessReward;
    components.diversity = diversityReward;
    components.stagnation = stagnationReward;
    components.curiosity = curiosityReward;
    components.total = reward;
    components.relativeImprovement = relImprovement;
    components.targetDiversity = targetDiversity;
    components.currentDiversity = currentDiversity;
    components.stagnationIters = stagnationIters;
end
