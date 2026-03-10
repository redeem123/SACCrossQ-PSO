classdef APEXPSO_Agent < handle
    % RRSACPSO SAC agent with retained two-component stack:
    %   - RankResidualControl
    %   - Pluggable SAC-side critic improvement under research

    properties
        config          % RRSACPSO configuration

        % Networks
        actor           % Actor network (Gaussian policy)
        critic1         % First critic network
        critic2         % Second critic network
        critics         % Critic ensemble for REDQ-style variants
        targetCritic1   % Target critic network (optional)
        targetCritic2   % Target critic network (optional)
        targetCritics   % Target critic ensemble for REDQ-style variants

        % Optimizers (using ADAM)
        actorOptimizer
        critic1Optimizer
        critic2Optimizer
        criticOptimizers
        alphaOptimizer

        % SAC entropy temperature
        logAlpha        % Log of entropy coefficient (learned)
        targetEntropy   % Target entropy for auto-tuning

        % Replay buffer
        replayBuffer

        % Training state
        trainingStep
        explorationNoise
        useTargetNetworks
        obsNormCount
        obsNormMean
        obsNormM2
    end

    methods
        function obj = APEXPSO_Agent(config)
            % Constructor
            obj.config = config;
            obj.config = obj.normalizeRuntimeConfig(obj.config);
            config = obj.config;

            obj.critics = {};
            obj.targetCritics = {};
            obj.criticOptimizers = {};

            % Create networks
            obj.actor = createActorNetwork(config);
            if obj.getConfigValue('useREDQCritic', false)
                obj.useTargetNetworks = true;
                numCritics = max(2, round(obj.getConfigValue('numCritics', 5)));
                obj.critics = cell(1, numCritics);
                obj.criticOptimizers = cell(1, numCritics);
                for criticIdx = 1:numCritics
                    obj.critics{criticIdx} = createCriticNetwork(config);
                    obj.criticOptimizers{criticIdx} = struct('avg', [], 'avgSq', []);
                end
                obj.critic1 = obj.critics{1};
                obj.critic2 = obj.critics{2};
                if obj.useTargetNetworks
                    obj.targetCritics = cell(1, numCritics);
                    for criticIdx = 1:numCritics
                        obj.targetCritics{criticIdx} = obj.cloneCriticNetwork(obj.critics{criticIdx});
                    end
                    obj.targetCritic1 = obj.targetCritics{1};
                    obj.targetCritic2 = obj.targetCritics{2};
                end
            else
                obj.critic1 = createCriticNetwork(config);
                obj.critic2 = createCriticNetwork(config);

                % Optional target networks (standard SAC)
                obj.useTargetNetworks = isfield(config, 'useTargetNetworks') && config.useTargetNetworks;
                if obj.useTargetNetworks
                    obj.targetCritic1 = obj.cloneCriticNetwork(obj.critic1);
                    obj.targetCritic2 = obj.cloneCriticNetwork(obj.critic2);
                end
            end
            obj.applyCriticWeightNorm();
            obj.syncREDQMirrors();

            % Initialize entropy temperature
            obj.logAlpha = log(config.initAlpha);
            obj.targetEntropy = config.targetEntropy;

            % Create optimizers (Adam for networks, custom Adam for alpha scalar)
            % avg = first moment, avgSq = second moment (auto-initialized by adamupdate)
            obj.actorOptimizer   = struct('avg', [], 'avgSq', []);
            obj.critic1Optimizer = struct('avg', [], 'avgSq', []);
            obj.critic2Optimizer = struct('avg', [], 'avgSq', []);
            if obj.getConfigValue('useREDQCritic', false)
                obj.critic1Optimizer = obj.criticOptimizers{1};
                obj.critic2Optimizer = obj.criticOptimizers{2};
            end
            obj.alphaOptimizer = struct('m', 0, 'v', 0, 'beta1', 0.9, 'beta2', 0.999, 'eps', 1e-8);

            % Create replay buffer
            obj.replayBuffer = ReplayBuffer(config.bufferSize, ...
                config.stateSize, config.actionSize, config.useGPU, config);

            % Initialize training state
            obj.trainingStep = 0;
            obj.explorationNoise = config.explorationNoiseStart;
            obj.obsNormCount = 0;
            obj.obsNormMean = zeros(config.stateSize, 1, 'single');
            obj.obsNormM2 = ones(config.stateSize, 1, 'single');
        end

        function action = getAction(obj, state, training)
            % Get action from actor network
            %
            % Inputs:
            %   state: Current state [stateSize × 1]
            %   training: If true, sample stochastically
            %
            % Outputs:
            %   action: Selected action [actionSize × 1]

            if nargin < 3
                training = true;
            end

            state = obj.normalizeStateVector(single(state));
            stateDL = dlarray(state, 'CB');

            % Use predict for acting to keep single-state decisions decoupled from
            % any training-mode normalization behavior.
            output = predict(obj.actor, stateDL);
            output = extractdata(output);

            % Check for NaN/Inf in network output
            if any(isnan(output(:))) || any(isinf(output(:)))
                warning('RRSACPSO:NaNDetected', 'NaN/Inf detected in actor network output. Resetting to zeros.');
                output = zeros(size(output));
            end

            % Split into mean and log_std
            actionSize = obj.config.actionSize;
            meanAction = output(1:actionSize);
            logStd = output(actionSize+1:end);

            % Keep collection-time exploration aligned with the policy used
            % for SAC targets and actor optimization.
            logStd = clampPolicyLogStd(logStd);

            if training
                % Sample from Gaussian distribution (SAC stochastic policy)
                std = exp(logStd);
                action = meanAction + randn(size(meanAction)) .* std;
            else
                % Deterministic action for evaluation
                action = meanAction;
            end

            % Apply tanh squashing to bound actions to [-1, 1]
            action = tanh(action);

            % Final NaN check after all operations
            if any(isnan(action(:))) || any(isinf(action(:)))
                warning('RRSACPSO:NaNDetected', 'NaN/Inf detected in final action. Resetting to zeros.');
                action = zeros(size(action));
            end
        end

        function storeTransition(obj, state, action, reward, nextState, done)
            % Store transition in replay buffer
            obj.replayBuffer.add(state, action, reward, nextState, done);
        end

        function losses = train(obj)
            % Perform one SAC training step
            %
            % Returns:
            %   losses: Struct with actor_loss, critic_loss, alpha_loss

            if ~obj.replayBuffer.canSample(obj.config.batchSize)
                losses = struct('actor', 0, 'critic', 0, 'alpha', 0);
                return;
            end

            % Sample batch from replay buffer
            beta = obj.getPriorityReplayBeta();
            recentWindowSize = obj.getERERecentWindowSize();
            [states, actions, rewards, nextStates, dones, discounts, sampleIdx, sampleWeights, ...
                auxRewards, auxNextStates, auxDones, auxDiscounts] = ...
                obj.replayBuffer.sample(obj.config.batchSize, beta, recentWindowSize);

            if obj.getConfigValue('usePilarReturns', false)
                obj.updateObservationNormalizer(states, [nextStates; auxNextStates]);
            else
                obj.updateObservationNormalizer(states, nextStates);
            end
            statesNorm = obj.normalizeStateBatch(states);
            nextStatesNorm = obj.normalizeStateBatch(nextStates);
            auxNextStatesNorm = obj.normalizeStateBatch(auxNextStates);

            % Convert to dlarray
            statesDL = dlarray(statesNorm', 'CB');
            actionsDL = dlarray(actions', 'CB');
            nextStatesDL = dlarray(nextStatesNorm', 'CB');
            donesDL = dlarray(dones', 'CB');
            rewardsDL = dlarray(rewards', 'CB');
            discountsDL = dlarray(discounts', 'CB');
            sampleWeightsDL = dlarray(sampleWeights', 'CB');
            auxNextStatesDL = dlarray(auxNextStatesNorm', 'CB');
            auxDonesDL = dlarray(auxDones', 'CB');
            auxRewardsDL = dlarray(auxRewards', 'CB');
            auxDiscountsDL = dlarray(auxDiscounts', 'CB');

            % Get current alpha
            alpha = exp(obj.logAlpha);

            % ===== UPDATE CRITICS =====
            stateAction = cat(1, statesDL, actionsDL);
            useJointBatch = obj.useJointCriticBatch();

            if obj.getConfigValue('useREDQCritic', false)
                losses = obj.trainREDQUpdate(stateAction, statesDL, nextStatesDL, rewardsDL, donesDL, ...
                    discountsDL, sampleIdx, sampleWeightsDL, auxRewardsDL, auxNextStatesDL, ...
                    auxDonesDL, auxDiscountsDL, alpha);

                % Increment training step
                obj.trainingStep = obj.trainingStep + 1;

                % Decay exploration noise
                obj.explorationNoise = max(obj.config.explorationNoiseEnd, ...
                    obj.explorationNoise * obj.config.explorationDecay);
                return;
            end

            if obj.getConfigValue('useTQCCritic', false)
                targetQuantiles = obj.buildTQCTargetQuantiles( ...
                    stateAction, nextStatesDL, rewardsDL, donesDL, discountsDL, alpha);
                if obj.getConfigValue('usePilarReturns', false)
                    auxTargetQuantiles = obj.buildTQCTargetQuantiles( ...
                        stateAction, auxNextStatesDL, auxRewardsDL, auxDonesDL, auxDiscountsDL, alpha);
                    mixCoeff = obj.getConfigValue('pilarMixCoefficient', 0.406);
                    targetQuantiles = (1 - mixCoeff) .* targetQuantiles + mixCoeff .* auxTargetQuantiles;
                    targetQuantiles = dlarray(extractdata(targetQuantiles), 'CB');
                end
                targetQMean = mean(targetQuantiles, 1);
                tauHat = obj.getTQCTauHat();
                huberKappa = obj.getConfigValue('tqcHuberKappa', 1.0);

                [loss1, critic1Grads, critic1State, q1Pred] = dlfeval( ...
                    @tqcCriticModelLoss, obj.critic1, stateAction, targetQuantiles, tauHat, huberKappa, sampleWeightsDL);
                critic1Grads = obj.clipGradients(critic1Grads, 1.0);
                [obj.critic1.Learnables, obj.critic1Optimizer.avg, obj.critic1Optimizer.avgSq] = ...
                    adamupdate(obj.critic1.Learnables, critic1Grads, ...
                    obj.critic1Optimizer.avg, obj.critic1Optimizer.avgSq, ...
                    obj.trainingStep + 1, obj.config.criticLR, ...
                    obj.getConfigValue('criticAdamBeta1', 0.9), ...
                    obj.getConfigValue('criticAdamBeta2', 0.999));
                obj.critic1.State = critic1State;
                obj.critic1 = obj.projectCriticWeights(obj.critic1);

                [loss2, critic2Grads, critic2State, q2Pred] = dlfeval( ...
                    @tqcCriticModelLoss, obj.critic2, stateAction, targetQuantiles, tauHat, huberKappa, sampleWeightsDL);
                critic2Grads = obj.clipGradients(critic2Grads, 1.0);
                [obj.critic2.Learnables, obj.critic2Optimizer.avg, obj.critic2Optimizer.avgSq] = ...
                    adamupdate(obj.critic2.Learnables, critic2Grads, ...
                    obj.critic2Optimizer.avg, obj.critic2Optimizer.avgSq, ...
                    obj.trainingStep + 1, obj.config.criticLR, ...
                    obj.getConfigValue('criticAdamBeta1', 0.9), ...
                    obj.getConfigValue('criticAdamBeta2', 0.999));
                obj.critic2.State = critic2State;
                obj.critic2 = obj.projectCriticWeights(obj.critic2);

                tdErr = 0.5 .* (abs(extractdata(q1Pred - targetQMean)) + abs(extractdata(q2Pred - targetQMean)));
            else
                [nextActions, nextLogProbs] = obj.sampleAction(nextStatesDL);
                nextStateAction = cat(1, nextStatesDL, nextActions);

                if obj.useTargetNetworks
                    updateTargetState = obj.getConfigValue('targetCriticTrainMode', false);
                    [~, q1Next, targetState1] = obj.forwardCriticTargets( ...
                        obj.targetCritic1, stateAction, nextStateAction, updateTargetState);
                    [~, q2Next, targetState2] = obj.forwardCriticTargets( ...
                        obj.targetCritic2, stateAction, nextStateAction, updateTargetState);
                    if updateTargetState
                        obj.targetCritic1.State = targetState1;
                        obj.targetCritic2.State = targetState2;
                    end
                else
                    [~, q1Next, criticState1] = obj.forwardCriticTargets( ...
                        obj.critic1, stateAction, nextStateAction, useJointBatch);
                    [~, q2Next, criticState2] = obj.forwardCriticTargets( ...
                        obj.critic2, stateAction, nextStateAction, useJointBatch);
                    if useJointBatch
                        obj.critic1.State = criticState1;
                        obj.critic2.State = criticState2;
                    end
                end
                qNext = obj.aggregateTargetCritics(q1Next, q2Next);

                targetQ = rewardsDL + discountsDL .* (1 - donesDL) .* ...
                         (qNext - alpha .* nextLogProbs);
                if obj.getConfigValue('usePilarReturns', false)
                    [auxActions, auxLogProbs] = obj.sampleAction(auxNextStatesDL);
                    auxStateAction = cat(1, statesDL, actionsDL);
                    auxNextStateAction = cat(1, auxNextStatesDL, auxActions);
                    if obj.useTargetNetworks
                        updateTargetState = obj.getConfigValue('targetCriticTrainMode', false);
                        [~, q1AuxNext, targetState1] = obj.forwardCriticTargets( ...
                            obj.targetCritic1, auxStateAction, auxNextStateAction, updateTargetState);
                        [~, q2AuxNext, targetState2] = obj.forwardCriticTargets( ...
                            obj.targetCritic2, auxStateAction, auxNextStateAction, updateTargetState);
                        if updateTargetState
                            obj.targetCritic1.State = targetState1;
                            obj.targetCritic2.State = targetState2;
                        end
                    else
                        [~, q1AuxNext, criticState1] = obj.forwardCriticTargets( ...
                            obj.critic1, auxStateAction, auxNextStateAction, useJointBatch);
                        [~, q2AuxNext, criticState2] = obj.forwardCriticTargets( ...
                            obj.critic2, auxStateAction, auxNextStateAction, useJointBatch);
                        if useJointBatch
                            obj.critic1.State = criticState1;
                            obj.critic2.State = criticState2;
                        end
                    end
                    auxQNext = obj.aggregateTargetCritics(q1AuxNext, q2AuxNext);
                    auxTargetQ = auxRewardsDL + auxDiscountsDL .* (1 - auxDonesDL) .* ...
                        (auxQNext - alpha .* auxLogProbs);
                    mixCoeff = obj.getConfigValue('pilarMixCoefficient', 0.406);
                    targetQ = (1 - mixCoeff) .* targetQ + mixCoeff .* auxTargetQ;
                end
                targetQ = dlarray(extractdata(targetQ), 'CB');

                [loss1, critic1Grads, critic1State, q1Pred] = dlfeval( ...
                    @criticModelLoss, obj.critic1, stateAction, nextStateAction, targetQ, useJointBatch, sampleWeightsDL);
                critic1Grads = obj.clipGradients(critic1Grads, 1.0);
                [obj.critic1.Learnables, obj.critic1Optimizer.avg, obj.critic1Optimizer.avgSq] = ...
                    adamupdate(obj.critic1.Learnables, critic1Grads, ...
                    obj.critic1Optimizer.avg, obj.critic1Optimizer.avgSq, ...
                    obj.trainingStep + 1, obj.config.criticLR, ...
                    obj.getConfigValue('criticAdamBeta1', 0.9), ...
                    obj.getConfigValue('criticAdamBeta2', 0.999));
                obj.critic1.State = critic1State;
                obj.critic1 = obj.projectCriticWeights(obj.critic1);

                [loss2, critic2Grads, critic2State, q2Pred] = dlfeval( ...
                    @criticModelLoss, obj.critic2, stateAction, nextStateAction, targetQ, useJointBatch, sampleWeightsDL);
                critic2Grads = obj.clipGradients(critic2Grads, 1.0);
                [obj.critic2.Learnables, obj.critic2Optimizer.avg, obj.critic2Optimizer.avgSq] = ...
                    adamupdate(obj.critic2.Learnables, critic2Grads, ...
                    obj.critic2Optimizer.avg, obj.critic2Optimizer.avgSq, ...
                    obj.trainingStep + 1, obj.config.criticLR, ...
                    obj.getConfigValue('criticAdamBeta1', 0.9), ...
                    obj.getConfigValue('criticAdamBeta2', 0.999));
                obj.critic2.State = critic2State;
                obj.critic2 = obj.projectCriticWeights(obj.critic2);

                tdErr = obj.aggregateTDError(q1Pred, q2Pred, targetQ);
            end
            meanTdErr = mean(tdErr(:));
            obj.replayBuffer.updatePriorities(sampleIdx, tdErr(:));

            criticLoss = extractdata(loss1) + extractdata(loss2);
            if obj.useTargetNetworks
                obj.softUpdateTargetNetworks();
            end

            shouldUpdateActor = obj.shouldUpdateActorNow();
            actorLoss = dlarray(0, 'CB');
            alphaLoss = dlarray(0, 'CB');
            targetEntropyNow = obj.targetEntropy;
            avgLogProb = NaN;

            if shouldUpdateActor
                % ===== UPDATE ACTOR =====
                [actorLoss, actorGrads, actorState] = dlfeval(@actorModelLoss, obj.actor, statesDL, ...
                    obj.critic1, obj.critic2, alpha, obj.getConfigValue('useAQECritic', false), ...
                    obj.getConfigValue('useTQCCritic', false));
                actorGrads = obj.clipGradients(actorGrads, 1.0);
                [obj.actor.Learnables, obj.actorOptimizer.avg, obj.actorOptimizer.avgSq] = ...
                    adamupdate(obj.actor.Learnables, actorGrads, ...
                    obj.actorOptimizer.avg, obj.actorOptimizer.avgSq, ...
                    obj.trainingStep + 1, obj.config.actorLR, ...
                    obj.getConfigValue('actorAdamBeta1', 0.9), ...
                    obj.getConfigValue('actorAdamBeta2', 0.999));
                obj.actor.State = actorState;

                % ===== UPDATE ALPHA (Automatic Entropy Tuning) =====
                if isfield(obj.config, 'entropyAnnealStrength') && isfield(obj.config, 'entropyAnnealSteps')
                    annealProgress = min(1, obj.trainingStep / max(1, obj.config.entropyAnnealSteps));
                    annealScale = 1 - obj.config.entropyAnnealStrength * annealProgress;
                    targetEntropyNow = obj.targetEntropy * annealScale;
                end
                [alphaLossCandidate, avgLogProbValue] = obj.alphaLossFunc(statesDL, targetEntropyNow);
                avgLogProb = extractdata(avgLogProbValue);
                if obj.getConfigValue('useAutomaticEntropyTuning', true)
                    alphaLoss = alphaLossCandidate;
                    alphaGrad = -avgLogProb - targetEntropyNow;

                    if isnan(alphaGrad) || isinf(alphaGrad)
                        alphaGrad = 0;
                    else
                        alphaGrad = max(min(alphaGrad, 10), -10);
                    end

                    [obj.logAlpha, obj.alphaOptimizer] = obj.adamUpdateScalar(obj.logAlpha, ...
                        alphaGrad, obj.alphaOptimizer, obj.config.alphaLR, obj.trainingStep);

                    if isnan(obj.logAlpha) || isinf(obj.logAlpha)
                        obj.logAlpha = log(0.2);
                    else
                        obj.logAlpha = max(min(obj.logAlpha, 1), -10);
                    end
                end
            end

            % Increment training step
            obj.trainingStep = obj.trainingStep + 1;

            % Decay exploration noise
            obj.explorationNoise = max(obj.config.explorationNoiseEnd, ...
                obj.explorationNoise * obj.config.explorationDecay);

            % Return losses
            losses = struct();
            losses.actor = extractdata(actorLoss);
            losses.critic = criticLoss;  % Already extracted
            losses.alpha = extractdata(alphaLoss);
            losses.alphaValue = exp(obj.logAlpha);
            losses.targetEntropy = targetEntropyNow;
            losses.meanTdError = meanTdErr;
            losses.actorLoss = losses.actor;
            losses.critic1Loss = extractdata(loss1);
            losses.critic2Loss = extractdata(loss2);
            losses.qValue = mean(extractdata(obj.aggregateActorCritics(q1Pred, q2Pred)), 'all');
            losses.entropy = -avgLogProb;
        end

        function [actions, logProbs] = sampleAction(obj, states)
            % Sample action from Gaussian policy
            %
            % Inputs:
            %   states: Batch of states [stateSize × batchSize]
            %
            % Outputs:
            %   actions: Sampled actions [actionSize × batchSize]
            %   logProbs: Log probabilities [1 × batchSize]

            % Get output from actor (mean and log_std concatenated)
            output = predict(obj.actor, states);

            % Check for NaN/Inf in network output
            if any(isnan(output(:))) || any(isinf(output(:)))
                warning('RRSACPSO:NaNDetected', 'NaN/Inf detected in sampleAction forward pass. Resetting to zeros.');
                output = zeros(size(output), 'like', output);
            end

            % Split into mean and log_std
            actionSize = obj.config.actionSize;
            meanActions = output(1:actionSize, :);
            logStd = output(actionSize+1:end, :);

            % Match the same bounded policy parameterization used at
            % collection time so replay and target sampling stay aligned.
            logStd = clampPolicyLogStd(logStd);
            std = exp(logStd);

            % Sample from Gaussian
            epsilon = randn(size(meanActions), 'like', meanActions);
            unsquashedActions = meanActions + std .* epsilon;

            % Apply tanh squashing
            actions = tanh(unsquashedActions);

            % Compute log probability with tanh correction
            % log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u))
            gaussianLogProb = -0.5 * sum(epsilon.^2 + 2 * logStd + log(2 * pi), 1);

            % Robust tanh correction with clipping to prevent NaN
            actionSquared = actions.^2;
            actionSquared = min(actionSquared, 0.9999);  % Prevent 1 - x^2 from being too small
            tanhCorrection = sum(log(1 - actionSquared + 1e-6), 1);

            logProbs = gaussianLogProb - tanhCorrection;

            % Final safety: Clip log probs to reasonable range
            logProbs = max(min(logProbs, 100), -100);
        end

        function config = normalizeRuntimeConfig(~, config)
            % Research overrides are applied after APEXPSO_Config runs, so
            % sanitize the effective runtime config here as well.
            if ~isfield(config, 'useREDQCritic')
                config.useREDQCritic = false;
            end
            if ~isfield(config, 'useCrossQCritic')
                config.useCrossQCritic = false;
            end
            if ~isfield(config, 'redqNumCritics')
                config.redqNumCritics = getFieldOrDefault(config, 'numCritics', 5);
            end
            if ~isfield(config, 'redqTargetSubsetSize')
                config.redqTargetSubsetSize = 2;
            end
            if ~isfield(config, 'redqTargetMode')
                config.redqTargetMode = 'min';
            end
            if ~isfield(config, 'redqPolicyUpdateDelay')
                config.redqPolicyUpdateDelay = getFieldOrDefault(config, 'actorUpdateInterval', 5);
            end
            if ~isfield(config, 'redqWarmupSteps')
                config.redqWarmupSteps = getFieldOrDefault(config, 'warmupPeriod', 128);
            end

            if config.useREDQCritic
                config.useCrossQCritic = false;
                config.useAQECritic = false;
                config.useDroQCritic = false;
                config.useTQCCritic = false;
                config.useCriticBatchNorm = false;
                config.useActorBatchNorm = false;
                config.useJointCriticBatchForBN = false;
                config.useWeightNormCritic = false;
                config.useTargetNetworks = true;
                config.redqNumCritics = max(3, round(max( ...
                    getFieldOrDefault(config, 'redqNumCritics', 5), ...
                    getFieldOrDefault(config, 'numCritics', 5))));
                config.numCritics = config.redqNumCritics;
                config.redqTargetSubsetSize = max(2, min(config.numCritics, ...
                    round(getFieldOrDefault(config, 'redqTargetSubsetSize', 2))));
                config.redqPolicyUpdateDelay = max(1, round(getFieldOrDefault(config, ...
                    'redqPolicyUpdateDelay', getFieldOrDefault(config, 'actorUpdateInterval', 5))));
                config.redqWarmupSteps = max(0, round(getFieldOrDefault(config, ...
                    'redqWarmupSteps', getFieldOrDefault(config, 'warmupPeriod', 128))));
                config.utdRatio = max(1, round(getFieldOrDefault(config, 'utdRatio', 5)));
                config.gradientStepsPerTraining = max(1, round( ...
                    getFieldOrDefault(config, 'gradientStepsPerTraining', config.utdRatio)));
                config.useDelayedPolicyUpdates = config.gradientStepsPerTraining > 1;
                if config.useDelayedPolicyUpdates
                    config.actorUpdateInterval = max(1, round(getFieldOrDefault(config, ...
                        'actorUpdateInterval', config.gradientStepsPerTraining)));
                else
                    config.actorUpdateInterval = 1;
                end
                config.warmupPeriod = max(round(getFieldOrDefault(config, 'warmupPeriod', 0)), ...
                    config.redqWarmupSteps);
            elseif config.useCrossQCritic
                config.useREDQCritic = false;
                config.useAQECritic = false;
                config.useDroQCritic = false;
                config.useTQCCritic = false;
                config.useCriticBatchNorm = true;
                config.useJointCriticBatchForBN = true;
                config.useTargetNetworks = false;
                config.utdRatio = 1;
                config.gradientStepsPerTraining = 1;
                config.useDelayedPolicyUpdates = false;
                config.actorUpdateInterval = 1;
                config.numCritics = 2;
            elseif getFieldOrDefault(config, 'useTQCCritic', false)
                config.useCrossQCritic = false;
                config.useREDQCritic = false;
                config.useAQECritic = false;
                config.useDroQCritic = false;
                config.useCriticBatchNorm = false;
                config.useActorBatchNorm = false;
                config.useJointCriticBatchForBN = false;
                config.useWeightNormCritic = false;
                config.useTargetNetworks = true;
                config.useDelayedPolicyUpdates = false;
                config.actorUpdateInterval = 1;
                config.numCritics = 2;
                config.utdRatio = 1;
                config.gradientStepsPerTraining = 1;
                config.tqcNumQuantiles = max(5, round(getFieldOrDefault(config, 'tqcNumQuantiles', 25)));
                config.tqcDropQuantilesPerCritic = max(0, min(config.tqcNumQuantiles - 1, ...
                    round(getFieldOrDefault(config, 'tqcDropQuantilesPerCritic', 2))));
                config.tqcHuberKappa = max(eps, getFieldOrDefault(config, 'tqcHuberKappa', 1.0));
            elseif getFieldOrDefault(config, 'useAQECritic', false)
                totalAqeHeads = max(1, round(getFieldOrDefault(config, 'numCritics', 2))) * ...
                    max(1, round(getFieldOrDefault(config, 'aqeHeadsPerCritic', 3)));
                config.aqeKeepHeads = max(1, min(totalAqeHeads, ...
                    round(getFieldOrDefault(config, 'aqeKeepHeads', 2))));
            else
                config.numCritics = max(2, round(getFieldOrDefault(config, 'numCritics', 2)));
                config.gradientStepsPerTraining = max(1, round( ...
                    getFieldOrDefault(config, 'gradientStepsPerTraining', ...
                    getFieldOrDefault(config, 'utdRatio', 1))));
                config.actorUpdateInterval = max(1, round(getFieldOrDefault(config, 'actorUpdateInterval', 1)));
            end
        end

        function syncREDQMirrors(obj)
            if ~obj.getConfigValue('useREDQCritic', false) || isempty(obj.critics)
                return;
            end

            obj.critic1 = obj.critics{1};
            obj.critic2 = obj.critics{2};
            obj.critic1Optimizer = obj.criticOptimizers{1};
            obj.critic2Optimizer = obj.criticOptimizers{2};

            if obj.useTargetNetworks && ~isempty(obj.targetCritics)
                obj.targetCritic1 = obj.targetCritics{1};
                obj.targetCritic2 = obj.targetCritics{2};
            end
        end

        function shouldUpdate = shouldUpdateActorNow(obj)
            if ~obj.getConfigValue('useDelayedPolicyUpdates', false)
                shouldUpdate = true;
                return;
            end

            interval = max(1, round(obj.getConfigValue('actorUpdateInterval', 2)));
            shouldUpdate = mod(obj.trainingStep + 1, interval) == 0;
        end

        function criticSet = getCriticEnsemble(obj)
            if obj.getConfigValue('useREDQCritic', false)
                criticSet = obj.critics;
            else
                criticSet = {obj.critic1, obj.critic2};
            end
        end

        function criticSet = getTargetCriticEnsemble(obj)
            if obj.getConfigValue('useREDQCritic', false)
                criticSet = obj.targetCritics;
            else
                criticSet = {obj.targetCritic1, obj.targetCritic2};
            end
        end

        function [qValues, criticSet] = forwardCriticEnsembleTargets(obj, criticSet, currentInput, nextInput, updateState)
            numCritics = numel(criticSet);
            qCells = cell(1, numCritics);
            for criticIdx = 1:numCritics
                [~, qCells{criticIdx}, newState] = obj.forwardCriticTargets( ...
                    criticSet{criticIdx}, currentInput, nextInput, updateState);
                if updateState
                    criticSet{criticIdx}.State = newState;
                end
            end
            qValues = cat(1, qCells{:});
        end

        function aggregatedQ = aggregateTargetEnsemble(obj, qValues)
            targetMode = lower(string(obj.getConfigValue('redqTargetMode', 'min')));
            if targetMode == "ave"
                aggregatedQ = mean(qValues, 1);
                return;
            end
            subsetSize = min(size(qValues, 1), max(2, round(obj.getConfigValue('redqTargetSubsetSize', 2))));
            subsetIdx = randperm(size(qValues, 1), subsetSize);
            aggregatedQ = min(qValues(subsetIdx, :), [], 1);
        end

        function aggregatedQ = aggregateActorEnsemble(~, qValues)
            aggregatedQ = mean(qValues, 1);
        end

        function tdErr = aggregateEnsembleTDError(obj, qPreds, targetQ)
            targetData = extractdata(targetQ);
            tdErr = zeros(1, size(targetData, 2), 'single');
            numCritics = numel(qPreds);
            for criticIdx = 1:numCritics
                reducedPred = obj.reduceCriticPrediction(qPreds{criticIdx});
                tdErr = tdErr + abs(extractdata(reducedPred) - targetData);
            end
            tdErr = tdErr ./ max(1, numCritics);
        end

        function netCopy = cloneCriticNetwork(obj, net)
            % Clone a critic network while preserving graph connectivity.
            netCopy = createCriticNetwork(obj.config);
            netCopy.Learnables = net.Learnables;
            netCopy.State = net.State;
        end

        function softUpdateTargetNetworks(obj)
            % Soft update target critics (standard SAC).
            tau = obj.config.tau;
            if obj.getConfigValue('useREDQCritic', false)
                for criticIdx = 1:numel(obj.critics)
                    obj.targetCritics{criticIdx} = obj.softUpdateNetwork( ...
                        obj.targetCritics{criticIdx}, obj.critics{criticIdx}, tau);
                    obj.targetCritics{criticIdx} = obj.projectCriticWeights(obj.targetCritics{criticIdx});
                end
                obj.syncREDQMirrors();
            else
                obj.targetCritic1 = obj.softUpdateNetwork(obj.targetCritic1, obj.critic1, tau);
                obj.targetCritic2 = obj.softUpdateNetwork(obj.targetCritic2, obj.critic2, tau);
                obj.targetCritic1 = obj.projectCriticWeights(obj.targetCritic1);
                obj.targetCritic2 = obj.projectCriticWeights(obj.targetCritic2);
            end
        end

        function targetNet = softUpdateNetwork(~, targetNet, sourceNet, tau)
            targetLearnables = targetNet.Learnables;
            sourceLearnables = sourceNet.Learnables;
            for i = 1:height(targetLearnables)
                targetLearnables.Value{i} = tau * sourceLearnables.Value{i} + ...
                    (1 - tau) * targetLearnables.Value{i};
            end
            targetNet.Learnables = targetLearnables;
            targetNet.State = sourceNet.State;
        end

        function [loss, loss1, loss2] = criticLossFunc(obj, states, actions, targetQ)
            % Critic loss function - returns losses only
            stateAction = cat(1, states, actions);

            q1 = forward(obj.critic1, stateAction);
            q2 = forward(obj.critic2, stateAction);

            loss1 = mean((q1 - targetQ).^2);
            loss2 = mean((q2 - targetQ).^2);
            loss = loss1 + loss2;
        end

        function loss = critic1LossFunc(obj, states, actions, targetQ)
            % Critic 1 loss only
            stateAction = cat(1, states, actions);
            q1 = forward(obj.critic1, stateAction);
            loss = mean((q1 - targetQ).^2);
        end

        function loss = critic2LossFunc(obj, states, actions, targetQ)
            % Critic 2 loss only
            stateAction = cat(1, states, actions);
            q2 = forward(obj.critic2, stateAction);
            loss = mean((q2 - targetQ).^2);
        end

        function loss = actorLossFunc(obj, states, alpha)
            % Actor loss function (maximize Q - α*log_prob)
            [actions, logProbs] = obj.sampleAction(states);

            stateAction = cat(1, states, actions);
            q1 = forward(obj.critic1, stateAction);
            q2 = forward(obj.critic2, stateAction);
            q = obj.aggregateActorCritics(q1, q2);

            % SAC objective: maximize Q - α*log_prob
            % Loss: minimize -(Q - α*log_prob)
            loss = mean(alpha .* logProbs - q);
        end

        function targetQ = buildREDQTargetQ(obj, stateAction, nextStates, rewards, dones, discounts, alpha)
            [nextActions, nextLogProbs] = obj.sampleAction(nextStates);
            nextStateAction = cat(1, nextStates, nextActions);

            if obj.useTargetNetworks
                updateTargetState = obj.getConfigValue('targetCriticTrainMode', false);
                targetCriticSet = obj.getTargetCriticEnsemble();
                [qNextAll, targetCriticSet] = obj.forwardCriticEnsembleTargets( ...
                    targetCriticSet, stateAction, nextStateAction, updateTargetState);
                if updateTargetState
                    obj.targetCritics = targetCriticSet;
                    obj.syncREDQMirrors();
                end
            else
                criticSet = obj.getCriticEnsemble();
                [qNextAll, criticSet] = obj.forwardCriticEnsembleTargets( ...
                    criticSet, stateAction, nextStateAction, false);
                if obj.getConfigValue('useREDQCritic', false)
                    obj.critics = criticSet;
                    obj.syncREDQMirrors();
                end
            end

            qNext = obj.aggregateTargetEnsemble(qNextAll);
            targetQ = rewards + discounts .* (1 - dones) .* (qNext - alpha .* nextLogProbs);
            targetQ = dlarray(extractdata(targetQ), 'CB');
        end

        function losses = trainREDQUpdate(obj, stateAction, statesDL, nextStatesDL, rewardsDL, donesDL, ...
                discountsDL, sampleIdx, sampleWeightsDL, auxRewardsDL, auxNextStatesDL, ...
                auxDonesDL, auxDiscountsDL, alpha)
            targetQ = obj.buildREDQTargetQ(stateAction, nextStatesDL, rewardsDL, donesDL, discountsDL, alpha);
            if obj.getConfigValue('usePilarReturns', false)
                auxTargetQ = obj.buildREDQTargetQ(stateAction, auxNextStatesDL, ...
                    auxRewardsDL, auxDonesDL, auxDiscountsDL, alpha);
                mixCoeff = obj.getConfigValue('pilarMixCoefficient', 0.406);
                targetQ = (1 - mixCoeff) .* targetQ + mixCoeff .* auxTargetQ;
                targetQ = dlarray(extractdata(targetQ), 'CB');
            end

            numCritics = numel(obj.critics);
            criticLossValues = zeros(1, numCritics);
            qPreds = cell(1, numCritics);
            for criticIdx = 1:numCritics
                [loss, criticGrads, criticState, qPred] = dlfeval( ...
                    @criticModelLoss, obj.critics{criticIdx}, stateAction, [], targetQ, false, sampleWeightsDL);
                criticGrads = obj.clipGradients(criticGrads, 1.0);
                [obj.critics{criticIdx}.Learnables, obj.criticOptimizers{criticIdx}.avg, ...
                    obj.criticOptimizers{criticIdx}.avgSq] = ...
                    adamupdate(obj.critics{criticIdx}.Learnables, criticGrads, ...
                    obj.criticOptimizers{criticIdx}.avg, obj.criticOptimizers{criticIdx}.avgSq, ...
                    obj.trainingStep + 1, obj.config.criticLR, ...
                    obj.getConfigValue('criticAdamBeta1', 0.9), ...
                    obj.getConfigValue('criticAdamBeta2', 0.999));
                obj.critics{criticIdx}.State = criticState;
                obj.critics{criticIdx} = obj.projectCriticWeights(obj.critics{criticIdx});
                criticLossValues(criticIdx) = extractdata(loss);
                qPreds{criticIdx} = qPred;
            end
            obj.syncREDQMirrors();

            tdErr = obj.aggregateEnsembleTDError(qPreds, targetQ);
            meanTdErr = mean(tdErr(:));
            obj.replayBuffer.updatePriorities(sampleIdx, tdErr(:));

            if obj.useTargetNetworks
                obj.softUpdateTargetNetworks();
            end

            actorLoss = dlarray(0, 'CB');
            alphaLoss = dlarray(0, 'CB');
            targetEntropyNow = obj.targetEntropy;
            avgLogProb = NaN;
            if obj.shouldUpdateActorNow()
                [actorLoss, actorGrads, actorState] = dlfeval(@redqActorModelLoss, ...
                    obj.actor, statesDL, obj.critics, alpha);
                actorGrads = obj.clipGradients(actorGrads, 1.0);
                [obj.actor.Learnables, obj.actorOptimizer.avg, obj.actorOptimizer.avgSq] = ...
                    adamupdate(obj.actor.Learnables, actorGrads, ...
                    obj.actorOptimizer.avg, obj.actorOptimizer.avgSq, ...
                    obj.trainingStep + 1, obj.config.actorLR, ...
                    obj.getConfigValue('actorAdamBeta1', 0.9), ...
                    obj.getConfigValue('actorAdamBeta2', 0.999));
                obj.actor.State = actorState;

                if isfield(obj.config, 'entropyAnnealStrength') && isfield(obj.config, 'entropyAnnealSteps')
                    annealProgress = min(1, obj.trainingStep / max(1, obj.config.entropyAnnealSteps));
                    annealScale = 1 - obj.config.entropyAnnealStrength * annealProgress;
                    targetEntropyNow = obj.targetEntropy * annealScale;
                end
                [alphaLossCandidate, avgLogProbValue] = obj.alphaLossFunc(statesDL, targetEntropyNow);
                avgLogProb = extractdata(avgLogProbValue);
                if obj.getConfigValue('useAutomaticEntropyTuning', true)
                    alphaLoss = alphaLossCandidate;
                    alphaGrad = -avgLogProb - targetEntropyNow;
                    if isnan(alphaGrad) || isinf(alphaGrad)
                        alphaGrad = 0;
                    else
                        alphaGrad = max(min(alphaGrad, 10), -10);
                    end
                    [obj.logAlpha, obj.alphaOptimizer] = obj.adamUpdateScalar(obj.logAlpha, ...
                        alphaGrad, obj.alphaOptimizer, obj.config.alphaLR, obj.trainingStep);
                    if isnan(obj.logAlpha) || isinf(obj.logAlpha)
                        obj.logAlpha = log(0.2);
                    else
                        obj.logAlpha = max(min(obj.logAlpha, 1), -10);
                    end
                end
            end

            losses = struct();
            losses.actor = extractdata(actorLoss);
            losses.critic = sum(criticLossValues);
            losses.alpha = extractdata(alphaLoss);
            losses.alphaValue = exp(obj.logAlpha);
            losses.targetEntropy = targetEntropyNow;
            losses.meanTdError = meanTdErr;
            losses.actorLoss = losses.actor;
            losses.critic1Loss = criticLossValues(1);
            losses.critic2Loss = criticLossValues(min(2, numCritics));
            qValueMeans = zeros(1, numCritics);
            for criticIdx = 1:numCritics
                qValueMeans(criticIdx) = mean(extractdata(obj.reduceCriticPrediction(qPreds{criticIdx})), 'all');
            end
            losses.qValue = mean(qValueMeans);
            losses.entropy = -avgLogProb;
        end

        function [loss, avgLogProb] = alphaLossFunc(obj, states, targetEntropyNow)
            % Alpha loss function (automatic entropy tuning)
            if nargin < 3
                targetEntropyNow = obj.targetEntropy;
            end
            [~, logProbs] = obj.sampleAction(states);

            % Alpha objective: α * (H_target + log_prob)
            alpha = exp(obj.logAlpha);
            loss = mean(alpha .* (-logProbs - targetEntropyNow));

            % Return average log prob for manual gradient computation
            avgLogProb = mean(logProbs);
        end

        function enabled = useJointCriticBatch(obj)
            criticBN = obj.getConfigValue('useCriticBatchNorm', obj.getConfigValue('useBatchNorm', false));
            enabled = criticBN && obj.getConfigValue('useJointCriticBatchForBN', false);
        end

        function aggregatedQ = aggregateTargetCritics(obj, q1, q2)
            if obj.getConfigValue('useAQECritic', false)
                combinedQ = cat(1, q1, q2);
                keepHeads = min(size(combinedQ, 1), obj.getConfigValue('aqeKeepHeads', 2));
                combinedQ = sort(combinedQ, 1, 'ascend');
                aggregatedQ = mean(combinedQ(1:keepHeads, :), 1);
            else
                aggregatedQ = min(q1, q2);
            end
        end

        function aggregatedQ = aggregateActorCritics(obj, q1, q2)
            if obj.getConfigValue('useAQECritic', false)
                aggregatedQ = mean(cat(1, q1, q2), 1);
            elseif obj.getConfigValue('useTQCCritic', false)
                aggregatedQ = mean(cat(1, q1, q2), 1);
            else
                if size(q1, 1) > 1
                    q1 = mean(q1, 1);
                end
                if size(q2, 1) > 1
                    q2 = mean(q2, 1);
                end
                aggregatedQ = min(q1, q2);
            end
        end

        function tdErr = aggregateTDError(obj, q1Pred, q2Pred, targetQ)
            q1Pred = obj.reduceCriticPrediction(q1Pred);
            q2Pred = obj.reduceCriticPrediction(q2Pred);
            targetData = dlarray(extractdata(targetQ), 'CB');
            tdErr = 0.5 .* (abs(extractdata(q1Pred - targetData)) + abs(extractdata(q2Pred - targetData)));
        end

        function reducedQ = reduceCriticPrediction(obj, qValues)
            if obj.getConfigValue('useAQECritic', false) && size(qValues, 1) > 1
                reducedQ = mean(qValues, 1);
            elseif size(qValues, 1) > 1
                reducedQ = mean(qValues, 1);
            else
                reducedQ = qValues;
            end
        end

        function [currentQ, nextQ, newState] = forwardCriticTargets(obj, net, currentInput, nextInput, updateState)
            currentQ = [];
            newState = net.State;

            if obj.useJointCriticBatch()
                mixedInput = cat(2, currentInput, nextInput);
                if updateState
                    [mixedQ, newState] = forward(net, mixedInput);
                else
                    mixedQ = forward(net, mixedInput);
                end
                currentBatch = size(currentInput, 2);
                currentQ = mixedQ(:, 1:currentBatch);
                nextQ = mixedQ(:, currentBatch+1:end);
            else
                if updateState
                    [nextQ, newState] = forward(net, nextInput);
                else
                    nextQ = forward(net, nextInput);
                end
            end

            if ~isempty(currentQ) && (any(isnan(currentQ(:))) || any(isinf(currentQ(:))))
                currentQ = zeros(size(currentQ), 'like', currentQ);
            end
            if any(isnan(nextQ(:))) || any(isinf(nextQ(:)))
                nextQ = zeros(size(nextQ), 'like', nextQ);
            end
        end

        function applyCriticWeightNorm(obj)
            if obj.getConfigValue('useREDQCritic', false)
                for criticIdx = 1:numel(obj.critics)
                    obj.critics{criticIdx} = obj.projectCriticWeights(obj.critics{criticIdx});
                end
                if obj.useTargetNetworks
                    for criticIdx = 1:numel(obj.targetCritics)
                        obj.targetCritics{criticIdx} = obj.projectCriticWeights(obj.targetCritics{criticIdx});
                    end
                end
                obj.syncREDQMirrors();
                return;
            end

            obj.critic1 = obj.projectCriticWeights(obj.critic1);
            obj.critic2 = obj.projectCriticWeights(obj.critic2);
            if obj.useTargetNetworks
                obj.targetCritic1 = obj.projectCriticWeights(obj.targetCritic1);
                obj.targetCritic2 = obj.projectCriticWeights(obj.targetCritic2);
            end
        end

        function net = projectCriticWeights(obj, net)
            if ~obj.getConfigValue('useWeightNormCritic', false)
                return;
            end

            radius = obj.getConfigValue('criticWeightNormRadius', 1.0);
            layerNames = obj.getCriticWeightNormLayerNames(net);
            learnables = net.Learnables;
            for i = 1:height(learnables)
                if ~strcmp(learnables.Parameter{i}, 'Weights')
                    continue;
                end
                if ~any(strcmp(string(learnables.Layer{i}), layerNames))
                    continue;
                end

                weights = extractdata(learnables.Value{i});
                rowNorms = sqrt(sum(weights.^2, 2));
                scale = min(1, radius ./ (rowNorms + 1e-8));
                learnables.Value{i} = dlarray(weights .* scale);
            end
            net.Learnables = learnables;
        end

        function layerNames = getCriticWeightNormLayerNames(~, net)
            allLayerNames = string(net.Learnables.Layer);
            if any(allLayerNames == "trunk_fc1")
                layerNames = ["trunk_fc1", "trunk_fc2"];
            else
                layerNames = ["fc1", "fc2"];
            end
        end

        function value = getConfigValue(obj, fieldName, defaultValue)
            if isfield(obj.config, fieldName)
                value = obj.config.(fieldName);
            else
                value = defaultValue;
            end
        end

        function updateObservationNormalizer(obj, states, nextStates)
            if ~obj.getConfigValue('useObservationNormalization', false)
                return;
            end

            samples = single([states; nextStates]);
            batchCount = size(samples, 1);
            if batchCount == 0
                return;
            end

            batchMean = mean(samples, 1)';
            centered = samples - batchMean';
            batchM2 = sum(centered .^ 2, 1)';

            if obj.obsNormCount == 0
                obj.obsNormCount = batchCount;
                obj.obsNormMean = batchMean;
                obj.obsNormM2 = max(batchM2, ones(size(batchM2), 'single'));
                return;
            end

            totalCount = obj.obsNormCount + batchCount;
            delta = batchMean - obj.obsNormMean;
            obj.obsNormMean = obj.obsNormMean + delta * (batchCount / totalCount);
            obj.obsNormM2 = obj.obsNormM2 + batchM2 + ...
                (delta .^ 2) * (obj.obsNormCount * batchCount / totalCount);
            obj.obsNormCount = totalCount;
        end

        function state = normalizeStateVector(obj, state)
            if ~obj.getConfigValue('useObservationNormalization', false) || obj.obsNormCount < 2
                return;
            end
            variance = obj.obsNormM2 / max(1, obj.obsNormCount - 1);
            state = (state - obj.obsNormMean) ./ sqrt(variance + 1e-6);
            clipVal = obj.getConfigValue('observationNormClip', 5.0);
            state = max(-clipVal, min(clipVal, state));
        end

        function states = normalizeStateBatch(obj, states)
            if ~obj.getConfigValue('useObservationNormalization', false) || obj.obsNormCount < 2
                states = single(states);
                return;
            end
            variance = obj.obsNormM2 / max(1, obj.obsNormCount - 1);
            states = single(states);
            states = (states - obj.obsNormMean') ./ sqrt(variance' + 1e-6);
            clipVal = obj.getConfigValue('observationNormClip', 5.0);
            states = max(-clipVal, min(clipVal, states));
        end

        function [net, opt] = adamUpdate(~, net, grads, opt, lr, step)
            % ADAM optimizer update for network
            if isempty(opt.m)
                opt.m = grads;
                opt.v = grads;
                for i = 1:height(grads)
                    opt.m.Value{i} = zeros(size(grads.Value{i}), 'like', grads.Value{i});
                    opt.v.Value{i} = zeros(size(grads.Value{i}), 'like', grads.Value{i});
                end
            end

            % ADAM update
            for i = 1:height(grads)
                opt.m.Value{i} = opt.beta1 * opt.m.Value{i} + (1 - opt.beta1) * grads.Value{i};
                opt.v.Value{i} = opt.beta2 * opt.v.Value{i} + (1 - opt.beta2) * grads.Value{i}.^2;

                mHat = opt.m.Value{i} / (1 - opt.beta1^(step + 1));
                vHat = opt.v.Value{i} / (1 - opt.beta2^(step + 1));

                net.Learnables.Value{i} = net.Learnables.Value{i} - ...
                    lr * mHat ./ (sqrt(vHat) + opt.eps);
            end
        end

        function [param, opt] = adamUpdateScalar(~, param, grad, opt, lr, step)
            % ADAM update for scalar parameter (alpha)
            opt.m = opt.beta1 * opt.m + (1 - opt.beta1) * grad;
            opt.v = opt.beta2 * opt.v + (1 - opt.beta2) * grad^2;

            mHat = opt.m / (1 - opt.beta1^(step + 1));
            vHat = opt.v / (1 - opt.beta2^(step + 1));

            param = param - lr * mHat / (sqrt(vHat) + opt.eps);
        end

        function beta = getPriorityReplayBeta(obj)
            if ~obj.getConfigValue('usePrioritizedReplay', false)
                beta = 0.0;
                return;
            end

            betaStart = obj.getConfigValue('priorityReplayBetaStart', 0.4);
            betaEnd = obj.getConfigValue('priorityReplayBetaEnd', 1.0);
            annealSteps = max(1, obj.getConfigValue('entropyAnnealSteps', 1200));
            progress = min(1, obj.trainingStep / annealSteps);
            beta = betaStart + progress * (betaEnd - betaStart);
        end

        function recentWindowSize = getERERecentWindowSize(obj)
            if ~obj.getConfigValue('useEmphasizingRecentExperience', false)
                recentWindowSize = obj.replayBuffer.size;
                return;
            end

            totalUpdates = max(1, round(obj.getConfigValue('gradientStepsPerTraining', 1)));
            updateIndex = mod(obj.trainingStep, totalUpdates) + 1;
            etaStart = obj.getConfigValue('ereEtaStart', 0.996);
            etaEnd = obj.getConfigValue('ereEtaEnd', 1.0);
            annealSteps = max(1, round(obj.getConfigValue('ereAnnealSteps', ...
                obj.getConfigValue('entropyAnnealSteps', 1200))));
            progress = min(1, obj.trainingStep / annealSteps);
            eta = etaStart + progress * (etaEnd - etaStart);
            exponentScale = obj.getConfigValue('ereExponentScale', 1000);
            minRecentSize = max(obj.config.batchSize, round(obj.getConfigValue('ereMinRecentSize', obj.config.batchSize)));

            % Adapt the ERE replay window to the currently-filled online buffer.
            recentWindowSize = round(obj.replayBuffer.size * eta ^ (updateIndex * exponentScale / totalUpdates));
            recentWindowSize = max(minRecentSize, recentWindowSize);
            recentWindowSize = min(obj.replayBuffer.size, recentWindowSize);
        end

        function targetQuantiles = buildTQCTargetQuantiles(obj, stateAction, nextStates, rewards, dones, discounts, alpha)
            [nextActions, nextLogProbs] = obj.sampleAction(nextStates);
            nextStateAction = cat(1, nextStates, nextActions);

            if obj.useTargetNetworks
                updateTargetState = obj.getConfigValue('targetCriticTrainMode', false);
                [~, q1Next, targetState1] = obj.forwardCriticTargets( ...
                    obj.targetCritic1, stateAction, nextStateAction, updateTargetState);
                [~, q2Next, targetState2] = obj.forwardCriticTargets( ...
                    obj.targetCritic2, stateAction, nextStateAction, updateTargetState);
                if updateTargetState
                    obj.targetCritic1.State = targetState1;
                    obj.targetCritic2.State = targetState2;
                end
            else
                useJointBatch = obj.useJointCriticBatch();
                [~, q1Next, criticState1] = obj.forwardCriticTargets( ...
                    obj.critic1, stateAction, nextStateAction, useJointBatch);
                [~, q2Next, criticState2] = obj.forwardCriticTargets( ...
                    obj.critic2, stateAction, nextStateAction, useJointBatch);
                if useJointBatch
                    obj.critic1.State = criticState1;
                    obj.critic2.State = criticState2;
                end
            end

            combinedQuantiles = cat(1, q1Next, q2Next);
            combinedQuantiles = sort(extractdata(combinedQuantiles), 1, 'ascend');
            dropTotal = obj.getConfigValue('tqcDropQuantilesPerCritic', 2) * 2;
            keepCount = max(1, size(combinedQuantiles, 1) - dropTotal);
            truncatedQuantiles = combinedQuantiles(1:keepCount, :);

            rewardsData = extractdata(rewards);
            donesData = extractdata(dones);
            discountsData = extractdata(discounts);
            logProbData = extractdata(nextLogProbs);
            targetQuantiles = rewardsData + discountsData .* (1 - donesData) .* ...
                (truncatedQuantiles - alpha .* logProbData);
            targetQuantiles = dlarray(targetQuantiles, 'CB');
        end

        function tauHat = getTQCTauHat(obj)
            numQuantiles = obj.getConfigValue('tqcNumQuantiles', 7);
            tauHat = dlarray(((2 * (1:numQuantiles) - 1) ./ (2 * numQuantiles))', 'CB');
        end

        function toCPU(obj)
            % Move all agent components to CPU
            
            % 1. Replay Buffer
            obj.replayBuffer.toCPU();
            
            % 2. Networks (gather learnables)
            % Learnables is a table, gather on table works in newer MATLAB, 
            % but safer to gather values if needed. Usually dlnetwork handles it,
            % but we ensure underlying data is CPU.
            obj.actor.Learnables = gather(obj.actor.Learnables);
            if obj.getConfigValue('useREDQCritic', false)
                for criticIdx = 1:numel(obj.critics)
                    obj.critics{criticIdx}.Learnables = gather(obj.critics{criticIdx}.Learnables);
                end
                if obj.useTargetNetworks
                    for criticIdx = 1:numel(obj.targetCritics)
                        obj.targetCritics{criticIdx}.Learnables = gather(obj.targetCritics{criticIdx}.Learnables);
                    end
                end
                obj.syncREDQMirrors();
            else
                obj.critic1.Learnables = gather(obj.critic1.Learnables);
                obj.critic2.Learnables = gather(obj.critic2.Learnables);
                if obj.useTargetNetworks
                    obj.targetCritic1.Learnables = gather(obj.targetCritic1.Learnables);
                    obj.targetCritic2.Learnables = gather(obj.targetCritic2.Learnables);
                end
            end
            
            % 3. Optimizers (gather states)
            obj.actorOptimizer = obj.gatherOptimizer(obj.actorOptimizer);
            if obj.getConfigValue('useREDQCritic', false)
                for criticIdx = 1:numel(obj.criticOptimizers)
                    obj.criticOptimizers{criticIdx} = obj.gatherOptimizer(obj.criticOptimizers{criticIdx});
                end
                obj.syncREDQMirrors();
            else
                obj.critic1Optimizer = obj.gatherOptimizer(obj.critic1Optimizer);
                obj.critic2Optimizer = obj.gatherOptimizer(obj.critic2Optimizer);
            end
            obj.alphaOptimizer = obj.gatherOptimizer(obj.alphaOptimizer);
            
            % 4. Scalars
            obj.logAlpha = gather(obj.logAlpha);
            
            % Update config
            obj.config.useGPU = false;
        end

        function opt = gatherOptimizer(~, opt)
            if isempty(opt)
                return;
            end

            % Handle scalar optimizer (alpha) - has 'm' and 'v' fields
            if isfield(opt, 'm') && isnumeric(opt.m)
                opt.m = gather(opt.m);
                opt.v = gather(opt.v);
                return;
            end

            % Handle network Adam optimizer - struct with 'avg' and 'avgSq' fields
            if isfield(opt, 'avg')
                if ~isempty(opt.avg) && istable(opt.avg)
                    for i = 1:height(opt.avg)
                        opt.avg.Value{i} = gather(opt.avg.Value{i});
                    end
                end
                if ~isempty(opt.avgSq) && istable(opt.avgSq)
                    for i = 1:height(opt.avgSq)
                        opt.avgSq.Value{i} = gather(opt.avgSq.Value{i});
                    end
                end
                return;
            end
        end

        function clippedGrads = clipGradients(~, gradients, maxNorm)
            % Clip gradients by global norm to prevent explosion
            %
            % Inputs:
            %   gradients: Table of gradients from dlfeval
            %   maxNorm: Maximum allowed gradient norm
            %
            % Outputs:
            %   clippedGrads: Gradients clipped to maxNorm

            % Calculate global gradient norm
            totalNorm = 0;
            for i = 1:height(gradients)
                grad = gradients.Value{i};
                % Check for NaN/Inf and reset to zero if found
                if any(isnan(grad(:))) || any(isinf(grad(:)))
                    warning('RRSACPSO:NaNGradient', 'NaN/Inf detected in gradient %s. Resetting to zeros.', gradients.Parameter{i});
                    gradients.Value{i} = zeros(size(grad), 'like', grad);
                    grad = gradients.Value{i};
                end
                totalNorm = totalNorm + sum(grad(:).^2);
            end
            totalNorm = sqrt(totalNorm);

            % Clip if needed
            if totalNorm > maxNorm
                clipCoef = maxNorm / (totalNorm + 1e-6);
                for i = 1:height(gradients)
                    gradients.Value{i} = gradients.Value{i} * clipCoef;
                end
            end

            clippedGrads = gradients;
        end
    end
end

% Local functions for dlfeval (must be outside classdef)
function [loss, gradients, state, pred] = criticModelLoss(net, input, auxInput, target, useJointBatch, sampleWeights)
    % Critic loss: MSE between prediction and target, with optional joint
    % current/next forwarding to keep BatchNorm statistics aligned.
    if nargin < 5
        useJointBatch = false;
    end
    if nargin < 6 || isempty(sampleWeights)
        sampleWeights = dlarray(ones(1, size(input, 2), 'single'), 'CB');
    end

    if useJointBatch && ~isempty(auxInput)
        jointInput = cat(2, input, auxInput);
        [jointPred, state] = forward(net, jointInput);
        batchSize = size(input, 2);
        pred = jointPred(:, 1:batchSize);
    else
        [pred, state] = forward(net, input);
    end

    % Check for NaN/Inf in critic output
    if any(isnan(pred(:))) || any(isinf(pred(:)))
        pred = zeros(size(pred), 'like', pred);
    end

    residual = pred - target;
    squaredError = residual.^2;
    headCount = max(1, size(pred, 1));
    loss = sum(sampleWeights .* squaredError, 'all') / ...
        (headCount * (sum(sampleWeights, 'all') + 1e-8));

    % Check for NaN/Inf in loss
    if isnan(loss) || isinf(loss)
        loss = dlarray(0, 'CB');
    end

    gradients = dlgradient(loss, net.Learnables);
end

function [loss, gradients, state, predMean] = tqcCriticModelLoss(net, input, targetQuantiles, tauHat, huberKappa, sampleWeights)
    if nargin < 6 || isempty(sampleWeights)
        sampleWeights = dlarray(ones(1, size(input, 2), 'single'), 'CB');
    end

    [pred, state] = forward(net, input);
    if any(isnan(pred(:))) || any(isinf(pred(:)))
        pred = zeros(size(pred), 'like', pred);
    end

    predMean = mean(pred, 1);
    numPred = size(pred, 1);
    numTarget = size(targetQuantiles, 1);
    batchSize = size(pred, 2);

    predExpanded = reshape(pred, [numPred, 1, batchSize]);
    targetExpanded = reshape(targetQuantiles, [1, numTarget, batchSize]);
    tdErrors = targetExpanded - predExpanded;
    absTd = abs(tdErrors);
    huber = 0.5 .* tdErrors.^2 .* (absTd <= huberKappa) + ...
        huberKappa .* (absTd - 0.5 .* huberKappa) .* (absTd > huberKappa);

    tauExpanded = reshape(tauHat, [numPred, 1, 1]);
    quantileWeights = abs(tauExpanded - cast(tdErrors < 0, 'like', tdErrors));
    quantileLoss = quantileWeights .* huber ./ huberKappa;
    sampleLoss = reshape(mean(quantileLoss, [1, 2]), 1, []);
    loss = sum(sampleWeights .* sampleLoss, 'all') / (sum(sampleWeights, 'all') + 1e-8);

    if isnan(loss) || isinf(loss)
        loss = dlarray(0, 'CB');
    end

    gradients = dlgradient(loss, net.Learnables);
end

function [loss, gradients, actorState] = redqActorModelLoss(actorNet, states, criticNets, alphaVal)
    [output, actorState] = forward(actorNet, states);
    if any(isnan(output(:))) || any(isinf(output(:)))
        output = zeros(size(output), 'like', output);
    end

    actionSize = size(output, 1) / 2;
    meanActions = output(1:actionSize, :);
    logStd = output(actionSize+1:end, :);
    logStd = clampPolicyLogStd(logStd);
    std = exp(logStd);
    epsilon = randn(size(meanActions), 'like', meanActions);
    unsquashedActions = meanActions + std .* epsilon;
    actions = tanh(unsquashedActions);

    gaussianLogProb = -0.5 * sum(epsilon.^2 + 2 * logStd + log(2 * pi), 1);
    actionSquared = min(actions.^2, 0.9999);
    tanhCorrection = sum(log(1 - actionSquared + 1e-6), 1);
    logProbs = max(min(gaussianLogProb - tanhCorrection, 100), -100);

    stateAction = cat(1, states, actions);
    qCells = cell(1, numel(criticNets));
    for criticIdx = 1:numel(criticNets)
        qValue = forward(criticNets{criticIdx}, stateAction);
        if any(isnan(qValue(:))) || any(isinf(qValue(:)))
            qValue = zeros(size(qValue), 'like', qValue);
        end
        if size(qValue, 1) > 1
            qValue = mean(qValue, 1);
        end
        qCells{criticIdx} = qValue;
    end
    q = mean(cat(1, qCells{:}), 1);

    loss = mean(alphaVal .* logProbs - q);
    if isnan(loss) || isinf(loss)
        loss = dlarray(0, 'CB');
    end

    gradients = dlgradient(loss, actorNet.Learnables);
end

function [loss, gradients, actorState] = actorModelLoss(actorNet, states, critic1Net, critic2Net, alphaVal, useAQECritic, useTQCCritic)
    % Standard SAC actor loss.
    if nargin < 6
        useAQECritic = false;
    end
    if nargin < 7
        useTQCCritic = false;
    end

    % Sample actions from actor
    [output, actorState] = forward(actorNet, states);

    % Check for NaN/Inf in actor output
    if any(isnan(output(:))) || any(isinf(output(:)))
        output = zeros(size(output), 'like', output);
    end

    actionSize = size(output, 1) / 2;
    meanActions = output(1:actionSize, :);
    logStd = output(actionSize+1:end, :);
    logStd = clampPolicyLogStd(logStd);
    std = exp(logStd);
    epsilon = randn(size(meanActions), 'like', meanActions);
    unsquashedActions = meanActions + std .* epsilon;
    actions = tanh(unsquashedActions);

    % Compute log probabilities
    gaussianLogProb = -0.5 * sum(epsilon.^2 + 2 * logStd + log(2 * pi), 1);

    % Robust tanh correction with clipping to prevent NaN
    actionSquared = actions.^2;
    actionSquared = min(actionSquared, 0.9999);  % Prevent 1 - x^2 from being too small
    tanhCorrection = sum(log(1 - actionSquared + 1e-6), 1);

    logProbs = gaussianLogProb - tanhCorrection;

    % Final safety: Clip log probs to reasonable range
    logProbs = max(min(logProbs, 100), -100);

    % Get Q-values
    stateAction = cat(1, states, actions);
    q1 = forward(critic1Net, stateAction);
    q2 = forward(critic2Net, stateAction);

    % Check for NaN/Inf in Q-values
    if any(isnan(q1(:))) || any(isinf(q1(:)))
        q1 = zeros(size(q1), 'like', q1);
    end
    if any(isnan(q2(:))) || any(isinf(q2(:)))
        q2 = zeros(size(q2), 'like', q2);
    end

    if useAQECritic || useTQCCritic
        q = mean(cat(1, q1, q2), 1);
    else
        if size(q1, 1) > 1
            q1 = mean(q1, 1);
        end
        if size(q2, 1) > 1
            q2 = mean(q2, 1);
        end
        q = min(q1, q2);
    end

    loss = mean(alphaVal .* logProbs - q);

    % Check for NaN/Inf in loss
    if isnan(loss) || isinf(loss)
        loss = dlarray(0, 'CB');
    end

    gradients = dlgradient(loss, actorNet.Learnables);
end

function value = getFieldOrDefault(structValue, fieldName, defaultValue)
    if isfield(structValue, fieldName)
        value = structValue.(fieldName);
    else
        value = defaultValue;
    end
end

function logStd = clampPolicyLogStd(logStd)
    minStd = 0.01;
    maxStd = 1.0;
    logStd = max(min(logStd, log(maxStd)), log(minStd));
end
