classdef APEXPSO_Agent < handle
    % CQSAC-PSO SAC Agent with Automatic Entropy Tuning
    %
    % Implements Soft Actor-Critic (SAC) with CrossQ optimizations:
    %   - Automatic entropy tuning (temperature parameter α)
    %   - Twin critics for stability
    %   - BatchNorm in networks (CrossQ)
    %   - Off-policy learning with experience replay
    %
    % Based on:
    %   - SAC: Haarnoja et al. (2018)
    %   - CrossQ: Bhat et al. (ICLR 2024)

    properties
        config          % CQSAC-PSO configuration

        % Networks
        actor           % Actor network (Gaussian policy)
        critic1         % First critic network
        critic2         % Second critic network
        targetCritic1   % Target critic network (optional)
        targetCritic2   % Target critic network (optional)

        % Optimizers (using ADAM)
        actorOptimizer
        critic1Optimizer
        critic2Optimizer
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
        disagreementEMA
    end

    methods
        function obj = APEXPSO_Agent(config)
            % Constructor
            obj.config = config;

            % Create networks
            obj.actor = createActorNetwork(config);
            obj.critic1 = createCriticNetwork(config);
            obj.critic2 = createCriticNetwork(config);

            % Optional target networks (standard SAC)
            obj.useTargetNetworks = isfield(config, 'useTargetNetworks') && config.useTargetNetworks;
            if obj.useTargetNetworks
                obj.targetCritic1 = obj.cloneNetwork(obj.critic1);
                obj.targetCritic2 = obj.cloneNetwork(obj.critic2);
            end

            % Initialize entropy temperature
            obj.logAlpha = log(config.initAlpha);
            obj.targetEntropy = config.targetEntropy;

            % Create optimizers (Adam for networks, custom Adam for alpha scalar)
            % avg = first moment, avgSq = second moment (auto-initialized by adamupdate)
            obj.actorOptimizer   = struct('avg', [], 'avgSq', []);
            obj.critic1Optimizer = struct('avg', [], 'avgSq', []);
            obj.critic2Optimizer = struct('avg', [], 'avgSq', []);
            obj.alphaOptimizer = struct('m', 0, 'v', 0, 'beta1', 0.9, 'beta2', 0.999, 'eps', 1e-8);

            % Create replay buffer
            obj.replayBuffer = ReplayBuffer(config.bufferSize, ...
                config.stateSize, config.actionSize, config.useGPU, config);

            % Initialize training state
            obj.trainingStep = 0;
            obj.explorationNoise = config.explorationNoiseStart;
            obj.disagreementEMA = 0;
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

            % Convert to dlarray
            stateDL = dlarray(state, 'CB');

            % Forward pass through actor (outputs mean and log_std concatenated)
            % Use predict in evaluation to avoid batch-norm collapse with batch size 1.
            if training
                output = forward(obj.actor, stateDL);
            else
                output = predict(obj.actor, stateDL);
            end
            output = extractdata(output);

            % Check for NaN/Inf in network output
            if any(isnan(output(:))) || any(isinf(output(:)))
                warning('APEXPSO:NaNDetected', 'NaN/Inf detected in actor network output. Resetting to zeros.');
                output = zeros(size(output));
            end

            % Split into mean and log_std
            actionSize = obj.config.actionSize;
            meanAction = output(1:actionSize);
            logStd = output(actionSize+1:end);

            % Clip logStd BEFORE exp() to prevent overflow
            logStd = max(min(logStd, 2.0), -20.0);

            if training
                % Sample from Gaussian distribution (SAC stochastic policy)
                std = exp(logStd);
                std = max(min(std, 1.0), 0.01);  % Clip std to reasonable range
                action = meanAction + randn(size(meanAction)) .* std;
            else
                % Deterministic action for evaluation
                action = meanAction;
            end

            % Apply tanh squashing to bound actions to [-1, 1]
            action = tanh(action);

            % Final NaN check after all operations
            if any(isnan(action(:))) || any(isinf(action(:)))
                warning('APEXPSO:NaNDetected', 'NaN/Inf detected in final action. Resetting to zeros.');
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
            [states, actions, rewards, nextStates, dones] = ...
                obj.replayBuffer.sample(obj.config.batchSize);

            % Convert to dlarray
            statesDL = dlarray(states', 'CB');
            actionsDL = dlarray(actions', 'CB');
            nextStatesDL = dlarray(nextStates', 'CB');
            donesDL = dlarray(dones', 'CB');
            rewardsDL = dlarray(rewards', 'CB');

            % Get current alpha
            alpha = exp(obj.logAlpha);

            % ===== UPDATE CRITICS =====
            % Compute target Q-value using next state
            [nextActions, nextLogProbs] = obj.sampleAction(nextStatesDL);

            % Concatenate next_state and next_action for critics
            nextStatAction = cat(1, nextStatesDL, nextActions);

            % Q-values from both critics (take minimum for stability)
            if obj.useTargetNetworks
                q1Next = forward(obj.targetCritic1, nextStatAction);
                q2Next = forward(obj.targetCritic2, nextStatAction);
            else
                q1Next = forward(obj.critic1, nextStatAction);
                q2Next = forward(obj.critic2, nextStatAction);
            end
            qNext = min(q1Next, q2Next);
            disagreementNext = abs(q1Next - q2Next);
            obj.updateDisagreementEMA(extractdata(mean(disagreementNext)));

            % SAC target: r + γ * (Q(s',a') - α * log π(a'|s'))
            targetQ = rewardsDL + obj.config.gamma .* (1 - donesDL) .* ...
                     (qNext - alpha .* nextLogProbs);
            % Stop gradient (detach from computation graph)
            targetQ = dlarray(extractdata(targetQ), 'CB');

            % Critic loss: MSE between Q and target
            stateAction = cat(1, statesDL, actionsDL);

            % Track TD error magnitude for diagnostics.
            q1Pred = forward(obj.critic1, stateAction);
            q2Pred = forward(obj.critic2, stateAction);
            tdErr = 0.5 .* (abs(extractdata(q1Pred - targetQ)) + abs(extractdata(q2Pred - targetQ)));
            meanTdErr = mean(tdErr(:));

            % Train critic 1 using dlfeval
            useHuber = isfield(obj.config, 'useHuberCriticLoss') && obj.config.useHuberCriticLoss;
            huberDelta = obj.getConfigValue('criticHuberDelta', 2.0);
            useOverestPenalty = isfield(obj.config, 'useOverestimationPenalty') && obj.config.useOverestimationPenalty;
            overestTau = obj.getConfigValue('overestimationPenaltyTau', 0.75);
            [loss1, critic1Grads] = dlfeval(@criticModelLoss, obj.critic1, stateAction, targetQ, ...
                useHuber, huberDelta, useOverestPenalty, overestTau);
            % Gradient clipping for critic 1
            critic1Grads = obj.clipGradients(critic1Grads, 1.0);
            [obj.critic1.Learnables, obj.critic1Optimizer.avg, obj.critic1Optimizer.avgSq] = ...
                adamupdate(obj.critic1.Learnables, critic1Grads, ...
                obj.critic1Optimizer.avg, obj.critic1Optimizer.avgSq, ...
                obj.trainingStep + 1, obj.config.criticLR);

            % Train critic 2 using dlfeval
            [loss2, critic2Grads] = dlfeval(@criticModelLoss, obj.critic2, stateAction, targetQ, ...
                useHuber, huberDelta, useOverestPenalty, overestTau);
            % Gradient clipping for critic 2
            critic2Grads = obj.clipGradients(critic2Grads, 1.0);
            [obj.critic2.Learnables, obj.critic2Optimizer.avg, obj.critic2Optimizer.avgSq] = ...
                adamupdate(obj.critic2.Learnables, critic2Grads, ...
                obj.critic2Optimizer.avg, obj.critic2Optimizer.avgSq, ...
                obj.trainingStep + 1, obj.config.criticLR);

            criticLoss = extractdata(loss1) + extractdata(loss2);
            if obj.useTargetNetworks
                obj.softUpdateTargetNetworks();
            end

            % ===== UPDATE ACTOR =====
            uncertaintyPenaltyWeight = 0.0;
            if isfield(obj.config, 'useActorUncertaintyPenalty') && obj.config.useActorUncertaintyPenalty && ...
                    isfield(obj.config, 'uncertaintyPenaltyWeight')
                uncertaintyPenaltyWeight = obj.config.uncertaintyPenaltyWeight;
            end
            [actorLoss, actorGrads] = dlfeval(@actorModelLoss, obj.actor, statesDL, ...
                obj.critic1, obj.critic2, alpha, uncertaintyPenaltyWeight);
            % Gradient clipping for actor
            actorGrads = obj.clipGradients(actorGrads, 1.0);
            [obj.actor.Learnables, obj.actorOptimizer.avg, obj.actorOptimizer.avgSq] = ...
                adamupdate(obj.actor.Learnables, actorGrads, ...
                obj.actorOptimizer.avg, obj.actorOptimizer.avgSq, ...
                obj.trainingStep + 1, obj.config.actorLR);

            % ===== UPDATE ALPHA (Automatic Entropy Tuning) =====
            targetEntropyNow = obj.targetEntropy;
            if isfield(obj.config, 'entropyAnnealStrength') && isfield(obj.config, 'entropyAnnealSteps')
                annealProgress = min(1, obj.trainingStep / max(1, obj.config.entropyAnnealSteps));
                annealScale = 1 - obj.config.entropyAnnealStrength * annealProgress;
                targetEntropyNow = obj.targetEntropy * annealScale;
            end
            if isfield(obj.config, 'useEntropyUncertaintyCoupling') && obj.config.useEntropyUncertaintyCoupling
                uncertaintySignal = 1 - exp(-obj.disagreementEMA);
                targetEntropyNow = targetEntropyNow * (1 + obj.getConfigValue('entropyUncertaintyGain', 0.25) * uncertaintySignal);
            end

            [alphaLoss, avgLogProb] = obj.alphaLossFunc(statesDL, targetEntropyNow);

            % Manual gradient for log_alpha: d/d(log_alpha) [alpha * (-log_prob - target)]
            % = exp(log_alpha) * (-log_prob - target) = -log_prob - target
            avgLogProb = extractdata(avgLogProb);
            alphaGrad = -avgLogProb - targetEntropyNow;

            % Safety: Check for NaN/Inf and clip gradient
            if isnan(alphaGrad) || isinf(alphaGrad)
                alphaGrad = 0;  % Skip update if invalid
            else
                alphaGrad = max(min(alphaGrad, 10), -10);  % Clip gradient
            end

            [obj.logAlpha, obj.alphaOptimizer] = obj.adamUpdateScalar(obj.logAlpha, ...
                alphaGrad, obj.alphaOptimizer, obj.config.alphaLR, obj.trainingStep);

            % Safety: Clip log_alpha to reasonable range and check for NaN
            if isnan(obj.logAlpha) || isinf(obj.logAlpha)
                obj.logAlpha = log(0.2);  % Reset to default
            else
                obj.logAlpha = max(min(obj.logAlpha, 1), -10);  % alpha in [0.000045, 2.718]
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
            losses.disagreementEMA = obj.disagreementEMA;
            losses.meanTdError = meanTdErr;
            losses.actorLoss = losses.actor;
            losses.critic1Loss = extractdata(loss1);
            losses.critic2Loss = extractdata(loss2);
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
            output = forward(obj.actor, states);

            % Check for NaN/Inf in network output
            if any(isnan(output(:))) || any(isinf(output(:)))
                warning('APEXPSO:NaNDetected', 'NaN/Inf detected in sampleAction forward pass. Resetting to zeros.');
                output = zeros(size(output), 'like', output);
            end

            % Split into mean and log_std
            actionSize = obj.config.actionSize;
            meanActions = output(1:actionSize, :);
            logStd = output(actionSize+1:end, :);

            % Clip log_std to reasonable range
            logStd = max(min(logStd, 2.0), -20.0);
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

        function netCopy = cloneNetwork(~, net)
            % Clone a dlnetwork with identical learnables and state.
            netCopy = dlnetwork(layerGraph(net.Layers));
            netCopy.Learnables = net.Learnables;
            netCopy.State = net.State;
        end

        function softUpdateTargetNetworks(obj)
            % Soft update target critics (standard SAC).
            tau = obj.config.tau;
            obj.targetCritic1 = obj.softUpdateNetwork(obj.targetCritic1, obj.critic1, tau);
            obj.targetCritic2 = obj.softUpdateNetwork(obj.targetCritic2, obj.critic2, tau);
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
            q = min(q1, q2);

            % SAC objective: maximize Q - α*log_prob
            % Loss: minimize -(Q - α*log_prob)
            loss = mean(alpha .* logProbs - q);
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

        function value = getConfigValue(obj, fieldName, defaultValue)
            if isfield(obj.config, fieldName)
                value = obj.config.(fieldName);
            else
                value = defaultValue;
            end
        end

        function updateDisagreementEMA(obj, batchDisagreement)
            if isnan(batchDisagreement) || isinf(batchDisagreement)
                return;
            end
            ema = 0.95;
            obj.disagreementEMA = ema * obj.disagreementEMA + (1 - ema) * batchDisagreement;
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

        function toCPU(obj)
            % Move all agent components to CPU
            
            % 1. Replay Buffer
            obj.replayBuffer.toCPU();
            
            % 2. Networks (gather learnables)
            % Learnables is a table, gather on table works in newer MATLAB, 
            % but safer to gather values if needed. Usually dlnetwork handles it,
            % but we ensure underlying data is CPU.
            obj.actor.Learnables = gather(obj.actor.Learnables);
            obj.critic1.Learnables = gather(obj.critic1.Learnables);
            obj.critic2.Learnables = gather(obj.critic2.Learnables);
            if obj.useTargetNetworks
                obj.targetCritic1.Learnables = gather(obj.targetCritic1.Learnables);
                obj.targetCritic2.Learnables = gather(obj.targetCritic2.Learnables);
            end
            
            % 3. Optimizers (gather states)
            obj.actorOptimizer = obj.gatherOptimizer(obj.actorOptimizer);
            obj.critic1Optimizer = obj.gatherOptimizer(obj.critic1Optimizer);
            obj.critic2Optimizer = obj.gatherOptimizer(obj.critic2Optimizer);
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
                    warning('APEXPSO:NaNGradient', 'NaN/Inf detected in gradient %s. Resetting to zeros.', gradients.Parameter{i});
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
function [loss, gradients] = criticModelLoss(net, input, target, useHuber, huberDelta, useOverestPenalty, overestTau)
    % Critic loss: MSE between prediction and target
    if nargin < 4
        useHuber = false;
    end
    if nargin < 5
        huberDelta = 2.0;
    end
    if nargin < 6
        useOverestPenalty = false;
    end
    if nargin < 7
        overestTau = 0.75;
    end

    pred = forward(net, input);

    % Check for NaN/Inf in critic output
    if any(isnan(pred(:))) || any(isinf(pred(:)))
        pred = zeros(size(pred), 'like', pred);
    end

    residual = pred - target;
    if useHuber
        absResidual = abs(residual);
        quadratic = min(absResidual, huberDelta);
        linear = absResidual - quadratic;
        perSampleLoss = 0.5 * quadratic.^2 + huberDelta .* linear;
    else
        perSampleLoss = residual.^2;
    end

    if useOverestPenalty
        overestTau = max(min(overestTau, 0.99), 0.5);
        residualData = extractdata(residual);
        weightData = overestTau * (residualData >= 0) + (1 - overestTau) * (residualData < 0);
        perSampleLoss = perSampleLoss .* dlarray(weightData, 'CB');
    end

    loss = mean(perSampleLoss);

    % Check for NaN/Inf in loss
    if isnan(loss) || isinf(loss)
        loss = dlarray(0, 'CB');
    end

    gradients = dlgradient(loss, net.Learnables);
end

function [loss, gradients] = actorModelLoss(actorNet, states, critic1Net, critic2Net, alphaVal, ...
    uncertaintyPenaltyWeight)
    % Actor loss for SAC
    if nargin < 6
        uncertaintyPenaltyWeight = 0.0;
    end
    % Sample actions from actor
    output = forward(actorNet, states);

    % Check for NaN/Inf in actor output
    if any(isnan(output(:))) || any(isinf(output(:)))
        output = zeros(size(output), 'like', output);
    end

    actionSize = size(output, 1) / 2;
    meanActions = output(1:actionSize, :);
    logStd = output(actionSize+1:end, :);
    logStd = max(min(logStd, 2.0), -20.0);
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

    q = min(q1, q2);

    % SAC actor loss with critic-disagreement uncertainty penalty.
    disagreement = abs(q1 - q2);
    loss = mean(alphaVal .* logProbs - q + uncertaintyPenaltyWeight .* disagreement);

    % Check for NaN/Inf in loss
    if isnan(loss) || isinf(loss)
        loss = dlarray(0, 'CB');
    end

    gradients = dlgradient(loss, actorNet.Learnables);
end
