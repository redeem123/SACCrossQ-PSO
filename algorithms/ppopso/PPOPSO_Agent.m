classdef PPOPSO_Agent < handle
    % PPO agent with continuous Gaussian policy for PSO parameter control.
    % Adapted from Klein et al. 2024 (iSOMA-RL) to PSO.
    %
    % Actor outputs 3D mean (w, c1, c2); log_std is a learnable parameter.
    % Actions are tanh-squashed and mapped to parameter ranges.

    properties
        config
        actor
        critic
        logStd        % Learnable 3x1 log standard deviation
        actorOptimizer
        criticOptimizer
        logStdOptimizer
        adamStep = 0  % Cumulative Adam step counter (persists across train() calls)
    end

    methods
        function obj = PPOPSO_Agent(config)
            obj.config = config;
            obj.actor = createActorNetwork(config);
            obj.critic = createCriticNetwork(config);
            obj.logStd = dlarray(ones(config.actionDim, 1, 'single') * config.initLogStd);
            obj.actorOptimizer = struct('avg', [], 'avgSq', []);
            obj.criticOptimizer = struct('avg', [], 'avgSq', []);
            obj.logStdOptimizer = struct('avg', [], 'avgSq', []);
        end

        function [action, logProb, value] = getAction(obj, state)
            stateDL = dlarray(state, 'CB');
            mean_raw = forward(obj.actor, stateDL);
            mean_raw = extractdata(mean_raw);

            std = exp(extractdata(obj.logStd));

            % Sample from Gaussian, then tanh squash
            noise = randn(obj.config.actionDim, 1) .* std;
            raw = mean_raw + noise;
            action_squashed = tanh(raw);  % in [-1, 1]

            % Map to parameter ranges
            action = mapToParams(action_squashed, obj.config);

            % Log probability under squashed Gaussian
            logProb = gaussianLogProb(raw, mean_raw, std, action_squashed);

            value = forward(obj.critic, stateDL);
            value = extractdata(value);
        end

        function value = getValue(obj, state)
            stateDL = dlarray(state, 'CB');
            value = forward(obj.critic, stateDL);
            value = extractdata(value);
        end

        function losses = train(obj, states, rawActions, oldLogProbs, returns, advantages)
            statesDL = dlarray(states, 'CB');
            rawActionsDL = dlarray(rawActions, 'CB');
            oldLogProbsDL = dlarray(oldLogProbs, 'CB');
            returnsDL = dlarray(returns, 'CB');
            advantagesDL = dlarray(advantages, 'CB');

            numSamples = size(states, 2);
            minibatchSize = min(obj.config.minibatchSize, numSamples);
            idx = 1:numSamples;

            actorLossSum = 0;
            criticLossSum = 0;
            updateCount = 0;

            for epoch = 1:obj.config.updateEpochs
                idx = idx(randperm(numSamples));
                for startIdx = 1:minibatchSize:numSamples
                    endIdx = min(startIdx + minibatchSize - 1, numSamples);
                    batchIdx = idx(startIdx:endIdx);

                    bS = statesDL(:, batchIdx);
                    bA = rawActionsDL(:, batchIdx);
                    bOLP = oldLogProbsDL(:, batchIdx);
                    bR = returnsDL(:, batchIdx);
                    bAdv = advantagesDL(:, batchIdx);

                    updateCount = updateCount + 1;
                    obj.adamStep = obj.adamStep + 1;

                    % Actor + logStd update
                    [aLoss, aGrads, lsGrads] = dlfeval(@ppoActorLoss, ...
                        obj.actor, obj.logStd, bS, bA, bOLP, bAdv, ...
                        obj.config.clipEpsilon, obj.config.entropyCoef);
                    aGrads = clipLearnablesGradients(aGrads, obj.config.maxGradNorm);
                    [obj.actor.Learnables, obj.actorOptimizer.avg, obj.actorOptimizer.avgSq] = ...
                        adamupdate(obj.actor.Learnables, aGrads, ...
                        obj.actorOptimizer.avg, obj.actorOptimizer.avgSq, ...
                        obj.adamStep, obj.config.actorLR, 0.9, 0.999, 1e-8);
                    % Update logStd
                    lsGrads = thresholdGrad(lsGrads, obj.config.maxGradNorm);
                    [obj.logStd, obj.logStdOptimizer.avg, obj.logStdOptimizer.avgSq] = ...
                        adamupdate(obj.logStd, lsGrads, ...
                        obj.logStdOptimizer.avg, obj.logStdOptimizer.avgSq, ...
                        obj.adamStep, obj.config.actorLR, 0.9, 0.999, 1e-8);

                    actorLossSum = actorLossSum + extractdata(aLoss);

                    % Critic update
                    [cLoss, cGrads] = dlfeval(@ppoCriticLoss, obj.critic, bS, bR, obj.config.valueCoef);
                    cGrads = clipLearnablesGradients(cGrads, obj.config.maxGradNorm);
                    [obj.critic.Learnables, obj.criticOptimizer.avg, obj.criticOptimizer.avgSq] = ...
                        adamupdate(obj.critic.Learnables, cGrads, ...
                        obj.criticOptimizer.avg, obj.criticOptimizer.avgSq, ...
                        obj.adamStep, obj.config.criticLR, 0.9, 0.999, 1e-8);
                    criticLossSum = criticLossSum + extractdata(cLoss);
                end
            end

            losses = struct();
            if updateCount > 0
                losses.actor = actorLossSum / updateCount;
                losses.critic = criticLossSum / updateCount;
            else
                losses.actor = NaN;
                losses.critic = NaN;
            end
        end
    end
end

%% Network constructors
function net = createActorNetwork(config)
    layers = [
        featureInputLayer(config.stateSize, 'Name', 'input', 'Normalization', 'none')
    ];
    for i = 1:length(config.actorHiddenLayers)
        h = config.actorHiddenLayers(i);
        layers = [layers
            fullyConnectedLayer(h, 'Name', sprintf('fc%d', i))
            tanhLayer('Name', sprintf('tanh%d', i))]; %#ok<AGROW>
    end
    layers = [layers
        fullyConnectedLayer(config.actionDim, 'Name', 'mean_out')];
    net = dlnetwork(layerGraph(layers));
    net = initializeNetworkWeights(net);
end

function net = createCriticNetwork(config)
    layers = [
        featureInputLayer(config.stateSize, 'Name', 'input', 'Normalization', 'none')
    ];
    for i = 1:length(config.criticHiddenLayers)
        h = config.criticHiddenLayers(i);
        layers = [layers
            fullyConnectedLayer(h, 'Name', sprintf('fc%d', i))
            tanhLayer('Name', sprintf('tanh%d', i))]; %#ok<AGROW>
    end
    layers = [layers
        fullyConnectedLayer(1, 'Name', 'value')];
    net = dlnetwork(layerGraph(layers));
    net = initializeNetworkWeights(net);
end


%% Loss functions
function [loss, actorGrads, logStdGrads] = ppoActorLoss(actorNet, logStd, states, rawActions, oldLogProbs, advantages, clipEps, entCoef)
    means = forward(actorNet, states);
    std = exp(logStd);

    % Log prob under current policy
    logProbs = gaussianLogProbDL(rawActions, means, std);

    ratio = exp(logProbs - oldLogProbs);
    clipped = max(min(ratio, 1 + clipEps), 1 - clipEps);
    surr = -mean(min(ratio .* advantages, clipped .* advantages));

    % Entropy bonus: H = 0.5 * ln(2*pi*e*sigma^2) per dim
    entropy = mean(sum(0.5 * log(2 * pi * exp(1) * std.^2)));

    loss = surr - entCoef * entropy;
    [actorGrads, logStdGrads] = dlgradient(loss, actorNet.Learnables, logStd);
end

function [loss, grads] = ppoCriticLoss(criticNet, states, returns, valueCoef)
    values = forward(criticNet, states);
    loss = valueCoef * mean((returns - values).^2);
    grads = dlgradient(loss, criticNet.Learnables);
end

%% Gaussian log probability (squashed)
function lp = gaussianLogProb(raw, mean_val, std_val, squashed)
    % Log prob of raw under N(mean, std), minus tanh correction
    var = std_val.^2 + 1e-8;
    lp_raw = -0.5 * sum((raw - mean_val).^2 ./ var + log(var) + log(2*pi));
    % Tanh squash correction: -sum(log(1 - tanh(raw)^2 + eps))
    lp_correction = sum(log(max(1 - squashed.^2, 1e-6)));
    lp = lp_raw - lp_correction;
end

function lp = gaussianLogProbDL(rawActions, means, std)
    % Batched dlarray version
    var = std.^2 + 1e-8;
    lp_raw = -0.5 * sum((rawActions - means).^2 ./ var + log(var) + log(2*pi), 1);
    squashed = tanh(rawActions);
    lp_correction = sum(log(max(1 - squashed.^2, 1e-6)), 1);
    lp = lp_raw - lp_correction;
end

%% Map squashed action [-1,1] to parameter ranges
function params = mapToParams(action_squashed, config)
    % action_squashed is 3x1 in [-1, 1]
    w  = config.wMin  + (action_squashed(1) + 1)/2 * (config.wMax  - config.wMin);
    c1 = config.c1Min + (action_squashed(2) + 1)/2 * (config.c1Max - config.c1Min);
    c2 = config.c2Min + (action_squashed(3) + 1)/2 * (config.c2Max - config.c2Min);
    params = [w; c1; c2];
end

%% Gradient utilities — uses shared/neural_networks/optimization/clipLearnablesGradients.m

function g = thresholdGrad(g, maxNorm)
    g(isnan(g) | isinf(g)) = 0;
    n = sqrt(sum(g(:).^2));
    if n > maxNorm
        g = g * (maxNorm / (n + 1e-6));
    end
end
