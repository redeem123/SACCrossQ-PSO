classdef PPOPSO_Agent < handle
    % PPO agent for PPOPSO discrete subgroup actions.

    properties
        config
        actor
        critic
        actorOptimizer
        criticOptimizer
    end

    methods
        function obj = PPOPSO_Agent(config)
            obj.config = config;
            obj.actor = createPPOActorNetwork(config);
            obj.critic = createPPOCriticNetwork(config);
            obj.actorOptimizer = [];
            obj.criticOptimizer = [];
        end

        function [action, logProb, value] = getAction(obj, state)
            stateDL = dlarray(state, 'CB');
            logits = forward(obj.actor, stateDL);

            numActions = obj.config.numActionConfigs;
            numSubgroups = obj.config.numSubgroups;
            logits = reshape(extractdata(logits), numActions, numSubgroups);

            action = zeros(numSubgroups, 1);
            logProb = 0;
            for j = 1:numSubgroups
                probs = softmaxVector(logits(:, j));
                action(j) = sampleDiscrete(probs);
                logProb = logProb + log(probs(action(j)) + 1e-8);
            end

            value = forward(obj.critic, stateDL);
            value = extractdata(value);
        end

        function value = getValue(obj, state)
            stateDL = dlarray(state, 'CB');
            value = forward(obj.critic, stateDL);
            value = extractdata(value);
        end

        function losses = train(obj, states, actions, oldLogProbs, returns, advantages)
            statesDL = dlarray(states, 'CB');
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

                    batchStates = statesDL(:, batchIdx);
                    batchActions = actions(:, batchIdx);
                    batchOldLogProbs = oldLogProbsDL(:, batchIdx);
                    batchReturns = returnsDL(:, batchIdx);
                    batchAdvantages = advantagesDL(:, batchIdx);

                    [actorLoss, actorGrads] = dlfeval(@ppoActorLoss, obj.actor, batchStates, ...
                        batchActions, batchOldLogProbs, batchAdvantages, obj.config.clipEpsilon, ...
                        obj.config.entropyCoef, obj.config.numActionConfigs, obj.config.numSubgroups);
                    actorGrads = obj.clipGradients(actorGrads, obj.config.maxGradNorm);
                    [obj.actor.Learnables, obj.actorOptimizer] = sgdmupdate( ...
                        obj.actor.Learnables, actorGrads, obj.actorOptimizer, obj.config.actorLR);
                    actorLossSum = actorLossSum + extractdata(actorLoss);

                    [criticLoss, criticGrads] = dlfeval(@ppoCriticLoss, obj.critic, batchStates, ...
                        batchReturns, obj.config.valueCoef);
                    criticGrads = obj.clipGradients(criticGrads, obj.config.maxGradNorm);
                    [obj.critic.Learnables, obj.criticOptimizer] = sgdmupdate( ...
                        obj.critic.Learnables, criticGrads, obj.criticOptimizer, obj.config.criticLR);
                    criticLossSum = criticLossSum + extractdata(criticLoss);
                    updateCount = updateCount + 1;
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

    methods (Access = private)
        function clippedGrads = clipGradients(obj, gradients, maxNorm)
            totalNorm = 0;
            for i = 1:height(gradients)
                grad = gradients.Value{i};
                if any(isnan(grad(:))) || any(isinf(grad(:)))
                    gradients.Value{i} = zeros(size(grad), 'like', grad);
                    grad = gradients.Value{i};
                end
                totalNorm = totalNorm + sum(grad(:).^2);
            end
            totalNorm = sqrt(totalNorm);
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

function net = createPPOActorNetwork(config)
    stateSize = config.stateSize;
    numActions = config.numActionConfigs;
    numSubgroups = config.numSubgroups;
    hiddenLayers = config.actorHiddenLayers;

    layers = [
        featureInputLayer(stateSize, 'Name', 'state_input', 'Normalization', 'none')
    ];

    for i = 1:length(hiddenLayers)
        hiddenSize = hiddenLayers(i);
        layers = [layers
            fullyConnectedLayer(hiddenSize, 'Name', sprintf('fc%d', i))
            tanhLayer('Name', sprintf('tanh%d', i))];
    end

    layers = [layers
        fullyConnectedLayer(numActions * numSubgroups, 'Name', 'logits')];

    lgraph = layerGraph(layers);
    net = dlnetwork(lgraph);
    net = initializeWeights(net);
end

function net = createPPOCriticNetwork(config)
    stateSize = config.stateSize;
    hiddenLayers = config.criticHiddenLayers;

    layers = [
        featureInputLayer(stateSize, 'Name', 'state_input', 'Normalization', 'none')
    ];

    for i = 1:length(hiddenLayers)
        hiddenSize = hiddenLayers(i);
        layers = [layers
            fullyConnectedLayer(hiddenSize, 'Name', sprintf('fc%d', i))
            tanhLayer('Name', sprintf('tanh%d', i))];
    end

    layers = [layers
        fullyConnectedLayer(1, 'Name', 'value')];

    lgraph = layerGraph(layers);
    net = dlnetwork(lgraph);
    net = initializeWeights(net);
end

function net = initializeWeights(net)
    learnables = net.Learnables;
    for i = 1:height(learnables)
        paramName = learnables.Parameter{i};
        if strcmp(paramName, 'Weights')
            weights = learnables.Value{i};
            [outputSize, inputSize] = size(weights);
            scale = sqrt(2.0 / inputSize);
            learnables.Value{i} = dlarray(randn(outputSize, inputSize, 'single') * scale);
        elseif strcmp(paramName, 'Bias')
            bias = learnables.Value{i};
            learnables.Value{i} = dlarray(zeros(size(bias), 'single'));
        end
    end
    net.Learnables = learnables;
end

function [loss, gradients] = ppoActorLoss(actorNet, states, actions, oldLogProbs, advantages, ...
    clipEpsilon, entropyCoef, numActions, numSubgroups)
    actions = max(1, min(numActions, round(actions)));
    logits = forward(actorNet, states);
    [logProbs, entropy] = computeCategoricalLogProbs(logits, actions, numActions, numSubgroups);

    ratio = exp(logProbs - oldLogProbs);
    clippedRatio = max(min(ratio, 1 + clipEpsilon), 1 - clipEpsilon);
    surrogate1 = ratio .* advantages;
    surrogate2 = clippedRatio .* advantages;
    actorLoss = -mean(min(surrogate1, surrogate2));

    entropyBonus = mean(entropy);
    loss = actorLoss - entropyCoef * entropyBonus;

    gradients = dlgradient(loss, actorNet.Learnables);
end

function [loss, gradients] = ppoCriticLoss(criticNet, states, returns, valueCoef)
    values = forward(criticNet, states);
    valueLoss = mean((returns - values).^2);
    loss = valueCoef * valueLoss;
    gradients = dlgradient(loss, criticNet.Learnables);
end

function [logProbs, entropy] = computeCategoricalLogProbs(logits, actions, numActions, numSubgroups)
    batchSize = size(logits, 2);
    logits = reshape(logits, [numActions, numSubgroups, batchSize]);
    logProbs = zeros(1, batchSize, 'like', logits);
    entropy = zeros(1, batchSize, 'like', logits);

    for b = 1:batchSize
        for j = 1:numSubgroups
            logitsJ = logits(:, j, b);
            logitsJ = logitsJ - max(logitsJ);
            expLogits = exp(logitsJ);
            probs = expLogits ./ sum(expLogits);

            actionIdx = actions(j, b);
            logProbs(1, b) = logProbs(1, b) + log(probs(actionIdx) + 1e-8);
            entropy(1, b) = entropy(1, b) - sum(probs .* log(probs + 1e-8));
        end
    end
end

function probs = softmaxVector(logits)
    logits = logits - max(logits);
    expLogits = exp(logits);
    probs = expLogits / sum(expLogits);
end

function idx = sampleDiscrete(probs)
    cdf = cumsum(probs);
    r = rand();
    idx = find(r <= cdf, 1, 'first');
    if isempty(idx)
        idx = numel(probs);
    end
end
