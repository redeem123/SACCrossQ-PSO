function loraAgent = trainLoRAFineTune(loraAgent)
    % Train only LoRA parameters while keeping base weights frozen
    % Modified training that updates only LoRA_A and LoRA_B matrices
    %
    % This function replaces the standard network training during fine-tuning
    % It ensures base weights remain frozen and only LoRA adapters are trained

    % Check if enough experiences
    bufferSize = getBufferSize(loraAgent);
    if bufferSize < loraAgent.batchSize
        return;
    end

    useGPU = loraAgent.useGPU && canUseGPU();

    % === SAMPLE BATCH ===
    [states, actions, rewards, nextStates, dones, isWeights] = sampleBatch(loraAgent, useGPU);

    % === COMPUTE TARGETS ===
    % Use LoRA-augmented forward pass
    targetActions = forwardPassLoRAActor(loraAgent.targetActor, nextStates);

    % Add target policy smoothing if TD3
    if isfield(loraAgent, 'targetNoiseStd')
        noise = loraAgent.targetNoiseStd * randn(size(targetActions));
        noise = max(-loraAgent.targetNoiseClip, min(loraAgent.targetNoiseClip, noise));
        if useGPU, noise = gpuArray(noise); end
        targetActions = max(-1, min(1, targetActions + noise));
    end

    % Compute target Q-values
    if isfield(loraAgent, 'targetCritic1')
        % TD3: Use minimum of twin critics
        targetQ1 = forwardPassLoRACritic(loraAgent.targetCritic1, nextStates, targetActions);
        targetQ2 = forwardPassLoRACritic(loraAgent.targetCritic2, nextStates, targetActions);
        targetQ = min(targetQ1, targetQ2);
    else
        % DDPG: Single critic
        targetQ = forwardPassLoRACritic(loraAgent.targetCritic, nextStates, targetActions);
    end

    yTargets = rewards + loraAgent.gamma * targetQ .* (1 - dones);

    % === UPDATE CRITICS (LoRA only) ===
    if isfield(loraAgent, 'critic1')
        % Update critic1
        [currentQ1, critic1Cache] = forwardPassLoRACriticWithCache(loraAgent.critic1, states, actions);
        tdError1 = currentQ1 - yTargets;
        loraAgent.critic1 = backwardPassLoRACritic(loraAgent.critic1, critic1Cache, ...
                                                   currentQ1, yTargets, loraAgent.criticLR, isWeights);

        % Update critic2
        [currentQ2, critic2Cache] = forwardPassLoRACriticWithCache(loraAgent.critic2, states, actions);
        tdError2 = currentQ2 - yTargets;
        loraAgent.critic2 = backwardPassLoRACritic(loraAgent.critic2, critic2Cache, ...
                                                   currentQ2, yTargets, loraAgent.criticLR, isWeights);
    else
        % Single critic
        [currentQ, criticCache] = forwardPassLoRACriticWithCache(loraAgent.critic, states, actions);
        loraAgent.critic = backwardPassLoRACritic(loraAgent.critic, criticCache, ...
                                                  currentQ, yTargets, loraAgent.criticLR, isWeights);
    end

    % === UPDATE ACTOR (LoRA only) - Delayed if TD3 ===
    loraAgent.trainStep = loraAgent.trainStep + 1;

    if ~isfield(loraAgent, 'policyDelay') || mod(loraAgent.trainStep, loraAgent.policyDelay) == 0
        % Forward pass
        [predictedActions, actorCache] = forwardPassLoRAActorWithCache(loraAgent.actor, states);

        % Get Q-values (use first critic for policy update)
        if isfield(loraAgent, 'critic1')
            [qValues, ~] = forwardPassLoRACriticWithCache(loraAgent.critic1, states, predictedActions);
        else
            [qValues, ~] = forwardPassLoRACriticWithCache(loraAgent.critic, states, predictedActions);
        end

        % Update actor LoRA parameters
        loraAgent.actor = backwardPassLoRAActor(loraAgent.actor, loraAgent.critic1, ...
                                               actorCache, states, predictedActions, ...
                                               loraAgent.actorLR);

        % Soft update targets
        loraAgent = softUpdateLoRATargets(loraAgent);
    end

    % === RESTORE FROZEN WEIGHTS ===
    % Ensure base weights haven't changed (safety check)
    loraAgent.actor = restoreFrozenWeights(loraAgent.actor, loraAgent.frozenWeights.actor);

    if isfield(loraAgent, 'critic1')
        loraAgent.critic1 = restoreFrozenWeights(loraAgent.critic1, loraAgent.frozenWeights.critic1);
        loraAgent.critic2 = restoreFrozenWeights(loraAgent.critic2, loraAgent.frozenWeights.critic2);
    else
        loraAgent.critic = restoreFrozenWeights(loraAgent.critic, loraAgent.frozenWeights.critic);
    end
end

%% ============ LORA-AUGMENTED FORWARD PASSES ============

function actions = forwardPassLoRAActor(actor, states)
    % Forward pass with LoRA: output = W*x + (B*A)*x
    z1 = applyLoRALayer(actor, 'W1', states) + actor.b1;
    h1 = leakyReLU(z1, 0.01);

    z2 = applyLoRALayer(actor, 'W2', h1) + actor.b2;
    h2 = leakyReLU(z2, 0.01);

    z3 = applyLoRALayer(actor, 'W3', h2) + actor.b3;
    h3 = leakyReLU(z3, 0.01);

    z4 = applyLoRALayer(actor, 'W4', h3) + actor.b4;
    actions = tanh(z4);
end

function [actions, cache] = forwardPassLoRAActorWithCache(actor, states)
    % Forward pass with caching for backprop
    z1 = applyLoRALayer(actor, 'W1', states) + actor.b1;
    h1 = leakyReLU(z1, 0.01);

    z2 = applyLoRALayer(actor, 'W2', h1) + actor.b2;
    h2 = leakyReLU(z2, 0.01);

    z3 = applyLoRALayer(actor, 'W3', h2) + actor.b3;
    h3 = leakyReLU(z3, 0.01);

    z4 = applyLoRALayer(actor, 'W4', h3) + actor.b4;
    actions = tanh(z4);

    cache.states = states;
    cache.z1 = z1; cache.h1 = h1;
    cache.z2 = z2; cache.h2 = h2;
    cache.z3 = z3; cache.h3 = h3;
    cache.z4 = z4; cache.actions = actions;
end

function qValues = forwardPassLoRACritic(critic, states, actions)
    % Forward pass through critic with LoRA
    input = [states; actions];

    z1 = applyLoRALayer(critic, 'W1', input) + critic.b1;
    h1 = leakyReLU(z1, 0.01);

    z2 = applyLoRALayer(critic, 'W2', h1) + critic.b2;
    h2 = leakyReLU(z2, 0.01);

    z3 = applyLoRALayer(critic, 'W3', h2) + critic.b3;
    h3 = leakyReLU(z3, 0.01);

    z4 = applyLoRALayer(critic, 'W4', h3) + critic.b4;
    h4 = leakyReLU(z4, 0.01);

    z5 = applyLoRALayer(critic, 'W5', h4) + critic.b5;
    h5 = leakyReLU(z5, 0.01);

    qValues = applyLoRALayer(critic, 'W6', h5) + critic.b6;
end

function [qValues, cache] = forwardPassLoRACriticWithCache(critic, states, actions)
    % Forward pass with caching
    input = [states; actions];

    z1 = applyLoRALayer(critic, 'W1', input) + critic.b1;
    h1 = leakyReLU(z1, 0.01);

    z2 = applyLoRALayer(critic, 'W2', h1) + critic.b2;
    h2 = leakyReLU(z2, 0.01);

    z3 = applyLoRALayer(critic, 'W3', h2) + critic.b3;
    h3 = leakyReLU(z3, 0.01);

    z4 = applyLoRALayer(critic, 'W4', h3) + critic.b4;
    h4 = leakyReLU(z4, 0.01);

    z5 = applyLoRALayer(critic, 'W5', h4) + critic.b5;
    h5 = leakyReLU(z5, 0.01);

    qValues = applyLoRALayer(critic, 'W6', h5) + critic.b6;

    cache.input = input;
    cache.z1 = z1; cache.h1 = h1;
    cache.z2 = z2; cache.h2 = h2;
    cache.z3 = z3; cache.h3 = h3;
    cache.z4 = z4; cache.h4 = h4;
    cache.z5 = z5; cache.h5 = h5;
end

function output = applyLoRALayer(net, weightName, input)
    % Apply LoRA-augmented weight: output = (W + B*A) * input
    %
    % W: frozen base weight
    % B: LoRA_B matrix (d × r)
    % A: LoRA_A matrix (r × k)
    % scale: alpha / r

    % Base weight multiplication
    W = net.(weightName);
    output = W * input;

    % LoRA adaptation: add (B * A) * input with scaling
    loraA_name = [weightName, '_LoRA_A'];
    loraB_name = [weightName, '_LoRA_B'];
    loraScale_name = [weightName, '_LoRA_scale'];

    if isfield(net, loraA_name) && isfield(net, loraB_name)
        A = net.(loraA_name);
        B = net.(loraB_name);
        scale = net.(loraScale_name);

        % Efficient computation: B * (A * input)
        loraOutput = B * (A * input);
        output = output + scale * loraOutput;
    end
end

%% ============ LORA-SPECIFIC BACKPROPAGATION ============

function critic = backwardPassLoRACritic(critic, cache, predictions, targets, lr, isWeights)
    % Backprop through critic - update ONLY LoRA parameters
    batchSize = size(predictions, 2);

    if nargin < 6
        isWeights = ones(1, batchSize);
    end

    dLoss = isWeights .* (predictions - targets) / batchSize;

    % Backprop through layer 6
    critic = backpropLoRALayer(critic, 'W6', dLoss, cache.h5, lr);
    dh5 = critic.W6' * dLoss;

    % Continue backprop (only update LoRA, not base weights)
    dz5 = dh5 .* leakyReLUGradient(cache.z5, 0.01);
    critic = backpropLoRALayer(critic, 'W5', dz5, cache.h4, lr);
    dh4 = critic.W5' * dz5;

    dz4 = dh4 .* leakyReLUGradient(cache.z4, 0.01);
    critic = backpropLoRALayer(critic, 'W4', dz4, cache.h3, lr);
    dh3 = critic.W4' * dz4;

    dz3 = dh3 .* leakyReLUGradient(cache.z3, 0.01);
    critic = backpropLoRALayer(critic, 'W3', dz3, cache.h2, lr);
    dh2 = critic.W3' * dz3;

    dz2 = dh2 .* leakyReLUGradient(cache.z2, 0.01);
    critic = backpropLoRALayer(critic, 'W2', dz2, cache.h1, lr);
    dh1 = critic.W2' * dz2;

    dz1 = dh1 .* leakyReLUGradient(cache.z1, 0.01);
    critic = backpropLoRALayer(critic, 'W1', dz1, cache.input, lr);
end

function actor = backwardPassLoRAActor(actor, critic, actorCache, states, actions, lr)
    % Backprop through actor - update ONLY LoRA parameters
    batchSize = size(states, 2);

    % Get dQ/da
    dQda = computeCriticGradientWrtActionLoRA(critic, states, actions);

    % Gradient through tanh
    dtanh = 1 - actorCache.actions.^2;
    dz4 = dQda .* dtanh;
    dz4 = -dz4 / batchSize;

    % Backprop through layers (LoRA only)
    actor = backpropLoRALayer(actor, 'W4', dz4, actorCache.h3, lr);
    dh3 = actor.W4' * dz4;

    dz3 = dh3 .* leakyReLUGradient(actorCache.z3, 0.01);
    actor = backpropLoRALayer(actor, 'W3', dz3, actorCache.h2, lr);
    dh2 = actor.W3' * dz3;

    dz2 = dh2 .* leakyReLUGradient(actorCache.z2, 0.01);
    actor = backpropLoRALayer(actor, 'W2', dz2, actorCache.h1, lr);
    dh1 = actor.W2' * dz2;

    dz1 = dh1 .* leakyReLUGradient(actorCache.z1, 0.01);
    actor = backpropLoRALayer(actor, 'W1', dz1, actorCache.states, lr);
end

function net = backpropLoRALayer(net, weightName, dOutput, hInput, lr)
    % Update LoRA matrices A and B for a single layer
    % dOutput: gradient flowing back
    % hInput: activations from previous layer

    loraA_name = [weightName, '_LoRA_A'];
    loraB_name = [weightName, '_LoRA_B'];
    loraScale_name = [weightName, '_LoRA_scale'];

    if ~isfield(net, loraA_name)
        return;  % No LoRA for this layer
    end

    scale = net.(loraScale_name);
    A = net.(loraA_name);
    B = net.(loraB_name);

    % Gradients for LoRA matrices
    % Forward: output = W*x + scale * B * (A * x)
    % dL/dB = scale * dOutput * (A * x)^T
    % dL/dA = scale * B^T * dOutput * x^T

    Ax = A * hInput;  % (r × batchSize)

    % Gradient for B: dL/dB = scale * dOutput * (A*x)^T
    dB = scale * dOutput * Ax';

    % Gradient for A: dL/dA = scale * B^T * dOutput * x^T
    dA = scale * (B' * dOutput) * hInput';

    % Gradient clipping
    maxGrad = 10.0;
    dA = clipGradient(dA, maxGrad);
    dB = clipGradient(dB, maxGrad);

    % Update LoRA matrices (leave base weight W frozen)
    net.(loraA_name) = A - lr * dA;
    net.(loraB_name) = B - lr * dB;
end

%% ============ HELPER FUNCTIONS ============

function net = restoreFrozenWeights(net, frozenWeights)
    % Ensure frozen base weights haven't changed
    fields = fieldnames(frozenWeights);
    for i = 1:length(fields)
        field = fields{i};
        if isfield(net, field)
            net.(field) = frozenWeights.(field);
        end
    end
end

function agent = softUpdateLoRATargets(agent)
    % Soft update target networks (including LoRA parameters)
    tau = agent.tau;

    agent.targetActor = softUpdateNetworkLoRA(agent.actor, agent.targetActor, tau);

    if isfield(agent, 'targetCritic1')
        agent.targetCritic1 = softUpdateNetworkLoRA(agent.critic1, agent.targetCritic1, tau);
        agent.targetCritic2 = softUpdateNetworkLoRA(agent.critic2, agent.targetCritic2, tau);
    else
        agent.targetCritic = softUpdateNetworkLoRA(agent.critic, agent.targetCritic, tau);
    end
end

function targetNet = softUpdateNetworkLoRA(sourceNet, targetNet, tau)
    % Soft update including LoRA parameters
    fields = fieldnames(sourceNet);

    for i = 1:length(fields)
        field = fields{i};
        if isnumeric(sourceNet.(field))
            targetNet.(field) = tau * sourceNet.(field) + (1 - tau) * targetNet.(field);
        end
    end
end

function [states, actions, rewards, nextStates, dones, isWeights] = sampleBatch(agent, useGPU)
    % Sample batch from replay buffer
    if agent.usePER
        [batch, ~, isWeights] = agent.replayBuffer.sample(agent.batchSize);
        [states, actions, rewards, nextStates, dones] = extractBatch(batch, agent);
        if useGPU, isWeights = gpuArray(isWeights'); end
    else
        bufferSize = length(agent.replayBuffer);
        indices = randperm(bufferSize, agent.batchSize);
        [states, actions, rewards, nextStates, dones] = extractUniformBatch(agent.replayBuffer, indices, agent);
        isWeights = ones(1, agent.batchSize);
        if useGPU, isWeights = gpuArray(isWeights); end
    end

    if useGPU
        states = gpuArray(states);
        actions = gpuArray(actions);
        rewards = gpuArray(rewards);
        nextStates = gpuArray(nextStates);
        dones = gpuArray(dones);
    end
end

function size = getBufferSize(agent)
    if agent.usePER
        size = agent.replayBuffer.length();
    else
        size = length(agent.replayBuffer);
    end
end

function [states, actions, rewards, nextStates, dones] = extractBatch(batch, agent)
    batchSize = length(batch);
    states = zeros(agent.stateSize, batchSize);
    actions = zeros(agent.actionSize, batchSize);
    rewards = zeros(1, batchSize);
    nextStates = zeros(agent.stateSize, batchSize);
    dones = zeros(1, batchSize);

    for i = 1:batchSize
        exp = batch{i};
        states(:, i) = exp.state;
        actions(:, i) = exp.action;
        rewards(i) = exp.reward;
        nextStates(:, i) = exp.nextState;
        dones(i) = exp.done;
    end
end

function [states, actions, rewards, nextStates, dones] = extractUniformBatch(buffer, indices, agent)
    batchSize = length(indices);
    states = zeros(agent.stateSize, batchSize);
    actions = zeros(agent.actionSize, batchSize);
    rewards = zeros(1, batchSize);
    nextStates = zeros(agent.stateSize, batchSize);
    dones = zeros(1, batchSize);

    for i = 1:batchSize
        exp = buffer(indices(i));
        states(:, i) = exp.state;
        actions(:, i) = exp.action;
        rewards(i) = exp.reward;
        nextStates(:, i) = exp.nextState;
        dones(i) = exp.done;
    end
end

function dQda = computeCriticGradientWrtActionLoRA(critic, states, actions)
    epsilon = 1e-4;
    batchSize = size(states, 2);
    actionSize = size(actions, 1);
    dQda = zeros(actionSize, batchSize);

    for i = 1:actionSize
        actionsPlus = actions;
        actionsPlus(i, :) = actionsPlus(i, :) + epsilon;
        qPlus = forwardPassLoRACritic(critic, states, actionsPlus);
        qCurrent = forwardPassLoRACritic(critic, states, actions);
        dQda(i, :) = (qPlus - qCurrent) / epsilon;
    end
end

function y = leakyReLU(x, alpha)
    y = max(alpha * x, x);
end

function dy = leakyReLUGradient(x, alpha)
    dy = ones(size(x));
    dy(x < 0) = alpha;
end

function gradClipped = clipGradient(grad, maxNorm)
    gradNorm = norm(grad(:));
    if gradNorm > maxNorm
        gradClipped = grad * (maxNorm / gradNorm);
    else
        gradClipped = grad;
    end
end
