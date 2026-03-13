function td3Agent = trainTD3Networks(td3Agent)
    % TD3 Network Training with Prioritized Experience Replay
    % Implements Twin Delayed DDPG algorithm
    %
    % Reference: Fujimoto et al., "Addressing Function Approximation Error in
    %            Actor-Critic Methods", ICML 2018
    %
    % Key TD3 Features:
    %   1. Clipped Double Q-learning: min(Q1, Q2) for target
    %   2. Delayed policy updates: Update actor every d iterations
    %   3. Target policy smoothing: Add noise to target actions
    %
    % Inputs:
    %   td3Agent: TD3 agent structure
    %
    % Outputs:
    %   td3Agent: Updated agent with trained networks

    % Check if enough experiences
    bufferSize = getBufferSize(td3Agent);
    if bufferSize < td3Agent.batchSize
        return;
    end

    useGPU = td3Agent.useGPU && canUseGPU();

    % === SAMPLE BATCH ===
    if td3Agent.usePER
        % Prioritized sampling
        [batch, treeIndices, isWeights] = td3Agent.replayBuffer.sample(td3Agent.batchSize);

        % Extract experiences
        [states, actions, rewards, nextStates, dones] = extractBatch(batch, td3Agent);

        % Convert importance weights to GPU if needed
        if useGPU
            isWeights = gpuArray(isWeights');  % (1 × batchSize)
        else
            isWeights = isWeights';
        end
    else
        % Uniform random sampling
        bufferSize = length(td3Agent.replayBuffer);
        batchIndices = randperm(bufferSize, td3Agent.batchSize);

        [states, actions, rewards, nextStates, dones] = extractUniformBatch(...
            td3Agent.replayBuffer, batchIndices, td3Agent);

        isWeights = ones(1, td3Agent.batchSize);  % Uniform weights
        if useGPU
            isWeights = gpuArray(isWeights);
        end
    end

    % Move to GPU
    if useGPU
        states = gpuArray(states);
        actions = gpuArray(actions);
        rewards = gpuArray(rewards);
        nextStates = gpuArray(nextStates);
        dones = gpuArray(dones);
    end

    % === TD3: TARGET POLICY SMOOTHING (Equation 4 in paper) ===
    % Select action with target actor and add clipped noise
    targetActions = forwardPassPaperActor(td3Agent.targetActor, nextStates);

    % Add Gaussian noise to target actions
    noise = td3Agent.targetNoiseStd * randn(size(targetActions));
    noise = max(-td3Agent.targetNoiseClip, min(td3Agent.targetNoiseClip, noise));

    if useGPU
        noise = gpuArray(noise);
    end

    targetActions = targetActions + noise;
    targetActions = max(-1, min(1, targetActions));  % Clip to action bounds

    % === TD3: CLIPPED DOUBLE Q-LEARNING (Equation 3 in paper) ===
    % Compute target Q-values from both critics and take minimum
    targetQ1 = forwardPassPaperCritic(td3Agent.targetCritic1, nextStates, targetActions);
    targetQ2 = forwardPassPaperCritic(td3Agent.targetCritic2, nextStates, targetActions);

    % Take minimum to reduce overestimation bias
    targetQ = min(targetQ1, targetQ2);

    % Compute TD targets: y = r + γ * min(Q1', Q2') * (1 - done)
    yTargets = rewards + td3Agent.gamma * targetQ .* (1 - dones);

    % === UPDATE CRITIC 1 ===
    [currentQ1, critic1Cache] = forwardPassPaperCriticWithCache(td3Agent.critic1, states, actions);

    % TD error for PER priority update
    tdError1 = currentQ1 - yTargets;

    % Weighted MSE loss (importance sampling weights from PER)
    criticLoss1 = mean(isWeights .* (tdError1.^2));

    % Backpropagation
    td3Agent.critic1 = backwardPassPaperCritic(td3Agent.critic1, critic1Cache, ...
                                               currentQ1, yTargets, td3Agent.criticLR, isWeights);

    % === UPDATE CRITIC 2 ===
    [currentQ2, critic2Cache] = forwardPassPaperCriticWithCache(td3Agent.critic2, states, actions);

    % TD error
    tdError2 = currentQ2 - yTargets;

    % Weighted MSE loss
    criticLoss2 = mean(isWeights .* (tdError2.^2));

    % Backpropagation
    td3Agent.critic2 = backwardPassPaperCritic(td3Agent.critic2, critic2Cache, ...
                                               currentQ2, yTargets, td3Agent.criticLR, isWeights);

    % === UPDATE PRIORITIES IN PER ===
    if td3Agent.usePER
        % Use average TD error from both critics for priority
        avgTdError = (abs(tdError1) + abs(tdError2)) / 2;

        % Move to CPU for buffer update
        if useGPU
            avgTdError = gather(avgTdError);
        end

        td3Agent.replayBuffer.update_priorities(treeIndices, avgTdError);
    end

    % === TD3: DELAYED POLICY UPDATE (Section 4.3 in paper) ===
    % Only update actor and target networks every d iterations
    td3Agent.trainStep = td3Agent.trainStep + 1;

    if mod(td3Agent.trainStep, td3Agent.policyDelay) == 0
        % === UPDATE ACTOR ===
        % Maximize Q1(s, μ(s)) - use only first critic for policy update
        [predictedActions, actorCache] = forwardPassPaperActorWithCache(td3Agent.actor, states);
        [qValues, ~] = forwardPassPaperCriticWithCache(td3Agent.critic1, states, predictedActions);

        % Actor loss: -Q(s, μ(s))
        actorLoss = -mean(qValues);

        % Backpropagation through actor
        td3Agent.actor = backwardPassPaperActor(td3Agent.actor, td3Agent.critic1, ...
                                                actorCache, states, predictedActions, ...
                                                td3Agent.actorLR);

        % === SOFT UPDATE TARGET NETWORKS ===
        % θ' ← τθ + (1-τ)θ'
        td3Agent.targetActor = softUpdateNetwork(td3Agent.actor, td3Agent.targetActor, td3Agent.tau);
        td3Agent.targetCritic1 = softUpdateNetwork(td3Agent.critic1, td3Agent.targetCritic1, td3Agent.tau);
        td3Agent.targetCritic2 = softUpdateNetwork(td3Agent.critic2, td3Agent.targetCritic2, td3Agent.tau);
    end
end

%% ============ HELPER FUNCTIONS ============

function size = getBufferSize(agent)
    % Get current buffer size
    if agent.usePER
        size = agent.replayBuffer.length();
    else
        size = length(agent.replayBuffer);
    end
end

function [states, actions, rewards, nextStates, dones] = extractBatch(batch, agent)
    % Extract batch data from PER sampled experiences
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
    % Extract batch data from standard replay buffer
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

function targetNet = softUpdateNetwork(sourceNet, targetNet, tau)
    % Soft update: target ← τ*source + (1-τ)*target
    fields = fieldnames(sourceNet);

    for i = 1:length(fields)
        field = fields{i};
        if isnumeric(sourceNet.(field))
            targetNet.(field) = tau * sourceNet.(field) + (1 - tau) * targetNet.(field);
        end
    end
end

function actions = forwardPassPaperActor(actor, states)
    % Forward pass through actor network
    z1 = actor.W1 * states + actor.b1;
    h1 = leakyReLU(z1, 0.01);

    z2 = actor.W2 * h1 + actor.b2;
    h2 = leakyReLU(z2, 0.01);

    z3 = actor.W3 * h2 + actor.b3;
    h3 = leakyReLU(z3, 0.01);

    z4 = actor.W4 * h3 + actor.b4;
    actions = tanh(z4);
end

function [actions, cache] = forwardPassPaperActorWithCache(actor, states)
    % Forward pass with caching
    z1 = actor.W1 * states + actor.b1;
    h1 = leakyReLU(z1, 0.01);

    z2 = actor.W2 * h1 + actor.b2;
    h2 = leakyReLU(z2, 0.01);

    z3 = actor.W3 * h2 + actor.b3;
    h3 = leakyReLU(z3, 0.01);

    z4 = actor.W4 * h3 + actor.b4;
    actions = tanh(z4);

    cache.states = states;
    cache.z1 = z1; cache.h1 = h1;
    cache.z2 = z2; cache.h2 = h2;
    cache.z3 = z3; cache.h3 = h3;
    cache.z4 = z4; cache.actions = actions;
end

function qValues = forwardPassPaperCritic(critic, states, actions)
    % Forward pass through critic network
    input = [states; actions];

    z1 = critic.W1 * input + critic.b1;
    h1 = leakyReLU(z1, 0.01);

    z2 = critic.W2 * h1 + critic.b2;
    h2 = leakyReLU(z2, 0.01);

    z3 = critic.W3 * h2 + critic.b3;
    h3 = leakyReLU(z3, 0.01);

    z4 = critic.W4 * h3 + critic.b4;
    h4 = leakyReLU(z4, 0.01);

    z5 = critic.W5 * h4 + critic.b5;
    h5 = leakyReLU(z5, 0.01);

    qValues = critic.W6 * h5 + critic.b6;
end

function [qValues, cache] = forwardPassPaperCriticWithCache(critic, states, actions)
    % Forward pass with caching
    input = [states; actions];

    z1 = critic.W1 * input + critic.b1;
    h1 = leakyReLU(z1, 0.01);

    z2 = critic.W2 * h1 + critic.b2;
    h2 = leakyReLU(z2, 0.01);

    z3 = critic.W3 * h2 + critic.b3;
    h3 = leakyReLU(z3, 0.01);

    z4 = critic.W4 * h3 + critic.b4;
    h4 = leakyReLU(z4, 0.01);

    z5 = critic.W5 * h4 + critic.b5;
    h5 = leakyReLU(z5, 0.01);

    qValues = critic.W6 * h5 + critic.b6;

    cache.input = input;
    cache.z1 = z1; cache.h1 = h1;
    cache.z2 = z2; cache.h2 = h2;
    cache.z3 = z3; cache.h3 = h3;
    cache.z4 = z4; cache.h4 = h4;
    cache.z5 = z5; cache.h5 = h5;
end

function critic = backwardPassPaperCritic(critic, cache, predictions, targets, lr, isWeights)
    % Backpropagation through critic with importance sampling weights
    batchSize = size(predictions, 2);

    % Gradient of weighted MSE loss
    if nargin < 6
        isWeights = ones(1, batchSize);
    end

    dLoss = isWeights .* (predictions - targets) / batchSize;

    % Backprop through layers
    dW6 = dLoss * cache.h5';
    db6 = sum(dLoss, 2);
    dh5 = critic.W6' * dLoss;

    dz5 = dh5 .* leakyReLUGradient(cache.z5, 0.01);
    dW5 = dz5 * cache.h4';
    db5 = sum(dz5, 2);
    dh4 = critic.W5' * dz5;

    dz4 = dh4 .* leakyReLUGradient(cache.z4, 0.01);
    dW4 = dz4 * cache.h3';
    db4 = sum(dz4, 2);
    dh3 = critic.W4' * dz4;

    dz3 = dh3 .* leakyReLUGradient(cache.z3, 0.01);
    dW3 = dz3 * cache.h2';
    db3 = sum(dz3, 2);
    dh2 = critic.W3' * dz3;

    dz2 = dh2 .* leakyReLUGradient(cache.z2, 0.01);
    dW2 = dz2 * cache.h1';
    db2 = sum(dz2, 2);
    dh1 = critic.W2' * dz2;

    dz1 = dh1 .* leakyReLUGradient(cache.z1, 0.01);
    dW1 = dz1 * cache.input';
    db1 = sum(dz1, 2);

    % Gradient clipping
    maxGrad = 10.0;
    dW1 = clipGradient(dW1, maxGrad);
    dW2 = clipGradient(dW2, maxGrad);
    dW3 = clipGradient(dW3, maxGrad);
    dW4 = clipGradient(dW4, maxGrad);
    dW5 = clipGradient(dW5, maxGrad);
    dW6 = clipGradient(dW6, maxGrad);

    % Update weights
    critic.W1 = critic.W1 - lr * dW1;
    critic.b1 = critic.b1 - lr * db1;
    critic.W2 = critic.W2 - lr * dW2;
    critic.b2 = critic.b2 - lr * db2;
    critic.W3 = critic.W3 - lr * dW3;
    critic.b3 = critic.b3 - lr * db3;
    critic.W4 = critic.W4 - lr * dW4;
    critic.b4 = critic.b4 - lr * db4;
    critic.W5 = critic.W5 - lr * dW5;
    critic.b5 = critic.b5 - lr * db5;
    critic.W6 = critic.W6 - lr * dW6;
    critic.b6 = critic.b6 - lr * db6;
end

function actor = backwardPassPaperActor(actor, critic, actorCache, states, actions, lr)
    % Backpropagation through actor using policy gradient
    batchSize = size(states, 2);

    % Get dQ/da from critic
    dQda = computeCriticGradientWrtAction(critic, states, actions);

    % Gradient through tanh
    dtanh = 1 - actorCache.actions.^2;
    dz4 = dQda .* dtanh;
    dz4 = -dz4 / batchSize;  % Negate for gradient ascent

    % Backprop through layers
    dW4 = dz4 * actorCache.h3';
    db4 = sum(dz4, 2);
    dh3 = actor.W4' * dz4;

    dz3 = dh3 .* leakyReLUGradient(actorCache.z3, 0.01);
    dW3 = dz3 * actorCache.h2';
    db3 = sum(dz3, 2);
    dh2 = actor.W3' * dz3;

    dz2 = dh2 .* leakyReLUGradient(actorCache.z2, 0.01);
    dW2 = dz2 * actorCache.h1';
    db2 = sum(dz2, 2);
    dh1 = actor.W2' * dz2;

    dz1 = dh1 .* leakyReLUGradient(actorCache.z1, 0.01);
    dW1 = dz1 * actorCache.states';
    db1 = sum(dz1, 2);

    % Gradient clipping
    maxGrad = 10.0;
    dW1 = clipGradient(dW1, maxGrad);
    dW2 = clipGradient(dW2, maxGrad);
    dW3 = clipGradient(dW3, maxGrad);
    dW4 = clipGradient(dW4, maxGrad);

    % Update weights
    actor.W1 = actor.W1 - lr * dW1;
    actor.b1 = actor.b1 - lr * db1;
    actor.W2 = actor.W2 - lr * dW2;
    actor.b2 = actor.b2 - lr * db2;
    actor.W3 = actor.W3 - lr * dW3;
    actor.b3 = actor.b3 - lr * db3;
    actor.W4 = actor.W4 - lr * dW4;
    actor.b4 = actor.b4 - lr * db4;
end

function dQda = computeCriticGradientWrtAction(critic, states, actions)
    % Compute ∇_a Q(s, a) using finite differences
    epsilon = 1e-4;
    batchSize = size(states, 2);
    actionSize = size(actions, 1);
    dQda = zeros(actionSize, batchSize);

    for i = 1:actionSize
        actionsPlus = actions;
        actionsPlus(i, :) = actionsPlus(i, :) + epsilon;
        qPlus = forwardPassPaperCritic(critic, states, actionsPlus);
        qCurrent = forwardPassPaperCritic(critic, states, actions);
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
