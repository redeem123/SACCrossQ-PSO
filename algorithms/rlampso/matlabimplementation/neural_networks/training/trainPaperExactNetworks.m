function rlamAgent = trainPaperExactNetworks(rlamAgent)
    % Proper DDPG training following paper equations and DDPG algorithm
    % Reference: Paper page 10-11 (Equations 2-4) + DDPG paper (Lillicrap et al. 2016)
    % GPU-accelerated version

    if length(rlamAgent.replayBuffer) < rlamAgent.batchSize
        return;  % Not enough experiences yet
    end

    % Check GPU availability
    useGPU = rlamAgent.useGPU && canUseGPU();

    % Sample random minibatch from replay buffer
    batchSize = rlamAgent.batchSize;
    bufferSize = length(rlamAgent.replayBuffer);
    batchIndices = randperm(bufferSize, batchSize);

    % Extract batch data
    states = zeros(rlamAgent.stateSize, batchSize);
    actions = zeros(rlamAgent.actionSize, batchSize);
    rewards = zeros(1, batchSize);
    nextStates = zeros(rlamAgent.stateSize, batchSize);
    dones = zeros(1, batchSize);

    for i = 1:batchSize
        exp = rlamAgent.replayBuffer(batchIndices(i));
        states(:, i) = exp.state;
        actions(:, i) = exp.action;
        rewards(i) = exp.reward;
        nextStates(:, i) = exp.nextState;
        dones(i) = exp.done;
    end

    % Move batch to GPU if available
    if useGPU
        states = gpuArray(states);
        actions = gpuArray(actions);
        rewards = gpuArray(rewards);
        nextStates = gpuArray(nextStates);
        dones = gpuArray(dones);
    end

    %% CRITIC UPDATE (Equation 3 from paper)
    % L(θ^Q) = (r(s_t, a_t) + γQ'(s_{t+1}, μ'(s_{t+1}|θ^μ')|θ^Q') - Q(s_t, a_t|θ^Q))^2

    % Get target actions from target actor: μ'(s_{t+1})
    targetActions = forwardPassPaperActor(rlamAgent.targetActor, nextStates);

    % Get target Q-values: Q'(s_{t+1}, μ'(s_{t+1}))
    targetQValues = forwardPassPaperCritic(rlamAgent.targetCritic, nextStates, targetActions);

    % Compute TD targets: y_i = r_i + γ * Q'(s_{i+1}, μ'(s_{i+1}))
    yTargets = rewards + rlamAgent.gamma * targetQValues .* (1 - dones);

    % Forward pass through critic
    [currentQValues, criticCache] = forwardPassPaperCriticWithCache(rlamAgent.critic, states, actions);

    % Compute critic loss
    criticLoss = mean((currentQValues - yTargets).^2);

    % Backward pass through critic
    rlamAgent.critic = backwardPassPaperCritic(rlamAgent.critic, criticCache, currentQValues, yTargets, rlamAgent.criticLR);

    %% ACTOR UPDATE (Equation 4 from paper)
    % ∇_{θ^μ} J ≈ ∇_a Q(s, a|θ^Q)|_{s=s_i, a=μ(s_i)} ∇_{θ^μ} μ(s|θ^μ)|_{s=s_i}

    % Forward pass through actor
    [predictedActions, actorCache] = forwardPassPaperActorWithCache(rlamAgent.actor, states);

    % Forward pass through critic to get Q(s, μ(s))
    [qValues, ~] = forwardPassPaperCriticWithCache(rlamAgent.critic, states, predictedActions);

    % Actor loss: -Q(s, μ(s)) (negative because we want to maximize Q)
    actorLoss = -mean(qValues);

    % Backward pass through actor (using policy gradient)
    rlamAgent.actor = backwardPassPaperActor(rlamAgent.actor, rlamAgent.critic, actorCache, states, predictedActions, rlamAgent.actorLR);

    %% SOFT UPDATE TARGET NETWORKS (Equation 2 from paper)
    % θ' ← τθ + (1-τ)θ'
    rlamAgent = softUpdatePaperTargets(rlamAgent);

    % Track training step
    rlamAgent.trainStep = rlamAgent.trainStep + 1;
end

%% Forward Pass Functions

function actions = forwardPassPaperActor(actor, states)
    % Forward pass through actor network (Figure 4 architecture)
    % Input → 64 → 64 → 64 → tanh → Output

    % Layer 1
    z1 = actor.W1 * states + actor.b1;
    h1 = leakyReLU(z1, 0.01);

    % Layer 2
    z2 = actor.W2 * h1 + actor.b2;
    h2 = leakyReLU(z2, 0.01);

    % Layer 3
    z3 = actor.W3 * h2 + actor.b3;
    h3 = leakyReLU(z3, 0.01);

    % Output layer with tanh
    z4 = actor.W4 * h3 + actor.b4;
    actions = tanh(z4);  % Output in [-1, 1]
end

function [actions, cache] = forwardPassPaperActorWithCache(actor, states)
    % Forward pass with caching for backpropagation

    % Layer 1
    z1 = actor.W1 * states + actor.b1;
    h1 = leakyReLU(z1, 0.01);

    % Layer 2
    z2 = actor.W2 * h1 + actor.b2;
    h2 = leakyReLU(z2, 0.01);

    % Layer 3
    z3 = actor.W3 * h2 + actor.b3;
    h3 = leakyReLU(z3, 0.01);

    % Output layer
    z4 = actor.W4 * h3 + actor.b4;
    actions = tanh(z4);

    % Cache for backprop
    cache.states = states;
    cache.z1 = z1; cache.h1 = h1;
    cache.z2 = z2; cache.h2 = h2;
    cache.z3 = z3; cache.h3 = h3;
    cache.z4 = z4; cache.actions = actions;
end

function qValues = forwardPassPaperCritic(critic, states, actions)
    % Forward pass through critic network (Figure 5 architecture)
    % [State; Action] → 64 → 64 → 32 → 32 → 16 → 1

    input = [states; actions];

    % Layer 1
    z1 = critic.W1 * input + critic.b1;
    h1 = leakyReLU(z1, 0.01);

    % Layer 2
    z2 = critic.W2 * h1 + critic.b2;
    h2 = leakyReLU(z2, 0.01);

    % Layer 3
    z3 = critic.W3 * h2 + critic.b3;
    h3 = leakyReLU(z3, 0.01);

    % Layer 4
    z4 = critic.W4 * h3 + critic.b4;
    h4 = leakyReLU(z4, 0.01);

    % Layer 5
    z5 = critic.W5 * h4 + critic.b5;
    h5 = leakyReLU(z5, 0.01);

    % Output layer (linear)
    qValues = critic.W6 * h5 + critic.b6;
end

function [qValues, cache] = forwardPassPaperCriticWithCache(critic, states, actions)
    % Forward pass with caching

    input = [states; actions];

    % Layer 1
    z1 = critic.W1 * input + critic.b1;
    h1 = leakyReLU(z1, 0.01);

    % Layer 2
    z2 = critic.W2 * h1 + critic.b2;
    h2 = leakyReLU(z2, 0.01);

    % Layer 3
    z3 = critic.W3 * h2 + critic.b3;
    h3 = leakyReLU(z3, 0.01);

    % Layer 4
    z4 = critic.W4 * h3 + critic.b4;
    h4 = leakyReLU(z4, 0.01);

    % Layer 5
    z5 = critic.W5 * h4 + critic.b5;
    h5 = leakyReLU(z5, 0.01);

    % Output layer
    qValues = critic.W6 * h5 + critic.b6;

    % Cache
    cache.input = input;
    cache.z1 = z1; cache.h1 = h1;
    cache.z2 = z2; cache.h2 = h2;
    cache.z3 = z3; cache.h3 = h3;
    cache.z4 = z4; cache.h4 = h4;
    cache.z5 = z5; cache.h5 = h5;
end

%% Backward Pass Functions

function critic = backwardPassPaperCritic(critic, cache, predictions, targets, lr)
    % Backpropagation through critic network

    batchSize = size(predictions, 2);

    % Gradient of MSE loss
    dLoss = (predictions - targets) / batchSize;  % (1 × N)

    % Backprop through layer 6 (output)
    dW6 = dLoss * cache.h5';
    db6 = sum(dLoss, 2);
    dh5 = critic.W6' * dLoss;

    % Backprop through layer 5
    dz5 = dh5 .* leakyReLUGradient(cache.z5, 0.01);
    dW5 = dz5 * cache.h4';
    db5 = sum(dz5, 2);
    dh4 = critic.W5' * dz5;

    % Backprop through layer 4
    dz4 = dh4 .* leakyReLUGradient(cache.z4, 0.01);
    dW4 = dz4 * cache.h3';
    db4 = sum(dz4, 2);
    dh3 = critic.W4' * dz4;

    % Backprop through layer 3
    dz3 = dh3 .* leakyReLUGradient(cache.z3, 0.01);
    dW3 = dz3 * cache.h2';
    db3 = sum(dz3, 2);
    dh2 = critic.W3' * dz3;

    % Backprop through layer 2
    dz2 = dh2 .* leakyReLUGradient(cache.z2, 0.01);
    dW2 = dz2 * cache.h1';
    db2 = sum(dz2, 2);
    dh1 = critic.W2' * dz2;

    % Backprop through layer 1
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

    % Update weights (simple SGD)
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

    % Gradient through tanh: d_tanh = 1 - tanh(z)^2
    dtanh = 1 - actorCache.actions.^2;
    dz4 = dQda .* dtanh;  % Chain rule

    % Negate for gradient ascent (maximize Q)
    dz4 = -dz4 / batchSize;

    % Backprop through layer 4
    dW4 = dz4 * actorCache.h3';
    db4 = sum(dz4, 2);
    dh3 = actor.W4' * dz4;

    % Backprop through layer 3
    dz3 = dh3 .* leakyReLUGradient(actorCache.z3, 0.01);
    dW3 = dz3 * actorCache.h2';
    db3 = sum(dz3, 2);
    dh2 = actor.W3' * dz3;

    % Backprop through layer 2
    dz2 = dh2 .* leakyReLUGradient(actorCache.z2, 0.01);
    dW2 = dz2 * actorCache.h1';
    db2 = sum(dz2, 2);
    dh1 = actor.W2' * dz2;

    % Backprop through layer 1
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
    % Compute ∇_a Q(s, a) using finite differences (more stable)

    epsilon = 1e-4;
    batchSize = size(states, 2);
    actionSize = size(actions, 1);

    dQda = zeros(actionSize, batchSize);

    for i = 1:actionSize
        % Perturb action
        actionsPlus = actions;
        actionsPlus(i, :) = actionsPlus(i, :) + epsilon;

        % Compute Q values
        qPlus = forwardPassPaperCritic(critic, states, actionsPlus);
        qCurrent = forwardPassPaperCritic(critic, states, actions);

        % Numerical gradient
        dQda(i, :) = (qPlus - qCurrent) / epsilon;
    end
end

%% Helper Functions

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
