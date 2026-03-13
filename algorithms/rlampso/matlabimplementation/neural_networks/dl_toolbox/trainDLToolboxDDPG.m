function ddpgAgent = trainDLToolboxDDPG(ddpgAgent)
    % Train DDPG networks using automatic differentiation (DL Toolbox)
    %
    % Replaces trainPaperExactNetworks.m with modern dlgradient-based training
    % Uses dlfeval for automatic differentiation - no manual backprop!
    %
    % Key improvements:
    %   - Automatic gradient computation with dlgradient
    %   - Adam optimizer (better than SGD)
    %   - Efficient GPU utilization
    %   - Cleaner, more maintainable code
    %
    % Inputs:
    %   ddpgAgent: DDPG agent with dlnetwork networks
    %
    % Outputs:
    %   ddpgAgent: Updated agent with trained networks

    % Check if enough experiences
    bufferSize = getBufferSize(ddpgAgent);
    if bufferSize < ddpgAgent.batchSize
        return;
    end

    useGPU = ddpgAgent.useGPU && canUseGPU();

    % === SAMPLE BATCH ===
    if ddpgAgent.usePER
        % Prioritized sampling
        [batch, treeIndices, isWeights] = ddpgAgent.replayBuffer.sample(ddpgAgent.batchSize);
        [states, actions, rewards, nextStates, dones] = extractBatchPER(batch, ddpgAgent);
        isWeights = isWeights';  % (1 × batchSize)
    else
        % Uniform random sampling
        bufferSize = length(ddpgAgent.replayBuffer);
        batchIndices = randperm(bufferSize, ddpgAgent.batchSize);
        [states, actions, rewards, nextStates, dones] = extractBatchUniform(...
            ddpgAgent.replayBuffer, batchIndices, ddpgAgent);
        isWeights = ones(1, ddpgAgent.batchSize);
    end

    % Convert to dlarray format
    dlStates = dlarray(states, 'CB');          % Channel-Batch
    dlActions = dlarray(actions, 'CB');
    dlRewards = dlarray(rewards, 'CB');
    dlNextStates = dlarray(nextStates, 'CB');
    dlDones = dlarray(dones, 'CB');
    dlIsWeights = dlarray(isWeights, 'CB');

    % Move to GPU if available
    if useGPU
        dlStates = gpuArray(dlStates);
        dlActions = gpuArray(dlActions);
        dlRewards = gpuArray(dlRewards);
        dlNextStates = gpuArray(dlNextStates);
        dlDones = gpuArray(dlDones);
        dlIsWeights = gpuArray(dlIsWeights);
    end

    % === LEARNING RATES ===
    % Use constant learning rates (RLAM approach: no warmup needed with conservative LR)
    currentActorLR = ddpgAgent.actorLR;
    currentCriticLR = ddpgAgent.criticLR;

    % === PER-FUNCTION EPISODE COUNTER (TRANSFER LEARNING FIX) ===
    % Use functionEpisodeCount instead of global trainStep for clamping
    % This prevents Q-value explosion when moving between functions
    if ~isfield(ddpgAgent, 'functionEpisodeCount')
        ddpgAgent.functionEpisodeCount = 0;
    end

    % === CRITIC UPDATE (with automatic differentiation) ===

    % Compute target Q-values (no gradients needed here)
    targetActions = predict(ddpgAgent.targetActor, dlNextStates);
    targetStateActions = cat(1, dlNextStates, targetActions);

    % CRITICAL: Stop gradients from flowing through target Q-values
    % This prevents destabilization and gradient explosion
    targetQ = extractdata(predict(ddpgAgent.targetCritic, targetStateActions));
    targetQ = dlarray(targetQ, 'CB');

    % Compute TD targets with stopped gradients
    yTargets = dlRewards + ddpgAgent.gamma * targetQ .* (1 - dlDones);

    % Q-value clamping relaxed with RLAM conservative learning rates
    % With 100× smaller learning rates, Q-value explosion is prevented naturally
    % RLAM-OPENSOURCE doesn't clamp - they rely on conservative LR
    clampRange = 100;  % Loose clamping (far exceeds typical Q-values)

    yTargets = max(-clampRange, min(clampRange, yTargets));

    % Compute critic gradients using dlfeval
    [criticLoss, criticGrads] = dlfeval(@criticLossFunction, ...
        ddpgAgent.critic, dlStates, dlActions, yTargets, dlIsWeights);

    % No gradient clipping (RLAM approach: let Adam optimizer handle scaling)

    % Update critic network with Adam optimizer
    [ddpgAgent.critic, ddpgAgent.criticOptimizer.averageGrad, ...
     ddpgAgent.criticOptimizer.averageSqGrad] = ...
        adamupdate(ddpgAgent.critic, criticGrads, ...
                   ddpgAgent.criticOptimizer.averageGrad, ...
                   ddpgAgent.criticOptimizer.averageSqGrad, ...
                   ddpgAgent.trainStep + 1, currentCriticLR, ...
                   ddpgAgent.criticOptimizer.gradDecay, ...
                   ddpgAgent.criticOptimizer.sqGradDecay);

    % === ACTOR UPDATE (with automatic differentiation) ===

    % Compute actor gradients using dlfeval
    [actorLoss, actorGrads] = dlfeval(@actorLossFunction, ...
        ddpgAgent.actor, ddpgAgent.critic, dlStates);

    % No gradient clipping (RLAM approach: let Adam optimizer handle scaling)

    % Update actor network with Adam optimizer
    [ddpgAgent.actor, ddpgAgent.actorOptimizer.averageGrad, ...
     ddpgAgent.actorOptimizer.averageSqGrad] = ...
        adamupdate(ddpgAgent.actor, actorGrads, ...
                   ddpgAgent.actorOptimizer.averageGrad, ...
                   ddpgAgent.actorOptimizer.averageSqGrad, ...
                   ddpgAgent.trainStep + 1, currentActorLR, ...
                   ddpgAgent.actorOptimizer.gradDecay, ...
                   ddpgAgent.actorOptimizer.sqGradDecay);

    % === UPDATE PRIORITIES (if using PER) ===
    if ddpgAgent.usePER
        % Compute TD errors for priority updates
        currentStateActions = cat(1, dlStates, dlActions);
        currentQ = predict(ddpgAgent.critic, currentStateActions);
        tdErrors = abs(extractdata(gather(yTargets - currentQ)));
        ddpgAgent.replayBuffer.update(treeIndices, tdErrors);
    end

    % === SOFT UPDATE TARGET NETWORKS ===
    ddpgAgent.targetActor = softUpdateNetwork(ddpgAgent.targetActor, ...
                                             ddpgAgent.actor, ddpgAgent.tau);
    ddpgAgent.targetCritic = softUpdateNetwork(ddpgAgent.targetCritic, ...
                                              ddpgAgent.critic, ddpgAgent.tau);

    % === STORE LOSSES FOR LOGGING ===
    ddpgAgent.lastActorLoss = extractdata(gather(actorLoss));
    ddpgAgent.lastCriticLoss = extractdata(gather(criticLoss));

    % Increment training step
    ddpgAgent.trainStep = ddpgAgent.trainStep + 1;
end

%% Loss Functions (for automatic differentiation)

function [loss, grads] = criticLossFunction(criticNet, states, actions, targets, isWeights)
    % Critic loss with importance sampling weights and Q-value regularization
    stateActions = cat(1, states, actions);
    qValues = forward(criticNet, stateActions);
    tdErrors = qValues - targets;

    % Ensure isWeights is properly shaped for broadcasting (1 × batchSize)
    isWeights = reshape(isWeights, 1, []);

    % Compute weighted TD errors
    weightedErrors = tdErrors.^2 .* isWeights;

    % Clamp to prevent numeric overflow (more lenient now with target clamping)
    weightedErrors = max(0, min(1e4, weightedErrors));

    % Q-value regularization (OPTIONAL with conservative learning rates)
    % RLAM-OPENSOURCE doesn't use regularization - relies on conservative LR instead
    % With 100× smaller learning rates, regularization is unnecessary
    qRegularization = 0;  % Disabled (RLAM approach: conservative LR instead of regularization)

    loss = mean(weightedErrors) + qRegularization;
    grads = dlgradient(loss, criticNet.Learnables);
end

function [loss, grads] = actorLossFunction(actorNet, criticNet, states)
    % Actor loss: maximize Q(s, actor(s))
    actions = forward(actorNet, states);
    stateActions = cat(1, states, actions);
    qValues = forward(criticNet, stateActions);
    loss = -mean(qValues);  % Negative because we want to maximize Q
    grads = dlgradient(loss, actorNet.Learnables);
end

%% Helper Functions

function bufferSize = getBufferSize(agent)
    if agent.usePER
        bufferSize = agent.replayBuffer.size();
    else
        bufferSize = length(agent.replayBuffer);
    end
end

function [states, actions, rewards, nextStates, dones] = extractBatchPER(batch, agent)
    batchSize = length(batch);
    states = zeros(agent.stateSize, batchSize);
    actions = zeros(agent.actionSize, batchSize);
    rewards = zeros(1, batchSize);
    nextStates = zeros(agent.stateSize, batchSize);
    dones = zeros(1, batchSize);

    for i = 1:batchSize
        states(:, i) = batch{i}.state;
        actions(:, i) = batch{i}.action;
        rewards(i) = batch{i}.reward;
        nextStates(:, i) = batch{i}.nextState;
        dones(i) = batch{i}.done;
    end
end

function [states, actions, rewards, nextStates, dones] = extractBatchUniform(buffer, indices, agent)
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

function targetNet = softUpdateNetwork(targetNet, sourceNet, tau)
    % Soft update: targetParams = tau * sourceParams + (1-tau) * targetParams
    targetParams = targetNet.Learnables;
    sourceParams = sourceNet.Learnables;

    for i = 1:height(targetParams)
        targetParams.Value{i} = tau * sourceParams.Value{i} + ...
                               (1 - tau) * targetParams.Value{i};
    end

    targetNet.Learnables = targetParams;
end
