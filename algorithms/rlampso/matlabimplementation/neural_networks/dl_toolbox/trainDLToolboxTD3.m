function td3Agent = trainDLToolboxTD3(td3Agent)
    % Train TD3 networks using automatic differentiation (DL Toolbox)
    %
    % Replaces trainTD3Networks.m with modern dlgradient-based training
    % Implements Twin Delayed DDPG with automatic differentiation
    %
    % TD3 Key Features:
    %   1. Twin Q-networks: Use min(Q1, Q2) for target (reduces overestimation)
    %   2. Delayed policy updates: Update actor every d critic updates
    %   3. Target policy smoothing: Add noise to target actions
    %
    % Inputs:
    %   td3Agent: TD3 agent with twin dlnetwork critics
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
        [states, actions, rewards, nextStates, dones] = extractBatchPER(batch, td3Agent);
        isWeights = isWeights';  % (1 × batchSize)
    else
        % Uniform random sampling
        bufferSize = length(td3Agent.replayBuffer);
        batchIndices = randperm(bufferSize, td3Agent.batchSize);
        [states, actions, rewards, nextStates, dones] = extractBatchUniform(...
            td3Agent.replayBuffer, batchIndices, td3Agent);
        isWeights = ones(1, td3Agent.batchSize);
    end

    % Convert to dlarray format
    dlStates = dlarray(states, 'CB');
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

    % === LEARNING RATE WARMUP ===
    % Linear warmup from initialLR to targetLR over warmupSteps
    if td3Agent.trainStep < td3Agent.warmupSteps
        warmupProgress = td3Agent.trainStep / td3Agent.warmupSteps;
        currentActorLR = td3Agent.initialActorLR + (td3Agent.actorLR - td3Agent.initialActorLR) * warmupProgress;
        currentCriticLR = td3Agent.initialCriticLR + (td3Agent.criticLR - td3Agent.initialCriticLR) * warmupProgress;
    else
        currentActorLR = td3Agent.actorLR;
        currentCriticLR = td3Agent.criticLR;
    end

    % === TD3: TARGET POLICY SMOOTHING ===
    % Add clipped noise to target actions for exploration
    targetActions = predict(td3Agent.targetActor, dlNextStates);

    % Generate smoothing noise
    noiseArray = td3Agent.targetNoiseStd * randn(size(targetActions));
    if useGPU
        noiseArray = gpuArray(noiseArray);
    end
    noise = dlarray(noiseArray, 'CB');

    % Clip noise to range
    noise = max(-td3Agent.targetNoiseClip, min(td3Agent.targetNoiseClip, noise));

    % Apply noise and clip to action bounds
    targetActions = targetActions + noise;
    targetActions = max(-1, min(1, targetActions));  % Action bounds [-1, 1]

    % === TD3: CLIPPED DOUBLE Q-LEARNING ===
    % Compute target Q-values from BOTH critics and take minimum
    targetStateActions = cat(1, dlNextStates, targetActions);

    % CRITICAL: Stop gradients from flowing through target Q-values
    % This prevents destabilization and gradient explosion
    targetQ1 = extractdata(predict(td3Agent.targetCritic1, targetStateActions));
    targetQ2 = extractdata(predict(td3Agent.targetCritic2, targetStateActions));
    targetQ1 = dlarray(targetQ1, 'CB');
    targetQ2 = dlarray(targetQ2, 'CB');

    targetQ = min(targetQ1, targetQ2);  % Take minimum to reduce overestimation

    % Compute TD targets with stopped gradients
    % Clamp targets to reasonable range to prevent explosion
    yTargets = dlRewards + td3Agent.gamma * targetQ .* (1 - dlDones);

    % Debug: Check for extreme values (first few training steps)
    if td3Agent.trainStep < 5
        maxTargetQ = max(extractdata(abs(targetQ)), [], 'all');
        maxYTarget = max(extractdata(abs(yTargets)), [], 'all');
        if maxTargetQ > 1000 || maxYTarget > 1000
            fprintf('  [DEBUG] Step %d: maxTargetQ=%.2e, maxYTarget=%.2e\n', ...
                    td3Agent.trainStep, maxTargetQ, maxYTarget);
        end
    end

    % More aggressive clamping for early training stability
    if td3Agent.trainStep < 1000
        yTargets = max(-100, min(100, yTargets));  % Tighter bounds early
    else
        yTargets = max(-1000, min(1000, yTargets));  % Relax later
    end

    % === UPDATE BOTH CRITICS ===

    % Update Critic 1
    [critic1Loss, critic1Grads] = dlfeval(@criticLossFunction, ...
        td3Agent.critic1, dlStates, dlActions, yTargets, dlIsWeights);

    % Clip gradients to prevent explosion
    critic1Grads = dlupdate(@(g) min(max(g, -1), 1), critic1Grads);

    [td3Agent.critic1, td3Agent.critic1Optimizer.averageGrad, ...
     td3Agent.critic1Optimizer.averageSqGrad] = ...
        adamupdate(td3Agent.critic1, critic1Grads, ...
                   td3Agent.critic1Optimizer.averageGrad, ...
                   td3Agent.critic1Optimizer.averageSqGrad, ...
                   td3Agent.trainStep + 1, currentCriticLR, ...
                   td3Agent.critic1Optimizer.gradDecay, ...
                   td3Agent.critic1Optimizer.sqGradDecay);

    % Update Critic 2
    [critic2Loss, critic2Grads] = dlfeval(@criticLossFunction, ...
        td3Agent.critic2, dlStates, dlActions, yTargets, dlIsWeights);

    % Clip gradients to prevent explosion
    critic2Grads = dlupdate(@(g) min(max(g, -1), 1), critic2Grads);

    [td3Agent.critic2, td3Agent.critic2Optimizer.averageGrad, ...
     td3Agent.critic2Optimizer.averageSqGrad] = ...
        adamupdate(td3Agent.critic2, critic2Grads, ...
                   td3Agent.critic2Optimizer.averageGrad, ...
                   td3Agent.critic2Optimizer.averageSqGrad, ...
                   td3Agent.trainStep + 1, currentCriticLR, ...
                   td3Agent.critic2Optimizer.gradDecay, ...
                   td3Agent.critic2Optimizer.sqGradDecay);

    % === STORE CRITIC LOSSES FOR LOGGING ===
    % Average of twin critics
    td3Agent.lastCriticLoss = (extractdata(gather(critic1Loss)) + extractdata(gather(critic2Loss))) / 2;

    % === TD3: DELAYED POLICY UPDATE ===
    % Only update actor every d critic updates
    if mod(td3Agent.trainStep, td3Agent.policyDelay) == 0

        % Compute actor gradients (only use Critic1 for policy gradient)
        [actorLoss, actorGrads] = dlfeval(@actorLossFunction, ...
            td3Agent.actor, td3Agent.critic1, dlStates);

        % Clip gradients to prevent explosion
        actorGrads = dlupdate(@(g) min(max(g, -1), 1), actorGrads);

        % Update actor network (with warmup LR)
        [td3Agent.actor, td3Agent.actorOptimizer.averageGrad, ...
         td3Agent.actorOptimizer.averageSqGrad] = ...
            adamupdate(td3Agent.actor, actorGrads, ...
                       td3Agent.actorOptimizer.averageGrad, ...
                       td3Agent.actorOptimizer.averageSqGrad, ...
                       td3Agent.trainStep + 1, currentActorLR, ...
                       td3Agent.actorOptimizer.gradDecay, ...
                       td3Agent.actorOptimizer.sqGradDecay);

        % Store actor loss for logging
        td3Agent.lastActorLoss = extractdata(gather(actorLoss));

        % Soft update target networks (only when actor is updated)
        td3Agent.targetActor = softUpdateNetwork(td3Agent.targetActor, ...
                                                 td3Agent.actor, td3Agent.tau);
        td3Agent.targetCritic1 = softUpdateNetwork(td3Agent.targetCritic1, ...
                                                   td3Agent.critic1, td3Agent.tau);
        td3Agent.targetCritic2 = softUpdateNetwork(td3Agent.targetCritic2, ...
                                                   td3Agent.critic2, td3Agent.tau);
    end

    % === UPDATE PRIORITIES (if using PER) ===
    if td3Agent.usePER
        % Compute TD errors using minimum of twin critics for priorities
        currentStateActions = cat(1, dlStates, dlActions);
        currentQ1 = predict(td3Agent.critic1, currentStateActions);
        currentQ2 = predict(td3Agent.critic2, currentStateActions);
        currentQ = min(currentQ1, currentQ2);
        tdErrors = abs(extractdata(gather(yTargets - currentQ)));
        td3Agent.replayBuffer.update_priorities(treeIndices, tdErrors);
    end

    % Increment training step
    td3Agent.trainStep = td3Agent.trainStep + 1;
end

%% Loss Functions (for automatic differentiation)

function [loss, grads] = criticLossFunction(criticNet, states, actions, targets, isWeights)
    % Critic loss with importance sampling weights
    stateActions = cat(1, states, actions);
    qValues = forward(criticNet, stateActions);
    tdErrors = qValues - targets;

    % Ensure isWeights is properly shaped for broadcasting (1 × batchSize)
    isWeights = reshape(isWeights, 1, []);

    % Compute weighted TD errors
    weightedErrors = tdErrors.^2 .* isWeights;

    % Clamp to prevent numeric overflow (more lenient now with target clamping)
    weightedErrors = max(0, min(1e4, weightedErrors));

    loss = mean(weightedErrors);
    grads = dlgradient(loss, criticNet.Learnables);
end

function [loss, grads] = actorLossFunction(actorNet, criticNet, states)
    % Actor loss: maximize Q(s, actor(s)) with action regularization
    actions = forward(actorNet, states);
    stateActions = cat(1, states, actions);
    qValues = forward(criticNet, stateActions);

    % CRITICAL FIX: Add L2 regularization to prevent action saturation
    % Penalize large actions to encourage the actor to use moderate actions
    % With conversion formula w=a(1)*0.4+0.7, action=0 gives optimal params (w=0.7, c1=c2=1.5)
    % Combined with lower LR (1e-5/1e-4), moderate reg should prevent saturation
    actionRegWeight = 0.1;  % Moderate penalty (0.01→0.5→5.0→0.1)
    actionL2 = mean(actions.^2, 'all');  % L2 norm of actions

    % Total loss: maximize Q while minimizing action magnitude
    loss = -mean(qValues) + actionRegWeight * actionL2;

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
