function ddpgAgent = trainDLToolboxDDPG(ddpgAgent)
    % Train DDPG networks using automatic differentiation (DL Toolbox)
    %
    % Paper Algorithm 1, steps 9–10:
    %   - Update critic by minimizing loss (Eq. 2)
    %   - Update actor via policy gradient (Eq. 3)
    %   - Soft-update target networks (Eq. 1)

    bufferSize = length(ddpgAgent.replayBuffer);
    if bufferSize < ddpgAgent.batchSize
        return;
    end

    % === SAMPLE BATCH (uniform random) ===
    batchIndices = randperm(bufferSize, ddpgAgent.batchSize);
    [states, actions, rewards, nextStates, dones] = extractBatch( ...
        ddpgAgent.replayBuffer, batchIndices, ddpgAgent);

    % Convert to dlarray
    dlStates     = dlarray(states,     'CB');
    dlActions    = dlarray(actions,    'CB');
    dlRewards    = dlarray(rewards,    'CB');
    dlNextStates = dlarray(nextStates, 'CB');
    dlDones      = dlarray(dones,      'CB');

    % === CRITIC UPDATE (Eq. 2) ===
    % Compute target Q: r + γ Q'(s', μ'(s'))
    targetActions      = predict(ddpgAgent.targetActor, dlNextStates);
    targetStateActions = cat(1, dlNextStates, targetActions);
    targetQ = extractdata(predict(ddpgAgent.targetCritic, targetStateActions));
    targetQ = dlarray(targetQ, 'CB');
    yTargets = dlRewards + ddpgAgent.gamma * targetQ .* (1 - dlDones);
    yTargets = max(-100, min(100, yTargets));   % Loose clamp

    [criticLoss, criticGrads] = dlfeval(@criticLossFunction, ...
        ddpgAgent.critic, dlStates, dlActions, yTargets);

    [ddpgAgent.critic, ddpgAgent.criticOptimizer.averageGrad, ...
     ddpgAgent.criticOptimizer.averageSqGrad] = ...
        adamupdate(ddpgAgent.critic, criticGrads, ...
                   ddpgAgent.criticOptimizer.averageGrad, ...
                   ddpgAgent.criticOptimizer.averageSqGrad, ...
                   ddpgAgent.trainStep + 1, ddpgAgent.criticLR, ...
                   ddpgAgent.criticOptimizer.gradDecay, ...
                   ddpgAgent.criticOptimizer.sqGradDecay);

    % === ACTOR UPDATE (Eq. 3) ===
    [actorLoss, actorGrads] = dlfeval(@actorLossFunction, ...
        ddpgAgent.actor, ddpgAgent.critic, dlStates);

    [ddpgAgent.actor, ddpgAgent.actorOptimizer.averageGrad, ...
     ddpgAgent.actorOptimizer.averageSqGrad] = ...
        adamupdate(ddpgAgent.actor, actorGrads, ...
                   ddpgAgent.actorOptimizer.averageGrad, ...
                   ddpgAgent.actorOptimizer.averageSqGrad, ...
                   ddpgAgent.trainStep + 1, ddpgAgent.actorLR, ...
                   ddpgAgent.actorOptimizer.gradDecay, ...
                   ddpgAgent.actorOptimizer.sqGradDecay);

    % === SOFT UPDATE TARGET NETWORKS (Eq. 1) ===
    ddpgAgent.targetActor  = softUpdateNetwork(ddpgAgent.targetActor,  ddpgAgent.actor,  ddpgAgent.tau);
    ddpgAgent.targetCritic = softUpdateNetwork(ddpgAgent.targetCritic, ddpgAgent.critic, ddpgAgent.tau);

    % Store losses for logging
    ddpgAgent.lastActorLoss  = extractdata(gather(actorLoss));
    ddpgAgent.lastCriticLoss = extractdata(gather(criticLoss));
    ddpgAgent.trainStep = ddpgAgent.trainStep + 1;
end

%% Loss functions

function [loss, grads] = criticLossFunction(criticNet, states, actions, targets)
    stateActions = cat(1, states, actions);
    qValues = forward(criticNet, stateActions);
    loss = mean((qValues - targets).^2);
    grads = dlgradient(loss, criticNet.Learnables);
end

function [loss, grads] = actorLossFunction(actorNet, criticNet, states)
    actions = forward(actorNet, states);
    stateActions = cat(1, states, actions);
    qValues = forward(criticNet, stateActions);
    loss = -mean(qValues);
    grads = dlgradient(loss, actorNet.Learnables);
end

%% Helpers

function [states, actions, rewards, nextStates, dones] = extractBatch(buffer, indices, agent)
    bs = length(indices);
    states     = zeros(agent.stateSize,  bs);
    actions    = zeros(agent.actionSize, bs);
    rewards    = zeros(1, bs);
    nextStates = zeros(agent.stateSize,  bs);
    dones      = zeros(1, bs);
    for i = 1:bs
        e = buffer(indices(i));
        states(:,i)     = e.state;
        actions(:,i)    = e.action;
        rewards(i)      = e.reward;
        nextStates(:,i) = e.nextState;
        dones(i)        = e.done;
    end
end

function targetNet = softUpdateNetwork(targetNet, sourceNet, tau)
    tp = targetNet.Learnables;
    sp = sourceNet.Learnables;
    for i = 1:height(tp)
        tp.Value{i} = tau * sp.Value{i} + (1-tau) * tp.Value{i};
    end
    targetNet.Learnables = tp;
end
