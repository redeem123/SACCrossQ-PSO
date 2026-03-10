% Test RRSACPSO SAC Components
% Run this script to verify all components work before full benchmark
%
% Usage: Run from MATLAB command window
%   cd 'C:\Users\PC\Desktop\Copy_of_New Folder - Copy'
%   addpath(genpath('.'))
%   algorithms/rrsacpso/test_rrsacpso_components

clc;
fprintf('╔══════════════════════════════════════════════════════════╗\n');
fprintf('║         RRSACPSO SAC Component Testing Suite              ║\n');
fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

testsPassed = 0;
totalTests = 9;

try
    %% Test 1: Configuration Creation
    fprintf('[Test 1/9] Creating configuration...\n');
    config = RRSACPSO_Config('fast');
    assert(config.stateSize == 15, 'State size mismatch');
    assert(config.actionSize == 9, 'Action size mismatch');
    assert(~config.useCrossQCritic, 'CrossQ critic should be disabled by default');
    assert(~config.useREDQCritic, 'REDQ critic should be disabled by default');
    assert(~config.useAQECritic, 'AQE critic should be disabled');
    assert(~config.useDroQCritic, 'DroQ critic should be disabled');
    assert(~config.useTQCCritic, 'TQC critic should be disabled by default');
    assert(~config.useCriticBatchNorm, 'Critic BatchNorm should be disabled by default');
    assert(~config.useJointCriticBatchForBN, 'Joint critic batches should be disabled by default');
    assert(config.useTargetNetworks, 'Default SAC stack should enable target networks');
    assert(~config.useD2RLBackbone, 'D2RL backbone should be disabled in retained 2-component stack');
    assert(~config.useEmphasizingRecentExperience, 'ERE replay should be disabled');
    assert(~config.useDelayedPolicyUpdates, 'Delayed actor updates should be disabled');
    assert(~config.useSimBaBackbone, 'SimBa backbone should be disabled');
    assert(~config.useResidualCriticDecomposition, 'Residual critic should be disabled');
    assert(config.gradientStepsPerTraining == config.utdRatio, 'UTD/training-step mismatch');
    assert(config.numCritics == 2, 'Default SAC stack should use twin critics');
    fprintf('  ✓ Config created: %dD state, %dD action\n', config.stateSize, config.actionSize);
    testsPassed = testsPassed + 1;

    %% Test 2: Actor Network Creation
    fprintf('\n[Test 2/9] Creating actor network...\n');
    actor = createActorNetwork(config);
    fprintf('  ✓ Actor network created\n');
    fprintf('    - Layers: %d\n', height(actor.Layers));
    fprintf('    - Learnables: %d parameters\n', height(actor.Learnables));
    testsPassed = testsPassed + 1;

    %% Test 3: Critic Network Creation
    fprintf('\n[Test 3/9] Creating critic networks...\n');
    critic1 = createCriticNetwork(config);
    critic2 = createCriticNetwork(config);
    fprintf('  ✓ Two critic networks created\n');
    fprintf('    - Layers: %d each\n', height(critic1.Layers));
    criticOutput = predict(critic1, dlarray(zeros(config.stateSize + config.actionSize, 1, 'single'), 'CB'));
    if config.useTQCCritic
        assert(size(criticOutput, 1) == config.tqcNumQuantiles, 'Critic quantile count mismatch');
    else
        assert(size(criticOutput, 1) == 1, 'Default critic should emit a scalar value');
    end
    testsPassed = testsPassed + 1;

    %% Test 4: Replay Buffer
    fprintf('\n[Test 4/9] Testing replay buffer...\n');
    buffer = ReplayBuffer(1000, config.stateSize, config.actionSize, false, config);
    zeroState = zeros(config.stateSize, 1);
    zeroAction = zeros(config.actionSize, 1);
    buffer.add(zeroState, zeroAction, 1.0, zeroState, 0);
    buffer.add(zeroState, zeroAction, 1.0, zeroState, 0);
    buffer.add(zeroState, zeroAction, 1.0, zeroState, 1);
    assert(buffer.size == 3, 'Buffer size incorrect');
    fprintf('  ✓ Replay buffer working\n');
    fprintf('    - Capacity: %d\n', buffer.capacity);
    fprintf('    - Current size: %d\n', buffer.size);
    if buffer.usePrioritizedReplay
        assert(all(buffer.priorities(1:buffer.size) > 0), 'Replay priorities must be positive');
    end
    if buffer.usePilarReturns
        expectedLongReturn = 1 + config.gamma + config.gamma^2;
        assert(abs(buffer.rewards(1) - 1.0) < 1e-6, 'One-step return incorrect');
        assert(abs(buffer.discounts(1) - config.gamma) < 1e-6, 'One-step discount incorrect');
        assert(abs(buffer.auxRewards(1) - expectedLongReturn) < 1e-4, 'PiLaR long return incorrect');
        assert(abs(buffer.auxDiscounts(1) - config.gamma^3) < 1e-6, 'PiLaR long discount incorrect');
    end
    for i = 1:6
        stateValue = single(i) * ones(config.stateSize, 1, 'single');
        buffer.add(stateValue, zeroAction, double(i), stateValue, 0);
    end
    [statesRecent, ~, ~, ~, ~, ~, recentIndices] = buffer.sample(3, 0.0, 3);
    assert(all(recentIndices >= buffer.size - 2), 'ERE recent-window sampling escaped the newest transitions');
    assert(all(ismember(round(statesRecent(:, 1)), [4, 5, 6])), 'ERE sample did not come from the newest states');
    testsPassed = testsPassed + 1;

    %% Test 5: State Encoder
    fprintf('\n[Test 5/9] Testing state encoder...\n');
    if config.useCrossScaleState
        encoder = CrossScaleStateEncoder(config);
    else
        encoder = SimpleStateEncoder(config);
    end
    for i = 1:5
        testFeatures = randn(9, 1);
        encoder.addIteration(testFeatures);
    end
    state = encoder.encode();
    assert(length(state) == config.stateSize, 'State dimension mismatch');
    fprintf('  ✓ State encoder working\n');
    fprintf('    - Temporal window: %d\n', encoder.temporalWindow);
    fprintf('    - Output dimension: %d\n', length(state));
    testsPassed = testsPassed + 1;

    %% Test 6: Agent Initialization
    fprintf('\n[Test 6/9] Initializing SAC agent...\n');
    agent = RRSACPSO_Agent(config);
    fprintf('  ✓ Agent initialized\n');
    fprintf('    - Actor LR: %.5f\n', config.actorLR);
    fprintf('    - Critic LR: %.5f\n', config.criticLR);
    fprintf('    - Target entropy: %.1f\n', config.targetEntropy);
    fprintf('    - Initial alpha: %.2f\n', config.initAlpha);
    redqConfig = config;
    redqConfig.useCrossQCritic = false;
    redqConfig.useREDQCritic = true;
    redqConfig.useCriticBatchNorm = false;
    redqConfig.useJointCriticBatchForBN = false;
    redqConfig.useTargetNetworks = true;
    redqConfig.useDelayedPolicyUpdates = true;
    redqConfig.redqNumCritics = 4;
    redqConfig.numCritics = redqConfig.redqNumCritics;
    redqConfig.redqTargetSubsetSize = 2;
    redqConfig.redqPolicyUpdateDelay = 2;
    redqConfig.actorUpdateInterval = 2;
    redqConfig.utdRatio = 2;
    redqConfig.gradientStepsPerTraining = 2;
    redqConfig.warmupPeriod = 0;
    redqConfig.batchSize = 32;
    redqAgent = RRSACPSO_Agent(redqConfig);
    assert(numel(redqAgent.critics) == redqConfig.redqNumCritics, 'REDQ critic ensemble size mismatch');
    assert(redqAgent.useTargetNetworks, 'REDQ should require target networks');
    fprintf('    - REDQ critics: %d (subset=%d)\n', numel(redqAgent.critics), redqConfig.redqTargetSubsetSize);
    testsPassed = testsPassed + 1;

    %% Test 7: Action Generation
    fprintf('\n[Test 7/9] Testing action generation...\n');
    testState = randn(config.stateSize, 1);
    action = agent.getAction(testState, true);
    assert(length(action) == 9, 'Action dimension mismatch');
    assert(all(action >= -1 & action <= 1), 'Actions not bounded to [-1,1]');
    fprintf('  ✓ Action generation working\n');
    fprintf('    - Action dimension: %d\n', length(action));
    fprintf('    - Action range: [%.3f, %.3f]\n', min(action), max(action));
    testsPassed = testsPassed + 1;

    %% Test 8: Action to Parameter Conversion
    fprintf('\n[Test 8/9] Testing action to parameter conversion...\n');
    particleParams = convertActionToPerParticleParams(action, config);
    assert(size(particleParams, 1) == 40, 'Particle count mismatch');
    assert(size(particleParams, 2) == 3, 'Parameter count mismatch');
    fprintf('  ✓ Action conversion working\n');
    fprintf('    - Particles: %d\n', size(particleParams, 1));
    fprintf('    - Parameters per particle: %d (w, c1, c2)\n', size(particleParams, 2));
    fprintf('    - w range: [%.3f, %.3f]\n', min(particleParams(:,1)), max(particleParams(:,1)));
    fprintf('    - c1 range: [%.3f, %.3f]\n', min(particleParams(:,2)), max(particleParams(:,2)));
    fprintf('    - c2 range: [%.3f, %.3f]\n', min(particleParams(:,3)), max(particleParams(:,3)));
    testsPassed = testsPassed + 1;

    %% Test 9: Single Training Step
    fprintf('\n[Test 9/9] Testing single SAC training step...\n');
    for i = 1:config.batchSize
        s = randn(config.stateSize, 1);
        a = tanh(randn(config.actionSize, 1));
        r = randn();
        ns = randn(config.stateSize, 1);
        done = double(i == config.batchSize);
        agent.storeTransition(s, a, r, ns, done);
    end
    previousStep = agent.trainingStep;
    losses = agent.train();
    assert(agent.trainingStep == previousStep + 1, 'Training step did not advance');
    assert(all(isfinite([losses.actor, losses.critic, losses.alpha, losses.alphaValue])), ...
        'Training produced non-finite losses');
    if config.useObservationNormalization
        assert(agent.obsNormCount > 0, 'Observation normalizer did not update');
        assert(all(isfinite(agent.obsNormMean)), 'Observation mean contains non-finite values');
        assert(all(isfinite(agent.obsNormM2)), 'Observation variance accumulator contains non-finite values');
    end
    fprintf('  ✓ Training step working\n');
    fprintf('    - Actor loss: %.4f\n', losses.actor);
    fprintf('    - Critic loss: %.4f\n', losses.critic);
    fprintf('    - Alpha: %.4f\n', losses.alphaValue);
    fprintf('    - Obs norm count: %d\n', agent.obsNormCount);
    for i = 1:redqConfig.batchSize
        s = randn(redqConfig.stateSize, 1);
        a = tanh(randn(redqConfig.actionSize, 1));
        r = randn();
        ns = randn(redqConfig.stateSize, 1);
        done = double(i == redqConfig.batchSize);
        redqAgent.storeTransition(s, a, r, ns, done);
    end
    redqPrevStep = redqAgent.trainingStep;
    redqLosses = redqAgent.train();
    assert(redqAgent.trainingStep == redqPrevStep + 1, 'REDQ training step did not advance');
    assert(all(isfinite([redqLosses.actor, redqLosses.critic, redqLosses.alpha, redqLosses.alphaValue])), ...
        'REDQ training produced non-finite losses');
    fprintf('    - REDQ critic loss: %.4f\n', redqLosses.critic);
    testsPassed = testsPassed + 1;

    %% Summary
    fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║                    Test Summary                          ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n\n');
    fprintf('  Tests Passed: %d/%d\n', testsPassed, totalTests);

    if testsPassed == totalTests
        fprintf('  Status: ✓ ALL TESTS PASSED\n');
        fprintf('\n  RRSACPSO SAC stack is ready for testing.\n');
        fprintf('  Run: run_comparison\n\n');
    else
        fprintf('  Status: ⚠ SOME TESTS FAILED\n');
        fprintf('  Please review errors above.\n\n');
    end

catch ME
    fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║                    Test Failed                           ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n\n');
    fprintf('  Error: %s\n', ME.message);
    fprintf('  Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    fprintf('\n  Tests Passed: %d/%d\n\n', testsPassed, totalTests);

    % Display full stack trace
    fprintf('  Stack trace:\n');
    for k = 1:length(ME.stack)
        fprintf('    %d. %s (line %d)\n', k, ME.stack(k).name, ME.stack(k).line);
    end
end
