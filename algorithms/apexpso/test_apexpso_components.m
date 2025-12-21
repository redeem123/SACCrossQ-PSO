% Test APEX-PSO Components
% Run this script to verify all components work before full benchmark
%
% Usage: Run from MATLAB command window
%   cd 'C:\Users\PC\Desktop\Copy_of_New Folder - Copy'
%   addpath(genpath('.'))
%   algorithms/apexpso/test_apexpso_components

clc;
fprintf('╔══════════════════════════════════════════════════════════╗\n');
fprintf('║         APEX-PSO Component Testing Suite                ║\n');
fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

testsPassed = 0;
totalTests = 8;

try
    %% Test 1: Configuration Creation
    fprintf('[Test 1/8] Creating configuration...\n');
    config = APEXPSO_Config('fast');
    assert(config.stateSize == 45, 'State size mismatch');
    assert(config.actionSize == 120, 'Action size mismatch');
    fprintf('  ✓ Config created: %dD state, %dD action\n', config.stateSize, config.actionSize);
    testsPassed = testsPassed + 1;

    %% Test 2: Actor Network Creation
    fprintf('\n[Test 2/8] Creating actor network...\n');
    actor = createActorNetwork(config);
    fprintf('  ✓ Actor network created\n');
    fprintf('    - Layers: %d\n', height(actor.Layers));
    fprintf('    - Learnables: %d parameters\n', height(actor.Learnables));
    testsPassed = testsPassed + 1;

    %% Test 3: Critic Network Creation
    fprintf('\n[Test 3/8] Creating critic networks...\n');
    critic1 = createCriticNetwork(config);
    critic2 = createCriticNetwork(config);
    fprintf('  ✓ Two critic networks created\n');
    fprintf('    - Layers: %d each\n', height(critic1.Layers));
    testsPassed = testsPassed + 1;

    %% Test 4: Replay Buffer
    fprintf('\n[Test 4/8] Testing replay buffer...\n');
    buffer = ReplayBuffer(1000, config.stateSize, config.actionSize, false);
    buffer.add(randn(45,1), randn(120,1), 1.0, randn(45,1), 0);
    buffer.add(randn(45,1), randn(120,1), -0.5, randn(45,1), 0);
    assert(buffer.size == 2, 'Buffer size incorrect');
    fprintf('  ✓ Replay buffer working\n');
    fprintf('    - Capacity: %d\n', buffer.capacity);
    fprintf('    - Current size: %d\n', buffer.size);
    testsPassed = testsPassed + 1;

    %% Test 5: State Encoder
    fprintf('\n[Test 5/8] Testing state encoder...\n');
    encoder = SimpleStateEncoder(config);
    for i = 1:5
        testFeatures = randn(9, 1);
        encoder.addIteration(testFeatures);
    end
    state = encoder.encode();
    assert(length(state) == 45, 'State dimension mismatch');
    fprintf('  ✓ State encoder working\n');
    fprintf('    - Temporal window: %d\n', encoder.temporalWindow);
    fprintf('    - Output dimension: %d\n', length(state));
    testsPassed = testsPassed + 1;

    %% Test 6: Agent Initialization
    fprintf('\n[Test 6/8] Initializing SAC agent...\n');
    agent = APEXPSO_Agent(config);
    fprintf('  ✓ Agent initialized\n');
    fprintf('    - Actor LR: %.5f\n', config.actorLR);
    fprintf('    - Critic LR: %.5f\n', config.criticLR);
    fprintf('    - Target entropy: %.1f\n', config.targetEntropy);
    fprintf('    - Initial alpha: %.2f\n', config.initAlpha);
    testsPassed = testsPassed + 1;

    %% Test 7: Action Generation
    fprintf('\n[Test 7/8] Testing action generation...\n');
    testState = randn(45, 1);
    action = agent.getAction(testState, true);
    assert(length(action) == 120, 'Action dimension mismatch');
    assert(all(action >= -1 & action <= 1), 'Actions not bounded to [-1,1]');
    fprintf('  ✓ Action generation working\n');
    fprintf('    - Action dimension: %d\n', length(action));
    fprintf('    - Action range: [%.3f, %.3f]\n', min(action), max(action));
    testsPassed = testsPassed + 1;

    %% Test 8: Action to Parameter Conversion
    fprintf('\n[Test 8/8] Testing action to parameter conversion...\n');
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

    %% Summary
    fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║                    Test Summary                          ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n\n');
    fprintf('  Tests Passed: %d/%d\n', testsPassed, totalTests);

    if testsPassed == totalTests
        fprintf('  Status: ✓ ALL TESTS PASSED\n');
        fprintf('\n  🎉 APEX-PSO is ready for testing!\n');
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
