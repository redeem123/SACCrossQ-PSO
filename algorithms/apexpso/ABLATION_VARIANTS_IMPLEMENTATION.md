# APEX-PSO Ablation Study: Variant Implementation Guide

**Document Version:** 3.0 Final (Merged Baseline)
**Date:** 2025-11-03
**Total Variants:** 12 configurations (1 unified baseline + 11 ablations)

---

## Quick Reference: All Variants

### Summary Table

| ID | Category | State/Action Dim | Description | Expected Fitness (S2) |
|----|----------|------------------|-------------|-----------------------|
| **APEX-Baseline** | **Unified Baseline** | 45D state, 120D action | Full APEX-PSO (all components enabled) | **~1300** |
| TE-V1-Instant | Temporal | 9D state | Ablation: No temporal memory | ~1800 (+38%) |
| PP-V1-Global | Control | 3D action | Ablation: Global control only | ~1700 (+31%) |
| PP-V2-Sub5 | Control | 15D action | Ablation: 5-subgroup control | ~1550 (+19%) |
| MR-V1-Fitness | Reward | - | Ablation: Fitness reward only | ~1650 (+27%) |
| MR-V2-NoDiversity | Reward | - | Ablation: No diversity reward | ~1600 (+23%) |
| MR-V3-NoCuriosity | Reward | - | Ablation: No curiosity reward | ~1320 (+2%) |
| MR-V4-NoConvergence | Reward | - | Ablation: No convergence reward | ~1350 (+4%) |
| BN-V1-None | Normalization | - | Ablation: No BatchNorm | ~1500 (+15%) |
| BN-V2-ActorOnly | Normalization | - | Ablation: Actor BatchNorm only | ~1380 (+6%) |
| BN-V3-CriticOnly | Normalization | - | Ablation: Critic BatchNorm only | ~1370 (+5%) |
| BN-V6-WithTargets | Normalization | - | Ablation: Add target networks | ~1350 (+4%) |

**Total Experiments:** 12 variants × 4 scenarios × 30 runs = **1,440 experiments**

**Note:** All ablation variants are compared against the single unified **APEX-Baseline**. Percentages show performance degradation vs baseline.

---

## 0. UNIFIED BASELINE: APEX-PSO

### 0.1 APEX-Baseline (Full Implementation)

**This is the single unified baseline used for ALL ablation comparisons.**

**Full Configuration:**
- **State encoding:** 45D temporal (Current + WeightedAvg + Mean + Std + Trend, 5-iter memory)
- **Action space:** 120D per-particle control (w, c1, c2 for each of 40 particles)
- **Reward function:** Multi-objective (Fitness: 1.0, Diversity: 0.2, Convergence: 0.1, Curiosity: 0.05)
- **Normalization:** BatchNorm in actor and critic
- **Target networks:** No (CrossQ approach)

**Implementation:**
```matlab
% APEX-Baseline combines all default components:

% 1. Temporal State Encoding (SimpleStateEncoder.m)
function state = encode(obj, features)
    current_features = features;  % 9D
    obj.updateBuffer(current_features);

    weighted_avg = obj.computeWeightedAverage();
    temporal_mean = obj.computeMean();
    temporal_std = obj.computeStd();
    temporal_trend = obj.computeTrend();

    state = [current_features; weighted_avg; temporal_mean;
             temporal_std; temporal_trend];  % 45D
end

% 2. Per-Particle Control (convertActionToPerParticleParams.m)
function [w_vec, c1_vec, c2_vec] = convertAction(action, num_particles)
    % action is [40×3] matrix, each row: [w_i, c1_i, c2_i]
    w_vec = action(:, 1);    % 40×1
    c1_vec = action(:, 2);   % 40×1
    c2_vec = action(:, 3);   % 40×1

    % Scale to ranges
    w_vec = 0.4 + 0.5 * w_vec;         % [0.4, 0.9]
    c1_vec = 0.5 + 2.0 * c1_vec;       % [0.5, 2.5]
    c2_vec = 0.5 + 2.0 * c2_vec;       % [0.5, 2.5]
end

% 3. Multi-Objective Reward (calculateMultiObjectiveReward.m)
function total_reward = calculateReward(features, state_encoder, config)
    reward_fitness = 1.0 * calculateFitnessReward(features);
    reward_diversity = 0.2 * calculateDiversityReward(features);
    reward_convergence = 0.1 * calculateConvergenceReward(features);
    reward_curiosity = 0.05 * calculateCuriosityReward(state_encoder);

    total_reward = reward_fitness + reward_diversity +
                   reward_convergence + reward_curiosity;
end

% 4. BatchNorm Networks (createActorNetwork.m, createCriticNetwork.m)
layers = [
    fullyConnectedLayer(128, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')  % BatchNorm enabled
    reluLayer('Name', 'relu1')
    % ... (3 layers total)
];

% 5. No Target Networks (APEXPSO_Agent.m updateCritic())
% Use current networks for Q-target computation (CrossQ)
nextActions = obj.actor.forward(nextStates);
nextQ1 = obj.critic1.forward([nextStates, nextActions]);
nextQ2 = obj.critic2.forward([nextStates, nextActions]);
targetQ = rewards + gamma * min(nextQ1, nextQ2);
```

**Network Configurations:**
- Actor input: 45D (state)
- Actor output: 240D (120D mean + 120D log_std)
- Critic input: 165D (45D state + 120D action)
- Target entropy: -120

**Files Used:**
- `SimpleStateEncoder.m` (temporal encoding)
- `convertActionToPerParticleParams.m` (per-particle control)
- `calculateMultiObjectiveReward.m` (multi-objective reward)
- `createActorNetwork.m` (with BatchNorm)
- `createCriticNetwork.m` (with BatchNorm)
- `APEXPSO_Agent.m` (without target networks)

**Training:**
- Episodes: 150
- Iterations per episode: 600
- Warmup: 50 iterations
- Batch size: 512 (GPU) / 256 (CPU)
- Learning rates: 3e-4 (actor, critic, alpha)

**Expected Performance:**
- S1 (empty): ~800 fitness
- S2 (medium): ~1300 fitness
- S3 (high): ~1600 fitness
- S4 (extreme): ~2000 fitness

---

## 1. TEMPORAL ENCODING VARIANTS (1 ablation)

### 1.1 TE-V1-Instant (Ablation: Remove Temporal Encoding)

**Configuration:**
- State dimension: 9D
- Components: Current features only
- Memory window: None

**Implementation:**
```matlab
% File: SimpleStateEncoder.m
% Create variant: SimpleStateEncoder_Instant.m

classdef SimpleStateEncoder_Instant < handle
    % Simplified encoder without temporal memory

    methods
        function state = encode(obj, features)
            % Return current features directly (9D)
            state = features;  % No temporal aggregation
        end
    end
end
```

**Network Configurations:**
- Actor input: 9D (modify createActorNetwork.m line 26)
- Critic input: 9D (state) + 120D (action) = 129D

**Files to Modify:**
1. Create `algorithms/apexpso/SimpleStateEncoder_Instant.m`
2. Modify `algorithms/apexpso/createActorNetwork.m`:
   ```matlab
   % Line 26
   inputSize = 9;  % Changed from 45
   ```
3. Modify `algorithms/apexpso/createCriticNetwork.m`:
   ```matlab
   % Line 26
   stateInputSize = 9;  % Changed from 45
   actionInputSize = 120;
   totalInputSize = 129;  % Changed from 165
   ```

---

## 2. PER-PARTICLE CONTROL VARIANTS (2 ablations)

### 2.1 PP-V1-Global (Ablation: Global Control)

**Configuration:**
- Action dimension: 3D
- Control scheme: Global (w, c1, c2) for all particles
- Particles per group: 40 (1 group)

**Implementation:**
```matlab
% File: convertActionToPerParticleParams_Global.m

function [w_vec, c1_vec, c2_vec] = convertAction(action, num_particles)
    % action is [1×3] vector: [w, c1, c2]

    % Broadcast to all particles
    w_vec = repmat(action(1), num_particles, 1);    % 40×1
    c1_vec = repmat(action(2), num_particles, 1);   % 40×1
    c2_vec = repmat(action(3), num_particles, 1);   % 40×1

    % Scale to appropriate ranges
    w_vec = 0.4 + 0.5 * w_vec;         % [0.4, 0.9]
    c1_vec = 0.5 + 2.0 * c1_vec;       % [0.5, 2.5]
    c2_vec = 0.5 + 2.0 * c2_vec;       % [0.5, 2.5]
end
```

**Network Configurations:**
- Actor output: 6D (3D mean + 3D log_std)
- Critic input: 45D (state) + 3D (action) = 48D
- Target entropy: -3

**Files to Modify:**
1. Create `algorithms/apexpso/convertActionToPerParticleParams_Global.m`
2. Modify `algorithms/apexpso/createActorNetwork.m`:
   ```matlab
   % Line 79
   outputSize = 6;  % Changed from 240 (3 params × 2 for mean+logstd)
   ```
3. Modify `algorithms/apexpso/createCriticNetwork.m`:
   ```matlab
   % Line 26
   actionInputSize = 3;  % Changed from 120
   totalInputSize = 48;  % 45 + 3
   ```
4. Modify `algorithms/apexpso/APEXPSO_Agent.m`:
   ```matlab
   % Line 160
   targetEntropy = -3;  % Changed from -120
   ```

---

### 2.2 PP-V2-Sub5 (Ablation: 5-Subgroup Control)

**Configuration:**
- Action dimension: 15D
- Control scheme: 5 subgroups × (w, c1, c2)
- Particles per group: 8 (5 groups)

**Implementation:**
```matlab
% File: convertActionToPerParticleParams_Sub5.m

function [w_vec, c1_vec, c2_vec] = convertAction(action, num_particles)
    % action is [5×3] matrix
    % Each row controls 8 particles

    % Repeat each subgroup's parameters for 8 particles
    particles_per_group = num_particles / 5;  % 40/5 = 8

    w_vec = repelem(action(:, 1), particles_per_group);    % 40×1
    c1_vec = repelem(action(:, 2), particles_per_group);   % 40×1
    c2_vec = repelem(action(:, 3), particles_per_group);   % 40×1

    % Scale to appropriate ranges
    w_vec = 0.4 + 0.5 * w_vec;         % [0.4, 0.9]
    c1_vec = 0.5 + 2.0 * c1_vec;       % [0.5, 2.5]
    c2_vec = 0.5 + 2.0 * c2_vec;       % [0.5, 2.5]
end
```

**Network Configurations:**
- Actor output: 30D (15D mean + 15D log_std)
- Critic input: 45D (state) + 15D (action) = 60D
- Target entropy: -15

**Files to Modify:**
1. Create `algorithms/apexpso/convertActionToPerParticleParams_Sub5.m`
2. Modify `algorithms/apexpso/createActorNetwork.m`:
   ```matlab
   % Line 79
   outputSize = 30;  % Changed from 240 (15 params × 2)
   ```
3. Modify `algorithms/apexpso/createCriticNetwork.m`:
   ```matlab
   % Line 26
   actionInputSize = 15;  % Changed from 120
   totalInputSize = 60;   % 45 + 15
   ```
4. Modify `algorithms/apexpso/APEXPSO_Agent.m`:
   ```matlab
   % Line 160
   targetEntropy = -15;  % Changed from -120
   ```

---

## 3. MULTI-OBJECTIVE REWARD VARIANTS (4 ablations)

### 3.1 MR-V1-Fitness (Ablation: Fitness Reward Only)

**Configuration:**
- Fitness weight: 1.0
- Diversity weight: 0.0 ← **disabled**
- Convergence weight: 0.0 ← **disabled**
- Curiosity weight: 0.0 ← **disabled**
- Total weight sum: 1.0

**Implementation:**
```matlab
% File: calculateMultiObjectiveReward_FitnessOnly.m

function total_reward = calculateReward(features, state_encoder, config)
    % Only fitness reward
    reward_fitness = 1.0 * calculateFitnessReward(features);

    total_reward = reward_fitness;  % No other components
end
```

---

### 3.2 MR-V2-NoDiversity (Ablation: Remove Diversity Reward)

**Configuration:**
- Fitness weight: 1.0
- Diversity weight: 0.0 ← **disabled**
- Convergence weight: 0.1
- Curiosity weight: 0.05
- Total weight sum: 1.15

**Implementation:**
```matlab
% File: calculateMultiObjectiveReward_NoDiversity.m

function total_reward = calculateReward(features, state_encoder, config)
    % All components except diversity
    reward_fitness = 1.0 * calculateFitnessReward(features);
    reward_convergence = 0.1 * calculateConvergenceReward(features);
    reward_curiosity = 0.05 * calculateCuriosityReward(state_encoder);

    total_reward = reward_fitness + reward_convergence + reward_curiosity;
end
```

---

### 3.3 MR-V3-NoCuriosity (Ablation: Remove Curiosity Reward)

**Configuration:**
- Fitness weight: 1.0
- Diversity weight: 0.2
- Convergence weight: 0.1
- Curiosity weight: 0.0 ← **disabled**
- Total weight sum: 1.3

**Implementation:**
```matlab
% File: calculateMultiObjectiveReward_NoCuriosity.m

function total_reward = calculateReward(features, state_encoder, config)
    % All components except curiosity
    reward_fitness = 1.0 * calculateFitnessReward(features);
    reward_diversity = 0.2 * calculateDiversityReward(features);
    reward_convergence = 0.1 * calculateConvergenceReward(features);

    total_reward = reward_fitness + reward_diversity + reward_convergence;
end
```

---

### 3.4 MR-V4-NoConvergence (Ablation: Remove Convergence Reward)

**Configuration:**
- Fitness weight: 1.0
- Diversity weight: 0.2
- Convergence weight: 0.0 ← **disabled**
- Curiosity weight: 0.05
- Total weight sum: 1.25

**Implementation:**
```matlab
% File: calculateMultiObjectiveReward_NoConvergence.m

function total_reward = calculateReward(features, state_encoder, config)
    % All components except convergence
    reward_fitness = 1.0 * calculateFitnessReward(features);
    reward_diversity = 0.2 * calculateDiversityReward(features);
    reward_curiosity = 0.05 * calculateCuriosityReward(state_encoder);

    total_reward = reward_fitness + reward_diversity + reward_curiosity;
end
```

---

## 4. BATCHNORMALIZATION VARIANTS (4 ablations)

### 4.1 BN-V1-None (Ablation: Remove All BatchNorm)

**Configuration:**
- Actor normalization: None
- Critic normalization: None
- Target networks: No
- Architecture: No normalization layers

**Implementation:**
```matlab
% File: createActorNetwork_NoBN.m
% Remove all batchNormalizationLayer

% Layer structure (example Layer 1):
layers = [
    fullyConnectedLayer(128, 'Name', 'fc1')
    % batchNormalizationLayer removed
    reluLayer('Name', 'relu1')
];
```

**Files to Modify:**
1. Create `algorithms/apexpso/createActorNetwork_NoBN.m` (remove lines 33, 47, 61)
2. Create `algorithms/apexpso/createCriticNetwork_NoBN.m` (remove lines 33, 47, 61)

---

### 4.2 BN-V2-ActorOnly (Ablation: BatchNorm in Actor Only)

**Configuration:**
- Actor normalization: BatchNorm
- Critic normalization: None ← **disabled**
- Target networks: No
- Architecture: Partial BatchNorm

**Implementation:**
- Keep `createActorNetwork.m` as-is (BatchNorm enabled)
- Use `createCriticNetwork_NoBN.m` (BatchNorm disabled)

**Files to Modify:**
1. Use default `algorithms/apexpso/createActorNetwork.m`
2. Use `algorithms/apexpso/createCriticNetwork_NoBN.m`

---

### 4.3 BN-V3-CriticOnly (Ablation: BatchNorm in Critic Only)

**Configuration:**
- Actor normalization: None ← **disabled**
- Critic normalization: BatchNorm
- Target networks: No
- Architecture: Partial BatchNorm

**Implementation:**
- Use `createActorNetwork_NoBN.m` (BatchNorm disabled)
- Keep `createCriticNetwork.m` as-is (BatchNorm enabled)

**Files to Modify:**
1. Use `algorithms/apexpso/createActorNetwork_NoBN.m`
2. Use default `algorithms/apexpso/createCriticNetwork.m`

---

### 4.4 BN-V6-WithTargets (Ablation: Add Target Networks - TD3/SAC)

**Configuration:**
- Actor normalization: BatchNorm
- Critic normalization: BatchNorm
- Target networks: Yes ← **enabled** (τ=0.005)
- Architecture: Traditional TD3/SAC approach

**Implementation:**

**Step 1: Add target networks to APEXPSO_Agent.m**
```matlab
% Add to properties section
properties
    actor
    critic1
    critic2
    targetActor       % NEW
    targetCritic1     % NEW
    targetCritic2     % NEW
    targetUpdateRate = 0.005  % τ = 0.005
end

% Modify constructor
function obj = APEXPSO_Agent(config)
    % ... existing initialization ...

    % Create target networks (after main networks)
    obj.targetActor = copy(obj.actor);
    obj.targetCritic1 = copy(obj.critic1);
    obj.targetCritic2 = copy(obj.critic2);
end
```

**Step 2: Modify updateCritic() to use target networks**
```matlab
% File: APEXPSO_Agent.m
% Modify lines 261-310: updateCritic()

function updateCritic(obj, batch)
    % Extract batch
    states = batch.states;
    actions = batch.actions;
    rewards = batch.rewards;
    nextStates = batch.nextStates;
    dones = batch.dones;

    % Compute target Q-values using TARGET networks
    nextActions = obj.targetActor.forward(nextStates);  % Use target actor
    nextQ1 = obj.targetCritic1.forward([nextStates, nextActions]);
    nextQ2 = obj.targetCritic2.forward([nextStates, nextActions]);
    targetQ = rewards + obj.gamma * (1 - dones) .* min(nextQ1, nextQ2);

    % Compute current Q-values
    currentQ1 = obj.critic1.forward([states, actions]);
    currentQ2 = obj.critic2.forward([states, actions]);

    % Update critics (minimize MSE)
    loss1 = mean((currentQ1 - targetQ).^2);
    loss2 = mean((currentQ2 - targetQ).^2);

    obj.critic1.backward(loss1);
    obj.critic2.backward(loss2);

    % Soft update target networks
    obj.softUpdateTargetNetworks();
end
```

**Step 3: Add soft update method**
```matlab
% File: APEXPSO_Agent.m
% Add new method

function softUpdateTargetNetworks(obj)
    % Soft update: target = τ * main + (1-τ) * target
    tau = obj.targetUpdateRate;

    % Update target actor
    obj.targetActor.weights = tau * obj.actor.weights + ...
                              (1 - tau) * obj.targetActor.weights;

    % Update target critics
    obj.targetCritic1.weights = tau * obj.critic1.weights + ...
                                (1 - tau) * obj.targetCritic1.weights;
    obj.targetCritic2.weights = tau * obj.critic2.weights + ...
                                (1 - tau) * obj.targetCritic2.weights;
end
```

**Files to Modify:**
1. Modify `algorithms/apexpso/APEXPSO_Agent.m`:
   - Add target network properties
   - Initialize target networks in constructor
   - Modify `updateCritic()` to use target networks (lines 261-310)
   - Add `softUpdateTargetNetworks()` method

---

## 5. VARIANT SWITCHING FRAMEWORK

### 5.1 Configuration System

Create a unified configuration file to switch between variants:

```matlab
% File: AblationConfig.m

classdef AblationConfig
    properties
        variantID           % e.g., 'TE-V1', 'PP-V2', 'MR-V1', 'BN-V6'

        % Temporal Encoding
        useTemporalEncoding = true;
        stateEncoderClass = 'SimpleStateEncoder';

        % Per-Particle Control
        actionDimension = 120;
        actionConverterClass = 'convertActionToPerParticleParams';

        % Multi-Objective Reward
        rewardCalculatorClass = 'calculateMultiObjectiveReward';

        % BatchNormalization
        useActorBatchNorm = true;
        useCriticBatchNorm = true;
        useTargetNetworks = false;
    end

    methods
        function obj = AblationConfig(variantID)
            obj.variantID = variantID;
            obj = obj.configureVariant(variantID);
        end

        function obj = configureVariant(obj, variantID)
            switch variantID
                % UNIFIED BASELINE
                case 'APEX-Baseline'
                    % All defaults (already set in properties)
                    % No changes needed - full APEX-PSO

                % Temporal Encoding Ablations
                case 'TE-V1-Instant'
                    obj.stateEncoderClass = 'SimpleStateEncoder_Instant';

                % Per-Particle Control Ablations
                case 'PP-V1-Global'
                    obj.actionConverterClass = 'convertActionToPerParticleParams_Global';
                    obj.actionDimension = 3;

                case 'PP-V2-Sub5'
                    obj.actionConverterClass = 'convertActionToPerParticleParams_Sub5';
                    obj.actionDimension = 15;

                % Multi-Objective Reward Ablations
                case 'MR-V1-Fitness'
                    obj.rewardCalculatorClass = 'calculateMultiObjectiveReward_FitnessOnly';

                case 'MR-V2-NoDiversity'
                    obj.rewardCalculatorClass = 'calculateMultiObjectiveReward_NoDiversity';

                case 'MR-V3-NoCuriosity'
                    obj.rewardCalculatorClass = 'calculateMultiObjectiveReward_NoCuriosity';

                case 'MR-V4-NoConvergence'
                    obj.rewardCalculatorClass = 'calculateMultiObjectiveReward_NoConvergence';

                % BatchNormalization Ablations
                case 'BN-V1-None'
                    obj.useActorBatchNorm = false;
                    obj.useCriticBatchNorm = false;
                    obj.useTargetNetworks = false;

                case 'BN-V2-ActorOnly'
                    obj.useActorBatchNorm = true;
                    obj.useCriticBatchNorm = false;
                    obj.useTargetNetworks = false;

                case 'BN-V3-CriticOnly'
                    obj.useActorBatchNorm = false;
                    obj.useCriticBatchNorm = true;
                    obj.useTargetNetworks = false;

                case 'BN-V6-WithTargets'
                    obj.useActorBatchNorm = true;
                    obj.useCriticBatchNorm = true;
                    obj.useTargetNetworks = true;

                otherwise
                    error('Unknown variant ID: %s', variantID);
            end
        end
    end
end
```

### 5.2 Master Experiment Script

```matlab
% File: run_ablation_study.m

function run_ablation_study()
    % Define all variants (1 baseline + 11 ablations)
    variants = {
        'APEX-Baseline', ...                              % Unified baseline
        'TE-V1-Instant', ...                              % Temporal ablation
        'PP-V1-Global', 'PP-V2-Sub5', ...                 % Control ablations
        'MR-V1-Fitness', 'MR-V2-NoDiversity', ...         % Reward ablations
        'MR-V3-NoCuriosity', 'MR-V4-NoConvergence', ...
        'BN-V1-None', 'BN-V2-ActorOnly', ...              % BatchNorm ablations
        'BN-V3-CriticOnly', 'BN-V6-WithTargets'
    };

    % Define scenarios
    scenarios = {'S1', 'S2', 'S3', 'S4'};
    num_runs = 30;

    % Run experiments
    for v = 1:length(variants)
        for s = 1:length(scenarios)
            for r = 1:num_runs
                variant = variants{v};
                scenario = scenarios{s};

                fprintf('Running %s, Scenario %s, Run %d/%d\n', ...
                        variant, scenario, r, num_runs);

                % Configure variant
                config = AblationConfig(variant);

                % Run experiment
                results = run_single_experiment(config, scenario, r);

                % Save results
                save_results(results, variant, scenario, r);
            end
        end
    end
end

function results = run_single_experiment(config, scenario, run_num)
    % Initialize environment
    env = setupEnvironment(scenario);

    % Initialize agent with variant configuration
    agent = APEXPSO_Agent(config);

    % Train if needed
    if requires_training(config.variantID)
        agent = trainAgent(agent, env, config);
    end

    % Evaluate
    results = evaluateAgent(agent, env);
end
```

---

## 6. FILE STRUCTURE

### Required Files

```
algorithms/apexpso/
├── ABLATION_VARIANTS_IMPLEMENTATION.md  (this file)
├── AblationConfig.m                     (variant switching)
├── run_ablation_study.m                 (master script)
│
├── UNIFIED BASELINE (APEX-PSO):
│   ├── SimpleStateEncoder.m             (45D temporal encoding)
│   ├── convertActionToPerParticleParams.m (120D per-particle control)
│   ├── calculateMultiObjectiveReward.m  (multi-objective reward)
│   ├── createActorNetwork.m             (BatchNorm enabled)
│   ├── createCriticNetwork.m            (BatchNorm enabled)
│   └── APEXPSO_Agent.m                  (no target networks - CrossQ)
│
├── ABLATION VARIANTS:
│   ├── Temporal Encoding:
│   │   └── SimpleStateEncoder_Instant.m (TE-V1: 9D instant)
│   │
│   ├── Per-Particle Control:
│   │   ├── convertActionToPerParticleParams_Global.m (PP-V1: 3D global)
│   │   └── convertActionToPerParticleParams_Sub5.m   (PP-V2: 15D subgroups)
│   │
│   ├── Multi-Objective Reward:
│   │   ├── calculateMultiObjectiveReward_FitnessOnly.m    (MR-V1)
│   │   ├── calculateMultiObjectiveReward_NoDiversity.m    (MR-V2)
│   │   ├── calculateMultiObjectiveReward_NoCuriosity.m    (MR-V3)
│   │   └── calculateMultiObjectiveReward_NoConvergence.m  (MR-V4)
│   │
│   └── BatchNormalization:
│       ├── createActorNetwork_NoBN.m    (BN-V1, BN-V3: no BatchNorm)
│       ├── createCriticNetwork_NoBN.m   (BN-V1, BN-V2: no BatchNorm)
│       └── APEXPSO_Agent_WithTargets.m  (BN-V6: add target networks)
```

**Note:** All ablations compare against the single APEX-Baseline (first 6 files).

---

## 7. COMPUTATIONAL SUMMARY

### Experimental Load

| Ablation Category | Variants | Experiments per Scenario | Total Experiments |
|-------------------|----------|-------------------------|-------------------|
| **Unified Baseline** | 1 | 1 × 30 = 30 | 30 × 4 = 120 |
| Temporal Encoding | 1 | 1 × 30 = 30 | 30 × 4 = 120 |
| Per-Particle Control | 2 | 2 × 30 = 60 | 60 × 4 = 240 |
| Multi-Objective Reward | 4 | 4 × 30 = 120 | 120 × 4 = 480 |
| BatchNormalization | 4 | 4 × 30 = 120 | 120 × 4 = 480 |
| **Total** | **12** | **360** | **1,440** |

**Savings from merged baseline:** 1,800 → 1,440 experiments (20% reduction, ~3 GPU days saved)

### Timeline (8 GPUs)

- **Phase 1 (Pilot):** 12 variants × 5 runs × 1 scenario = 60 experiments (~1 day)
- **Phase 2 (S2 Full):** 12 variants × 30 runs × 1 scenario = 360 experiments (~2-3 days)
- **Phase 3 (All Scenarios):** 12 variants × 30 runs × 4 scenarios = 1,440 experiments (~8-10 days)

**Total: ~11-14 days with 8 GPUs** (reduced from 14-17 days)

### Storage Requirements

- Per run: ~100 KB
- Total: 1,440 runs × 100 KB = **144 MB** (raw data)
- With figures and analysis: **~400 MB total**

---

## 8. QUICK START GUIDE

### Running the Baseline

```matlab
% Configure unified baseline
config = AblationConfig('APEX-Baseline');

% Setup environment
env = setupEnvironment('S2');  % Medium complexity

% Create and train agent
agent = APEXPSO_Agent(config);
agent = trainAgent(agent, env, 150);  % 150 episodes

% Evaluate
results = evaluateAgent(agent, env);

% Save baseline results (used for all comparisons)
save('results/APEX-Baseline_S2_run01.mat', 'results');
```

### Running an Ablation Variant

```matlab
% Configure ablation (example: TE-V1)
config = AblationConfig('TE-V1-Instant');

% Setup environment
env = setupEnvironment('S2');

% Create and train agent
agent = APEXPSO_Agent(config);
agent = trainAgent(agent, env, 150);

% Evaluate
results = evaluateAgent(agent, env);

% Save
save('results/TE-V1_S2_run01.mat', 'results');

% Compare to baseline
baseline = load('results/APEX-Baseline_S2_run01.mat');
degradation = (results.fitness - baseline.fitness) / baseline.fitness * 100;
fprintf('Performance degradation: %.1f%%\n', degradation);
```

### Running All Variants (Parallel)

```matlab
% Run full ablation study with parallel processing
run_ablation_study();
```

---

**END OF IMPLEMENTATION GUIDE**
