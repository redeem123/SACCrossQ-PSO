function run_rrsacpso_research_cycle(numRuns, maxIterations)
%RUN_APEXPSO_RESEARCH_CYCLE Run the retained RRSACPSO ablation suite.
%
% This script benchmarks:
%   1) The retained full RRSACPSO stack.
%   2) Leave-one-out ablations for the retained components:
%      - RankResidualControl
%      - The active SAC-side critic component
%
% All variants are evaluated with paired seeds under a shared scenario set.

    if nargin < 1 || isempty(numRuns)
        numRuns = getenvOrDefault('APEXPSO_RESEARCH_RUNS', 5);
    end
    if nargin < 2 || isempty(maxIterations)
        maxIterations = getenvOrDefault('APEXPSO_RESEARCH_MAX_ITERS', 1500);
    end
    popSize = getenvOrDefault('APEXPSO_RESEARCH_POP_SIZE', 40);

    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    cd(repoRoot);

    addpath(repoRoot, '-begin');
    addpath(genpath(fullfile(repoRoot, 'algorithms')), '-begin');
    addpath(genpath(fullfile(repoRoot, 'shared')), '-begin');
    addpath(fullfile(repoRoot, 'data'), '-begin');
    rehash path;

    outputDir = resolveResearchOutputDir(repoRoot);
    ensureDir(outputDir);
    checkpointFile = fullfile(outputDir, 'rrsacpso_checkpoint.csv');
    ensureCheckpointFile(checkpointFile);
    upgradeCheckpointFileIfNeeded(checkpointFile);
    completedKeys = loadCompletedKeyMap(checkpointFile);
    expectedKeys = containers.Map('KeyType', 'char', 'ValueType', 'logical');

    fprintf('==============================================\n');
    fprintf('  RRSACPSO Research Cycle Runner\n');
    fprintf('==============================================\n');
    fprintf('Timestamp:      %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss')));
    fprintf('Runs/variant:   %d\n', numRuns);
    fprintf('Population:     %d\n', popSize);
    fprintf('Max iterations: %d\n', maxIterations);
    fprintf('Output dir:     %s\n\n', outputDir);

    oldQuietEnv = getenv('APEXPSO_CONFIG_QUIET');
    quietCleanup = onCleanup(@() setenv('APEXPSO_CONFIG_QUIET', oldQuietEnv));
    setenv('APEXPSO_CONFIG_QUIET', '1');

    useParallel = getenvOrDefault('APEXPSO_RESEARCH_PARALLEL', 1) ~= 0;
    requestedWorkers = getenvOrDefault('APEXPSO_RESEARCH_WORKERS', 0);
    if useParallel && numRuns > 1
        if requestedWorkers > 0
            setenv('VIETANH_PARPOOL_WORKERS', num2str(requestedWorkers));
        end
        initializeParallelPool('parallel');
        pool = gcp('nocreate');
        if isempty(pool)
            warning('Parallel pool initialization failed; falling back to serial execution.');
            useParallel = false;
        else
            fprintf('Parallel mode: ON (%d workers)\n', pool.NumWorkers);
        end
    else
        useParallel = false;
        fprintf('Parallel mode: OFF\n');
    end

    variants = buildVariants(popSize, maxIterations);
    if isfield(variants(1).overrides, 'batchSize') && variants(1).overrides.batchSize > maxIterations
        warning(['Research batch size (%d) exceeds maxIterations (%d); online pilots will not reach the ', ...
                 'first gradient step. Lower APEXPSO_RESEARCH_BATCH_SIZE or increase maxIterations.'], ...
                 variants(1).overrides.batchSize, maxIterations);
    end
    variantFilter = strtrim(getenv('APEXPSO_RESEARCH_VARIANTS'));
    if ~isempty(variantFilter)
        wanted = split(string(variantFilter), ",");
        wanted = strtrim(wanted);
        keep = false(numel(variants), 1);
        for i = 1:numel(variants)
            keep(i) = any(strcmp(string(variants(i).id), wanted)) || ...
                      any(strcmp(string(variants(i).name), wanted));
        end
        variants = variants(keep);
        if isempty(variants)
            error('APEXPSO_RESEARCH_VARIANTS filter selected no variants.');
        end
    end
    scenarios = [0, 5, 10];
    scenarioOverride = strtrim(getenv('APEXPSO_RESEARCH_SCENARIOS'));
    if ~isempty(scenarioOverride)
        scenarios = str2num(scenarioOverride); %#ok<ST2NM>
        if isempty(scenarios)
            error('APEXPSO_RESEARCH_SCENARIOS must parse into a numeric vector.');
        end
    end
    environment = resolveTerrainEnvironment(struct( ...
        'mapSize', [100, 100, 100], ...
        'startPoint', [10, 95, 10], ...
        'goalPoint', [97, 2, 10], ...
        'terrainFile', ''));
    mapSize = environment.mapSize;
    startPoint = environment.startPoint;
    goalPoint = environment.goalPoint;
    terrainFile = environment.terrainFile;

    numScenarios = numel(scenarios);
    numVariants = numel(variants);
    for scenarioIdx = 1:numScenarios
        numDangerZones = scenarios(scenarioIdx);
        scenarioName = sprintf('Scenario%d_DZ%d', scenarioIdx, numDangerZones);

        [terrainGrid, terrainX, terrainY] = generateFixedTerrain(mapSize, scenarioIdx, terrainFile);
        dangerZones = generateDangerZones(numDangerZones, mapSize, terrainGrid, terrainX, terrainY);

        fprintf('\n--- %s (%d danger zones) ---\n', scenarioName, numDangerZones);
        fprintf('Variants (%d): %s\n', numVariants, strjoin(string({variants.name}), ', '));

        baseSeeds = zeros(numRuns, 1);
        for run = 1:numRuns
            baseSeeds(run) = run * 1000 + scenarioIdx * 100;
        end
        fairSeeds = getenvOrDefault('APEXPSO_RESEARCH_FAIR_SEEDS', 1) ~= 0;

        for v = 1:numVariants
            variant = variants(v);
            fprintf('  Running variant: %s\n', variant.name);
            seeds = zeros(numRuns, 1);
            if fairSeeds
                seeds = baseSeeds;
            else
                for run = 1:numRuns
                    seeds(run) = run * 1000 + scenarioIdx * 100 + v * 10;
                end
            end

            pendingRuns = [];
            for run = 1:numRuns
                runKey = makeRunKey(scenarioName, variant.name, run, seeds(run));
                expectedKeys(runKey) = true;
                if ~isKey(completedKeys, runKey)
                    pendingRuns(end+1, 1) = run; %#ok<AGROW>
                end
            end

            if isempty(pendingRuns)
                fprintf('    Resume: all %d runs already complete, skipping.\n', numRuns);
                continue;
            end

            if useParallel
                progressQueue = parallel.pool.DataQueue;
                afterEach(progressQueue, @checkpointRunResult);
                try
                    parfor idx = 1:numel(pendingRuns)
                        run = pendingRuns(idx);
                        [fitnessValue, runtimeSeconds] = executeSingleRun( ...
                            seeds(run), maxIterations, variant, ...
                            startPoint, goalPoint, dangerZones, ...
                            terrainGrid, terrainX, terrainY, mapSize);
                        send(progressQueue, struct( ...
                            'Scenario', scenarioName, ...
                            'Variant', variant.name, ...
                            'Run', run, ...
                            'Seed', seeds(run), ...
                            'Fitness', fitnessValue, ...
                            'RuntimeSeconds', runtimeSeconds));
                    end
                catch ME
                    warning('Parallel execution failed for %s/%s: %s', scenarioName, variant.name, ME.message);
                    useParallel = false;
                    completedKeys = loadCompletedKeyMap(checkpointFile);
                    pendingRuns = findMissingRuns(pendingRuns, seeds, completedKeys, scenarioName, variant.name);
                    if ~isempty(pendingRuns)
                        fprintf('    Falling back to serial execution for %d remaining runs.\n', numel(pendingRuns));
                        for idx = 1:numel(pendingRuns)
                            run = pendingRuns(idx);
                            [fitnessValue, runtimeSeconds] = executeSingleRun( ...
                                seeds(run), maxIterations, variant, ...
                                startPoint, goalPoint, dangerZones, ...
                                terrainGrid, terrainX, terrainY, mapSize);
                            checkpointRunResult(struct( ...
                                'Scenario', scenarioName, ...
                                'Variant', variant.name, ...
                                'Run', run, ...
                                'Seed', seeds(run), ...
                                'Fitness', fitnessValue, ...
                                'RuntimeSeconds', runtimeSeconds));
                        end
                    end
                end
            else
                for idx = 1:numel(pendingRuns)
                    run = pendingRuns(idx);
                    [fitnessValue, runtimeSeconds] = executeSingleRun( ...
                        seeds(run), maxIterations, variant, ...
                        startPoint, goalPoint, dangerZones, ...
                        terrainGrid, terrainX, terrainY, mapSize);
                    checkpointRunResult(struct( ...
                        'Scenario', scenarioName, ...
                        'Variant', variant.name, ...
                        'Run', run, ...
                        'Seed', seeds(run), ...
                        'Fitness', fitnessValue, ...
                        'RuntimeSeconds', runtimeSeconds));
                end
            end
        end
    end

    allRuns = readtable(checkpointFile, 'VariableNamingRule', 'preserve');
    if ~ismember('RuntimeSeconds', allRuns.Properties.VariableNames)
        allRuns.RuntimeSeconds = nan(height(allRuns), 1);
    end
    filteredMask = filterExpectedRows(allRuns, expectedKeys);
    staleRowCount = sum(~filteredMask);
    if staleRowCount > 0
        fprintf('Ignoring %d stale checkpoint rows outside the current run set.\n', staleRowCount);
    end
    allRuns = allRuns(filteredMask, :);
    allRuns = deduplicateRuns(allRuns);
    allRuns.Scenario = string(allRuns.Scenario);
    allRuns.Variant = string(allRuns.Variant);
    allRuns = sortrows(allRuns, {'Scenario', 'Variant', 'Run'}, {'ascend', 'ascend', 'ascend'});

    summary = summarizeRuns(allRuns, variants);

    writetable(allRuns, fullfile(outputDir, 'rrsacpso_all_runs.csv'));
    writetable(summary, fullfile(outputDir, 'rrsacpso_variant_summary.csv'));
    writeMarkdownSummary(summary, fullfile(outputDir, 'comparison.md'));

    fprintf('\nSaved files:\n');
    fprintf('  %s\n', fullfile(outputDir, 'rrsacpso_all_runs.csv'));
    fprintf('  %s\n', fullfile(outputDir, 'rrsacpso_variant_summary.csv'));
    fprintf('  %s\n', fullfile(outputDir, 'comparison.md'));

    function checkpointRunResult(runData)
        key = makeRunKey(runData.Scenario, runData.Variant, runData.Run, runData.Seed);
        if isKey(completedKeys, key)
            return;
        end
        appendCheckpointRow(checkpointFile, runData);
        completedKeys(key) = true;
        fprintf('    Run %2d/%2d | seed=%d | fitness=%.4f | %.2fs\n', ...
            runData.Run, numRuns, runData.Seed, runData.Fitness, runData.RuntimeSeconds);
    end
end

function pendingRuns = findMissingRuns(candidateRuns, seeds, completedKeys, scenarioName, variantName)
    keepMask = true(numel(candidateRuns), 1);
    for i = 1:numel(candidateRuns)
        runNumber = candidateRuns(i);
        keepMask(i) = ~isKey(completedKeys, makeRunKey(scenarioName, variantName, runNumber, seeds(runNumber)));
    end
    pendingRuns = candidateRuns(keepMask);
end

function variants = buildVariants(popSize, maxIterations)
    baseOverrides = struct();
    baseOverrides.disableVisualization = true;
    baseOverrides.popSize = popSize;
    baseOverrides.maxIterations = maxIterations;
    baseOverrides.useCrossScaleState = false;
    baseOverrides.stateSize = 15;
    baseOverrides.useCrossScaleBranching = false;
    baseOverrides.useCrossScaleGatedFusion = false;
    baseOverrides.useRankResidualControl = true;
    baseOverrides.paramsPerParticle = 3;
    baseOverrides.useCrossQCritic = false;
    baseOverrides.useREDQCritic = false;
    baseOverrides.useSimBaBackbone = false;
    baseOverrides.useAQECritic = false;
    baseOverrides.useDroQCritic = false;
    baseOverrides.useObservationNormalization = false;
    baseOverrides.usePrioritizedReplay = false;
    baseOverrides.useResidualCriticDecomposition = false;
    baseOverrides.usePilarReturns = false;
    baseOverrides.pilarEffectiveNStep = 3;
    baseOverrides.pilarLongHorizon = 6;
    baseOverrides.pilarMixCoefficient = 0.406;
    baseOverrides.useTQCCritic = true;
    baseOverrides.useD2RLBackbone = false;
    baseOverrides.useEmphasizingRecentExperience = false;
    baseOverrides.ereEtaStart = 0.996;
    baseOverrides.ereEtaEnd = 1.0;
    baseOverrides.ereExponentScale = 1000;
    baseOverrides.ereAnnealSteps = 1200;
    baseOverrides.ereMinRecentSize = 128;
    baseOverrides.useDelayedPolicyUpdates = false;
    baseOverrides.actorUpdateInterval = 1;
    baseOverrides.useBatchNorm = false;
    baseOverrides.useCriticBatchNorm = false;
    baseOverrides.useActorBatchNorm = false;
    baseOverrides.useJointCriticBatchForBN = false;
    baseOverrides.useWeightNormCritic = false;
    baseOverrides.criticWeightNormRadius = 1.0;
    baseOverrides.useTargetNetworks = true;
    baseOverrides.targetCriticTrainMode = false;
    baseOverrides.batchSize = 128;
    baseOverrides.actorAdamBeta1 = 0.9;
    baseOverrides.actorAdamBeta2 = 0.999;
    baseOverrides.criticAdamBeta1 = 0.9;
    baseOverrides.criticAdamBeta2 = 0.999;
    baseOverrides.redqNumCritics = 5;
    baseOverrides.redqTargetSubsetSize = 2;
    baseOverrides.redqTargetMode = 'min';
    baseOverrides.redqPolicyUpdateDelay = 5;
    baseOverrides.redqWarmupSteps = 128;
    baseOverrides.actorHiddenLayers = [256, 256];
    baseOverrides.criticHiddenLayers = [256, 256, 128];
    baseOverrides.aqeHeadsPerCritic = 3;
    baseOverrides.aqeKeepHeads = 2;
    baseOverrides.simbaActorWidth = 128;
    baseOverrides.simbaActorBlocks = 3;
    baseOverrides.simbaCriticWidth = 192;
    baseOverrides.simbaCriticBlocks = 3;
    baseOverrides.droqCriticWidth = 256;
    baseOverrides.droqCriticDepth = 3;
    baseOverrides.droqDropoutProbability = 0.01;
    baseOverrides.tqcNumQuantiles = 25;
    baseOverrides.tqcDropQuantilesPerCritic = 2;
    baseOverrides.tqcHuberKappa = 1.0;
    baseOverrides.crossScaleEncoderMode = 'delta27';
    baseOverrides.crossScaleActorGateHiddenWidth = 64;
    baseOverrides.crossScaleCriticGateHiddenWidth = 64;
    baseOverrides.utdRatio = 1;
    baseOverrides.gradientStepsPerTraining = 1;
    baseOverrides.numCritics = 2;

    baseOverrides.useCrossScaleState = getenvOrDefault('APEXPSO_RESEARCH_USE_CROSSSCALE_STATE', double(baseOverrides.useCrossScaleState)) ~= 0;
    baseOverrides.stateSize = getenvOrDefault('APEXPSO_RESEARCH_STATE_SIZE', baseOverrides.stateSize);
    baseOverrides.crossScaleEncoderMode = getenvStringOrDefault('APEXPSO_RESEARCH_ENCODER_MODE', baseOverrides.crossScaleEncoderMode);
    baseOverrides.useCrossScaleBranching = getenvOrDefault('APEXPSO_RESEARCH_USE_CROSSSCALE_BRANCHING', double(baseOverrides.useCrossScaleBranching)) ~= 0;
    baseOverrides.useCrossScaleGatedFusion = getenvOrDefault('APEXPSO_RESEARCH_USE_CROSSSCALE_GATED_FUSION', double(baseOverrides.useCrossScaleGatedFusion)) ~= 0;
    baseOverrides.crossScaleActorGateHiddenWidth = getenvOrDefault('APEXPSO_RESEARCH_CROSSSCALE_ACTOR_GATE_WIDTH', baseOverrides.crossScaleActorGateHiddenWidth);
    baseOverrides.crossScaleCriticGateHiddenWidth = getenvOrDefault('APEXPSO_RESEARCH_CROSSSCALE_CRITIC_GATE_WIDTH', baseOverrides.crossScaleCriticGateHiddenWidth);
    baseOverrides.useCrossQCritic = getenvOrDefault('APEXPSO_RESEARCH_USE_CROSSQ_CRITIC', double(baseOverrides.useCrossQCritic)) ~= 0;
    baseOverrides.useREDQCritic = getenvOrDefault('APEXPSO_RESEARCH_USE_REDQ_CRITIC', double(baseOverrides.useREDQCritic)) ~= 0;
    baseOverrides.useAQECritic = getenvOrDefault('APEXPSO_RESEARCH_USE_AQE_CRITIC', double(baseOverrides.useAQECritic)) ~= 0;
    baseOverrides.numCritics = getenvOrDefault('APEXPSO_RESEARCH_NUM_CRITICS', baseOverrides.numCritics);
    baseOverrides.redqNumCritics = getenvOrDefault('APEXPSO_RESEARCH_REDQ_NUM_CRITICS', baseOverrides.redqNumCritics);
    baseOverrides.redqTargetSubsetSize = getenvOrDefault('APEXPSO_RESEARCH_REDQ_TARGET_SUBSET', baseOverrides.redqTargetSubsetSize);
    baseOverrides.redqTargetMode = getenvStringOrDefault('APEXPSO_RESEARCH_REDQ_TARGET_MODE', baseOverrides.redqTargetMode);
    baseOverrides.redqPolicyUpdateDelay = getenvOrDefault('APEXPSO_RESEARCH_REDQ_POLICY_DELAY', baseOverrides.redqPolicyUpdateDelay);
    baseOverrides.redqWarmupSteps = getenvOrDefault('APEXPSO_RESEARCH_REDQ_WARMUP', baseOverrides.redqWarmupSteps);
    baseOverrides.aqeHeadsPerCritic = getenvOrDefault('APEXPSO_RESEARCH_AQE_HEADS_PER_CRITIC', baseOverrides.aqeHeadsPerCritic);
    baseOverrides.aqeKeepHeads = getenvOrDefault('APEXPSO_RESEARCH_AQE_KEEP_HEADS', baseOverrides.aqeKeepHeads);
    baseOverrides.useResidualCriticDecomposition = getenvOrDefault('APEXPSO_RESEARCH_USE_RESIDUAL_CRITIC', double(baseOverrides.useResidualCriticDecomposition)) ~= 0;
    baseOverrides.usePrioritizedReplay = getenvOrDefault('APEXPSO_RESEARCH_USE_PRIORITIZED_REPLAY', double(baseOverrides.usePrioritizedReplay)) ~= 0;
    baseOverrides.useDroQCritic = getenvOrDefault('APEXPSO_RESEARCH_USE_DROQ_CRITIC', double(baseOverrides.useDroQCritic)) ~= 0;
    baseOverrides.useObservationNormalization = getenvOrDefault('APEXPSO_RESEARCH_USE_OBS_NORM', double(baseOverrides.useObservationNormalization)) ~= 0;
    baseOverrides.usePilarReturns = getenvOrDefault('APEXPSO_RESEARCH_USE_PILAR_RETURNS', double(baseOverrides.usePilarReturns)) ~= 0;
    baseOverrides.pilarEffectiveNStep = getenvOrDefault('APEXPSO_RESEARCH_PILAR_EFFECTIVE_N', baseOverrides.pilarEffectiveNStep);
    baseOverrides.pilarLongHorizon = getenvOrDefault('APEXPSO_RESEARCH_PILAR_LONG_HORIZON', baseOverrides.pilarLongHorizon);
    baseOverrides.pilarMixCoefficient = getenvOrDefault('APEXPSO_RESEARCH_PILAR_MIX', baseOverrides.pilarMixCoefficient);
    baseOverrides.useTQCCritic = getenvOrDefault('APEXPSO_RESEARCH_USE_TQC_CRITIC', double(baseOverrides.useTQCCritic)) ~= 0;
    baseOverrides.tqcNumQuantiles = getenvOrDefault('APEXPSO_RESEARCH_TQC_NUM_QUANTILES', baseOverrides.tqcNumQuantiles);
    baseOverrides.tqcDropQuantilesPerCritic = getenvOrDefault('APEXPSO_RESEARCH_TQC_DROP_PER_CRITIC', baseOverrides.tqcDropQuantilesPerCritic);
    baseOverrides.tqcHuberKappa = getenvOrDefault('APEXPSO_RESEARCH_TQC_HUBER_KAPPA', baseOverrides.tqcHuberKappa);
    baseOverrides.useD2RLBackbone = getenvOrDefault('APEXPSO_RESEARCH_USE_D2RL', double(baseOverrides.useD2RLBackbone)) ~= 0;
    baseOverrides.useEmphasizingRecentExperience = getenvOrDefault('APEXPSO_RESEARCH_USE_ERE', double(baseOverrides.useEmphasizingRecentExperience)) ~= 0;
    baseOverrides.ereEtaStart = getenvOrDefault('APEXPSO_RESEARCH_ERE_ETA_START', baseOverrides.ereEtaStart);
    baseOverrides.ereEtaEnd = getenvOrDefault('APEXPSO_RESEARCH_ERE_ETA_END', baseOverrides.ereEtaEnd);
    baseOverrides.ereExponentScale = getenvOrDefault('APEXPSO_RESEARCH_ERE_EXPONENT_SCALE', baseOverrides.ereExponentScale);
    baseOverrides.ereAnnealSteps = getenvOrDefault('APEXPSO_RESEARCH_ERE_ANNEAL_STEPS', baseOverrides.ereAnnealSteps);
    baseOverrides.ereMinRecentSize = getenvOrDefault('APEXPSO_RESEARCH_ERE_MIN_RECENT_SIZE', baseOverrides.ereMinRecentSize);
    baseOverrides.useDelayedPolicyUpdates = getenvOrDefault('APEXPSO_RESEARCH_USE_DELAYED_POLICY_UPDATES', double(baseOverrides.useDelayedPolicyUpdates)) ~= 0;
    baseOverrides.actorUpdateInterval = getenvOrDefault('APEXPSO_RESEARCH_ACTOR_UPDATE_INTERVAL', baseOverrides.actorUpdateInterval);
    baseOverrides.useCriticBatchNorm = getenvOrDefault('APEXPSO_RESEARCH_USE_CRITIC_BATCHNORM', double(baseOverrides.useCriticBatchNorm)) ~= 0;
    baseOverrides.useActorBatchNorm = getenvOrDefault('APEXPSO_RESEARCH_USE_ACTOR_BATCHNORM', double(baseOverrides.useActorBatchNorm)) ~= 0;
    baseOverrides.useJointCriticBatchForBN = getenvOrDefault('APEXPSO_RESEARCH_USE_JOINT_CRITIC_BATCH_BN', double(baseOverrides.useJointCriticBatchForBN)) ~= 0;
    baseOverrides.useTargetNetworks = getenvOrDefault('APEXPSO_RESEARCH_USE_TARGET_NETWORKS', double(baseOverrides.useTargetNetworks)) ~= 0;
    baseOverrides.useWeightNormCritic = getenvOrDefault('APEXPSO_RESEARCH_USE_WEIGHT_NORM_CRITIC', double(baseOverrides.useWeightNormCritic)) ~= 0;
    baseOverrides.criticWeightNormRadius = getenvOrDefault('APEXPSO_RESEARCH_CRITIC_WEIGHT_NORM_RADIUS', baseOverrides.criticWeightNormRadius);
    baseOverrides.actorAdamBeta1 = getenvOrDefault('APEXPSO_RESEARCH_ACTOR_ADAM_BETA1', baseOverrides.actorAdamBeta1);
    baseOverrides.actorAdamBeta2 = getenvOrDefault('APEXPSO_RESEARCH_ACTOR_ADAM_BETA2', baseOverrides.actorAdamBeta2);
    baseOverrides.criticAdamBeta1 = getenvOrDefault('APEXPSO_RESEARCH_CRITIC_ADAM_BETA1', baseOverrides.criticAdamBeta1);
    baseOverrides.criticAdamBeta2 = getenvOrDefault('APEXPSO_RESEARCH_CRITIC_ADAM_BETA2', baseOverrides.criticAdamBeta2);
    baseOverrides.simbaActorWidth = getenvOrDefault('APEXPSO_RESEARCH_SIMBA_ACTOR_WIDTH', baseOverrides.simbaActorWidth);
    baseOverrides.simbaActorBlocks = getenvOrDefault('APEXPSO_RESEARCH_SIMBA_ACTOR_BLOCKS', baseOverrides.simbaActorBlocks);
    baseOverrides.simbaCriticWidth = getenvOrDefault('APEXPSO_RESEARCH_SIMBA_CRITIC_WIDTH', baseOverrides.simbaCriticWidth);
    baseOverrides.simbaCriticBlocks = getenvOrDefault('APEXPSO_RESEARCH_SIMBA_CRITIC_BLOCKS', baseOverrides.simbaCriticBlocks);
    baseOverrides.droqCriticWidth = getenvOrDefault('APEXPSO_RESEARCH_DROQ_CRITIC_WIDTH', baseOverrides.droqCriticWidth);
    baseOverrides.droqCriticDepth = getenvOrDefault('APEXPSO_RESEARCH_DROQ_CRITIC_DEPTH', baseOverrides.droqCriticDepth);
    baseOverrides.droqDropoutProbability = getenvOrDefault('APEXPSO_RESEARCH_DROQ_DROPOUT', baseOverrides.droqDropoutProbability);
    baseOverrides.utdRatio = getenvOrDefault('APEXPSO_RESEARCH_UTD_RATIO', baseOverrides.utdRatio);
    baseOverrides.gradientStepsPerTraining = getenvOrDefault('APEXPSO_RESEARCH_GRADIENT_STEPS', baseOverrides.gradientStepsPerTraining);
    baseOverrides.batchSize = getenvOrDefault('APEXPSO_RESEARCH_BATCH_SIZE', baseOverrides.batchSize);
    baseOverrides.warmupPeriod = getenvOrDefault('APEXPSO_RESEARCH_WARMUP', 0);
    actorHiddenOverride = strtrim(getenv('APEXPSO_RESEARCH_ACTOR_HIDDEN_LAYERS'));
    if ~isempty(actorHiddenOverride)
        actorHiddenLayers = str2num(actorHiddenOverride); %#ok<ST2NM>
        if isempty(actorHiddenLayers)
            error('APEXPSO_RESEARCH_ACTOR_HIDDEN_LAYERS must parse into a numeric vector.');
        end
        baseOverrides.actorHiddenLayers = actorHiddenLayers;
    end
    criticHiddenOverride = strtrim(getenv('APEXPSO_RESEARCH_CRITIC_HIDDEN_LAYERS'));
    if ~isempty(criticHiddenOverride)
        criticHiddenLayers = str2num(criticHiddenOverride); %#ok<ST2NM>
        if isempty(criticHiddenLayers)
            error('APEXPSO_RESEARCH_CRITIC_HIDDEN_LAYERS must parse into a numeric vector.');
        end
        baseOverrides.criticHiddenLayers = criticHiddenLayers;
    end

    baseOverrides = normalizeSacComponentOverrides(baseOverrides);

    baseOverrides = applyParamModeOverrides(baseOverrides, 'rank-residual');

    variants = repmat(struct( ...
        'name', '', ...
        'id', '', ...
        'family', '', ...
        'paramMode', '', ...
        'overrides', struct()), 1, 3);

    variants(1) = makeVariant('APEX-Full', 'apex_full', 'core_loo', 'rank-residual', baseOverrides);
    variants(2) = makeVariant('LOO-NoRankResidualControl', 'loo_no_rank_residual_control', 'core_loo', 'global', ...
        applyParamModeOverrides(mergeStruct(baseOverrides, struct('useRankResidualControl', false)), 'global'));
    [componentName, componentId, componentOffOverrides] = getActiveSacComponentOffOverrides(baseOverrides);
    variants(3) = makeVariant(sprintf('LOO-No%s', componentName), sprintf('loo_no_%s', componentId), ...
        'core_loo', 'rank-residual', mergeStruct(baseOverrides, componentOffOverrides));
end

function variant = makeVariant(name, id, family, paramMode, overrides)
    variant = struct( ...
        'name', name, ...
        'id', id, ...
        'family', family, ...
        'paramMode', paramMode, ...
        'overrides', overrides);
end

function [componentName, componentId, overrides] = getActiveSacComponentOffOverrides(baseOverrides)
    if isfield(baseOverrides, 'useREDQCritic') && baseOverrides.useREDQCritic
        componentName = 'REDQCritic';
        componentId = 'redq_critic';
        overrides = normalizeSacComponentOverrides(struct( ...
            'useCrossQCritic', false, ...
            'useREDQCritic', false, ...
            'useAQECritic', false, ...
            'useDroQCritic', false, ...
            'useTQCCritic', false, ...
            'numCritics', 2, ...
            'redqNumCritics', 2, ...
            'redqTargetSubsetSize', 2, ...
            'useCriticBatchNorm', false, ...
            'useActorBatchNorm', false, ...
            'useJointCriticBatchForBN', false, ...
            'useWeightNormCritic', false, ...
            'useTargetNetworks', true, ...
            'useDelayedPolicyUpdates', false, ...
            'actorUpdateInterval', 1, ...
            'utdRatio', 1, ...
            'gradientStepsPerTraining', 1, ...
            'warmupPeriod', 0));
        return;
    end

    if isfield(baseOverrides, 'useCrossQCritic') && baseOverrides.useCrossQCritic
        componentName = 'CrossQCritic';
        componentId = 'crossq_critic';
        overrides = normalizeSacComponentOverrides(struct( ...
            'useCrossQCritic', false, ...
            'useREDQCritic', false, ...
            'useAQECritic', false, ...
            'useDroQCritic', false, ...
            'useCriticBatchNorm', false, ...
            'useActorBatchNorm', false, ...
            'useJointCriticBatchForBN', false, ...
            'useWeightNormCritic', false, ...
            'useTargetNetworks', true));
        return;
    end

    if isfield(baseOverrides, 'useAQECritic') && baseOverrides.useAQECritic
        componentName = 'AQECritic';
        componentId = 'aqe_critic';
        overrides = normalizeSacComponentOverrides(struct( ...
            'useREDQCritic', false, ...
            'useAQECritic', false, ...
            'useDroQCritic', false, ...
            'aqeHeadsPerCritic', 1, ...
            'aqeKeepHeads', 1));
        return;
    end

    if isfield(baseOverrides, 'useTQCCritic') && baseOverrides.useTQCCritic
        componentName = 'TQCTruncQuantileTarget';
        componentId = 'tqc_trunc_quantile_target';
        overrides = normalizeSacComponentOverrides(struct( ...
            'useCrossQCritic', false, ...
            'useREDQCritic', false, ...
            'useAQECritic', false, ...
            'useDroQCritic', false, ...
            'useTQCCritic', true, ...
            'useTargetNetworks', true, ...
            'numCritics', 2, ...
            'tqcNumQuantiles', max(5, round(baseOverrides.tqcNumQuantiles)), ...
            'tqcDropQuantilesPerCritic', 0));
        return;
    end

    error('No supported SAC-side retained component is active in baseOverrides.');
end

function overrides = applyParamModeOverrides(overrides, paramMode)
    overrides.paramMode = paramMode;
    switch paramMode
        case 'global'
            overrides.usePerParticleActions = false;
            overrides.useRankResidualControl = false;
            overrides.actionSize = 3;
        case '5subgroup'
            overrides.usePerParticleActions = false;
            overrides.useRankResidualControl = false;
            overrides.actionSize = 15;
        case 'per-particle'
            overrides.usePerParticleActions = true;
            overrides.useRankResidualControl = false;
            overrides.actionSize = overrides.popSize * overrides.paramsPerParticle;
        case 'rank-residual'
            overrides.usePerParticleActions = false;
            overrides.useRankResidualControl = true;
            overrides.actionSize = 9;
        otherwise
            error('Unsupported paramMode: %s', paramMode);
    end
    overrides.targetEntropy = -overrides.actionSize;
end

function overrides = normalizeSacComponentOverrides(overrides)
    if ~isfield(overrides, 'useCrossQCritic')
        overrides.useCrossQCritic = false;
    end
    if ~isfield(overrides, 'useREDQCritic')
        overrides.useREDQCritic = false;
    end
    if ~isfield(overrides, 'useAQECritic')
        overrides.useAQECritic = false;
    end
    if ~isfield(overrides, 'useDroQCritic')
        overrides.useDroQCritic = false;
    end
    if ~isfield(overrides, 'redqNumCritics')
        overrides.redqNumCritics = 5;
    end
    if ~isfield(overrides, 'redqTargetSubsetSize')
        overrides.redqTargetSubsetSize = 2;
    end
    if ~isfield(overrides, 'redqTargetMode')
        overrides.redqTargetMode = 'min';
    end
    if ~isfield(overrides, 'redqPolicyUpdateDelay')
        overrides.redqPolicyUpdateDelay = 5;
    end
    if ~isfield(overrides, 'redqWarmupSteps')
        overrides.redqWarmupSteps = 128;
    end
    if ~isfield(overrides, 'warmupPeriod')
        overrides.warmupPeriod = 0;
    end
    if ~isfield(overrides, 'utdRatio')
        overrides.utdRatio = 1;
    end
    if ~isfield(overrides, 'gradientStepsPerTraining')
        overrides.gradientStepsPerTraining = overrides.utdRatio;
    end
    if ~isfield(overrides, 'numCritics')
        overrides.numCritics = 2;
    end
    if ~isfield(overrides, 'actorUpdateInterval')
        overrides.actorUpdateInterval = 1;
    end
    if ~isfield(overrides, 'useDelayedPolicyUpdates')
        overrides.useDelayedPolicyUpdates = false;
    end
    if ~isfield(overrides, 'tqcNumQuantiles')
        overrides.tqcNumQuantiles = 25;
    end
    if ~isfield(overrides, 'tqcDropQuantilesPerCritic')
        overrides.tqcDropQuantilesPerCritic = 2;
    end
    if ~isfield(overrides, 'tqcHuberKappa')
        overrides.tqcHuberKappa = 1.0;
    end

    if overrides.useCrossQCritic
        overrides.useREDQCritic = false;
        overrides.useAQECritic = false;
        overrides.useDroQCritic = false;
        overrides.useTQCCritic = false;
        overrides.useCriticBatchNorm = true;
        overrides.useActorBatchNorm = false;
        overrides.useJointCriticBatchForBN = true;
        overrides.useWeightNormCritic = false;
        overrides.useTargetNetworks = false;
        overrides.useDelayedPolicyUpdates = false;
        overrides.actorUpdateInterval = 1;
        overrides.utdRatio = 1;
        overrides.gradientStepsPerTraining = 1;
        overrides.warmupPeriod = max(0, round(overrides.warmupPeriod));
        return;
    end

    if overrides.useREDQCritic
        overrides.useCrossQCritic = false;
        overrides.useAQECritic = false;
        overrides.useDroQCritic = false;
        overrides.useTQCCritic = false;
        overrides.useCriticBatchNorm = false;
        overrides.useJointCriticBatchForBN = false;
        overrides.useTargetNetworks = true;
        overrides.numCritics = max(3, round(max(overrides.numCritics, overrides.redqNumCritics)));
        overrides.redqNumCritics = overrides.numCritics;
        overrides.redqTargetSubsetSize = max(2, min(overrides.numCritics, round(overrides.redqTargetSubsetSize)));
        overrides.useDelayedPolicyUpdates = overrides.gradientStepsPerTraining > 1;
        if overrides.useDelayedPolicyUpdates
            overrides.actorUpdateInterval = max(1, round(overrides.redqPolicyUpdateDelay));
        else
            overrides.actorUpdateInterval = 1;
        end
        overrides.warmupPeriod = max(overrides.warmupPeriod, round(overrides.redqWarmupSteps));
        overrides.utdRatio = max(1, round(overrides.utdRatio));
        overrides.gradientStepsPerTraining = max(1, round(overrides.gradientStepsPerTraining));
        return;
    end

    if ~isfield(overrides, 'useTQCCritic')
        overrides.useTQCCritic = false;
    end

    if overrides.useAQECritic
        overrides.useCrossQCritic = false;
        overrides.useREDQCritic = false;
        overrides.useDroQCritic = false;
        overrides.useTQCCritic = false;
        overrides.useCriticBatchNorm = false;
        overrides.useActorBatchNorm = false;
        overrides.useJointCriticBatchForBN = false;
        overrides.useWeightNormCritic = false;
        overrides.useTargetNetworks = true;
        overrides.useDelayedPolicyUpdates = false;
        overrides.actorUpdateInterval = 1;
        overrides.numCritics = max(2, round(overrides.numCritics));
        totalAqeHeads = overrides.numCritics * max(1, round(overrides.aqeHeadsPerCritic));
        overrides.aqeHeadsPerCritic = max(1, round(overrides.aqeHeadsPerCritic));
        overrides.aqeKeepHeads = max(1, min(totalAqeHeads, round(overrides.aqeKeepHeads)));
        overrides.utdRatio = max(1, round(overrides.utdRatio));
        overrides.gradientStepsPerTraining = max(1, round(overrides.gradientStepsPerTraining));
        overrides.warmupPeriod = max(0, round(overrides.warmupPeriod));
        return;
    end

    if overrides.useDroQCritic
        overrides.useCrossQCritic = false;
        overrides.useREDQCritic = false;
        overrides.useAQECritic = false;
        overrides.useTQCCritic = false;
        overrides.useCriticBatchNorm = false;
        overrides.useActorBatchNorm = false;
        overrides.useJointCriticBatchForBN = false;
        overrides.useTargetNetworks = true;
        overrides.useDelayedPolicyUpdates = false;
        overrides.actorUpdateInterval = 1;
        overrides.numCritics = 2;
        overrides.utdRatio = max(1, round(overrides.utdRatio));
        overrides.gradientStepsPerTraining = max(1, round(overrides.gradientStepsPerTraining));
        overrides.warmupPeriod = max(0, round(overrides.warmupPeriod));
        return;
    end

    if overrides.useTQCCritic
        overrides.useCrossQCritic = false;
        overrides.useREDQCritic = false;
        overrides.useAQECritic = false;
        overrides.useDroQCritic = false;
        overrides.useCriticBatchNorm = false;
        overrides.useActorBatchNorm = false;
        overrides.useJointCriticBatchForBN = false;
        overrides.useWeightNormCritic = false;
        overrides.useTargetNetworks = true;
        overrides.useDelayedPolicyUpdates = false;
        overrides.actorUpdateInterval = 1;
        overrides.numCritics = 2;
        overrides.utdRatio = 1;
        overrides.gradientStepsPerTraining = 1;
        overrides.warmupPeriod = max(0, round(overrides.warmupPeriod));
        overrides.tqcNumQuantiles = max(5, round(overrides.tqcNumQuantiles));
        overrides.tqcDropQuantilesPerCritic = max(0, min(overrides.tqcNumQuantiles - 1, ...
            round(overrides.tqcDropQuantilesPerCritic)));
        overrides.tqcHuberKappa = max(eps, overrides.tqcHuberKappa);
        return;
    end

    overrides.numCritics = max(2, round(overrides.numCritics));
    overrides.redqNumCritics = max(3, round(overrides.redqNumCritics));
    overrides.redqTargetSubsetSize = max(2, round(overrides.redqTargetSubsetSize));
    overrides.actorUpdateInterval = max(1, round(overrides.actorUpdateInterval));
    overrides.gradientStepsPerTraining = max(1, round(overrides.gradientStepsPerTraining));
    overrides.utdRatio = max(1, round(overrides.utdRatio));
    overrides.warmupPeriod = max(0, round(overrides.warmupPeriod));
end

function [fitnessValue, runtimeSeconds] = executeSingleRun(seed, maxIterations, variant, ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize)
    rng(seed, 'twister');

    params = struct();
    params.maxIterations = maxIterations;
    params.paramMode = variant.paramMode;
    params.variant = variant.id;
    params.configOverrides = variant.overrides;

    timerStart = tic;
    [~, stats] = callGlobalPlanningAlgorithm('RRSACPSO_Online', ...
        startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params);
    runtimeSeconds = toc(timerStart);

    if isfield(stats, 'actualBestFitness')
        fitnessValue = stats.actualBestFitness;
    elseif isfield(stats, 'finalFitness')
        fitnessValue = stats.finalFitness;
    else
        error('Missing fitness in algorithm stats.');
    end
end

function summary = summarizeRuns(allRuns, variants)
    scenarios = unique(allRuns.Scenario, 'stable');
    numScenarios = numel(scenarios);
    fullName = string(variants(1).name);
    numComparisons = numScenarios * (numel(variants) - 1);

    scenarioCol = strings(numComparisons, 1);
    familyCol = strings(numComparisons, 1);
    variantCol = strings(numComparisons, 1);
    nCol = zeros(numComparisons, 1);
    fullMeanCol = zeros(numComparisons, 1);
    fullStdCol = zeros(numComparisons, 1);
    variantMeanCol = zeros(numComparisons, 1);
    variantStdCol = zeros(numComparisons, 1);
    deltaCol = zeros(numComparisons, 1);
    fullMeanRuntimeCol = nan(numComparisons, 1);
    fullStdRuntimeCol = nan(numComparisons, 1);
    variantMeanRuntimeCol = nan(numComparisons, 1);
    variantStdRuntimeCol = nan(numComparisons, 1);
    tStatCol = zeros(numComparisons, 1);
    pValueCol = zeros(numComparisons, 1);
    significantCol = false(numComparisons, 1);
    winnerCol = strings(numComparisons, 1);

    row = 0;
    for s = 1:numScenarios
        scenarioName = scenarios(s);
        fullMask = allRuns.Scenario == scenarioName & allRuns.Variant == fullName;
        fullRuns = sortrows(allRuns(fullMask, {'Run', 'Fitness', 'RuntimeSeconds'}), 'Run');
        fullVals = fullRuns.Fitness;
        fullTimes = fullRuns.RuntimeSeconds;

        for v = 2:numel(variants)
            variantName = string(variants(v).name);
            variantMask = allRuns.Scenario == scenarioName & allRuns.Variant == variantName;
            variantRuns = sortrows(allRuns(variantMask, {'Run', 'Fitness', 'RuntimeSeconds'}), 'Run');

            if height(fullRuns) ~= height(variantRuns) || any(fullRuns.Run ~= variantRuns.Run)
                error('Run pairing mismatch in %s for %s.', scenarioName, variantName);
            end

            variantVals = variantRuns.Fitness;
            variantTimes = variantRuns.RuntimeSeconds;
            row = row + 1;
            scenarioCol(row) = scenarioName;
            familyCol(row) = string(variants(v).family);
            variantCol(row) = variantName;
            nCol(row) = numel(fullVals);
            fullMeanCol(row) = mean(fullVals);
            fullStdCol(row) = std(fullVals);
            variantMeanCol(row) = mean(variantVals);
            variantStdCol(row) = std(variantVals);
            deltaCol(row) = fullMeanCol(row) - variantMeanCol(row);
            fullMeanRuntimeCol(row) = meanOrNaN(fullTimes);
            fullStdRuntimeCol(row) = stdOrNaN(fullTimes);
            variantMeanRuntimeCol(row) = meanOrNaN(variantTimes);
            variantStdRuntimeCol(row) = stdOrNaN(variantTimes);

            [tStat, pOneSided] = pairedBetterPValue(variantVals, fullVals);
            tStatCol(row) = tStat;
            pValueCol(row) = pOneSided;
            significantCol(row) = pOneSided < 0.05 && deltaCol(row) < 0;

            if deltaCol(row) < 0
                winnerCol(row) = fullName;
            elseif deltaCol(row) > 0
                winnerCol(row) = variantName;
            else
                winnerCol(row) = "Tie";
            end
        end
    end

    summary = table(scenarioCol, familyCol, variantCol, nCol, fullMeanCol, fullStdCol, ...
        variantMeanCol, variantStdCol, deltaCol, fullMeanRuntimeCol, fullStdRuntimeCol, ...
        variantMeanRuntimeCol, variantStdRuntimeCol, ...
        tStatCol, pValueCol, ...
        significantCol, winnerCol, ...
        'VariableNames', {'Scenario', 'Family', 'Variant', 'NumRuns', 'FullMeanFitness', 'FullStdFitness', ...
        'VariantMeanFitness', 'VariantStdFitness', 'MeanDeltaFullMinusVariant', ...
        'FullMeanRuntimeSeconds', 'FullStdRuntimeSeconds', ...
        'VariantMeanRuntimeSeconds', 'VariantStdRuntimeSeconds', ...
        'PairedTStat', 'OneSidedPValueFullBetter', 'FullSignificantlyBetter', 'Winner'});
end

function writeMarkdownSummary(summary, outputPath)
    fid = fopen(outputPath, 'w');
    if fid == -1
        error('Failed to open markdown output: %s', outputPath);
    end
    cleaner = onCleanup(@() fclose(fid));

    fprintf(fid, '# RRSACPSO LOO Ablation Summary\n\n');
    fprintf(fid, '| Scenario | Family | Variant | Runs | Full Mean | Full Std | Variant Mean | Variant Std | Delta (Full-Variant) | Full Mean Runtime (s) | Variant Mean Runtime (s) | One-sided p (Full better) | Full significantly better? | Winner |\n');
    fprintf(fid, '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n');
    for i = 1:height(summary)
        fprintf(fid, '| %s | %s | %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.2f | %.2f | %.6f | %s | %s |\n', ...
            summary.Scenario(i), summary.Family(i), summary.Variant(i), summary.NumRuns(i), ...
            summary.FullMeanFitness(i), summary.FullStdFitness(i), ...
            summary.VariantMeanFitness(i), summary.VariantStdFitness(i), ...
            summary.MeanDeltaFullMinusVariant(i), ...
            summary.FullMeanRuntimeSeconds(i), summary.VariantMeanRuntimeSeconds(i), ...
            summary.OneSidedPValueFullBetter(i), ...
            boolToYesNo(summary.FullSignificantlyBetter(i)), summary.Winner(i));
    end
end

function out = mergeStruct(base, updates)
    out = base;
    f = fieldnames(updates);
    for i = 1:numel(f)
        out.(f{i}) = updates.(f{i});
    end
end

function ensureDir(pathStr)
    if exist(pathStr, 'dir') ~= 7
        mkdir(pathStr);
    end
end

function outputDir = resolveResearchOutputDir(repoRoot)
    resumeDir = strtrim(getenv('APEXPSO_RESEARCH_RESUME_DIR'));
    if ~isempty(resumeDir)
        outputDir = resumeDir;
        return;
    end

    outputOverride = strtrim(getenv('APEXPSO_RESEARCH_OUTPUT_DIR'));
    if ~isempty(outputOverride)
        outputDir = outputOverride;
        return;
    end

    runStamp = char(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss'));
    outputDir = fullfile(repoRoot, 'outputs', 'research', runStamp);
end

function ensureCheckpointFile(checkpointFile)
    if exist(checkpointFile, 'file') == 2
        return;
    end

    fid = fopen(checkpointFile, 'w');
    if fid == -1
        error('Failed to create checkpoint file: %s', checkpointFile);
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, 'Scenario,Variant,Run,Seed,Fitness,RuntimeSeconds\n');
end

function upgradeCheckpointFileIfNeeded(checkpointFile)
    checkpointTable = readtable(checkpointFile, 'VariableNamingRule', 'preserve');
    if ismember('RuntimeSeconds', checkpointTable.Properties.VariableNames)
        return;
    end

    checkpointTable.RuntimeSeconds = nan(height(checkpointTable), 1);
    writetable(checkpointTable, checkpointFile);
end

function completedKeys = loadCompletedKeyMap(checkpointFile)
    completedKeys = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    if exist(checkpointFile, 'file') ~= 2
        return;
    end

    checkpointTable = readtable(checkpointFile, 'VariableNamingRule', 'preserve');
    for i = 1:height(checkpointTable)
        completedKeys(makeRunKey(checkpointTable.Scenario(i), checkpointTable.Variant(i), ...
            checkpointTable.Run(i), checkpointTable.Seed(i))) = true;
    end
end

function appendCheckpointRow(checkpointFile, runData)
    fid = fopen(checkpointFile, 'a');
    if fid == -1
        error('Failed to append checkpoint file: %s', checkpointFile);
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s,%s,%d,%d,%.10f,%.10f\n', ...
        char(string(runData.Scenario)), char(string(runData.Variant)), ...
        runData.Run, runData.Seed, runData.Fitness, runData.RuntimeSeconds);
end

function key = makeRunKey(scenarioName, variantName, runNumber, seed)
    key = sprintf('%s|%s|%d|%d', char(string(scenarioName)), char(string(variantName)), runNumber, seed);
end

function v = getenvOrDefault(envName, defaultVal)
    raw = strtrim(getenv(envName));
    if isempty(raw)
        v = defaultVal;
        return;
    end
    parsed = str2double(raw);
    if isnan(parsed)
        v = defaultVal;
    else
        v = parsed;
    end
end

function v = getenvStringOrDefault(envName, defaultVal)
    raw = strtrim(getenv(envName));
    if isempty(raw)
        v = defaultVal;
    else
        v = raw;
    end
end

function out = boolToYesNo(flag)
    if flag
        out = 'YES';
    else
        out = 'NO';
    end
end

function out = meanOrNaN(values)
    values = values(isfinite(values));
    if isempty(values)
        out = nan;
    else
        out = mean(values);
    end
end

function out = stdOrNaN(values)
    values = values(isfinite(values));
    if isempty(values)
        out = nan;
    elseif isscalar(values)
        out = 0;
    else
        out = std(values);
    end
end

function keepMask = filterExpectedRows(allRuns, expectedKeys)
    keepMask = false(height(allRuns), 1);
    if isempty(allRuns)
        return;
    end

    for i = 1:height(allRuns)
        keepMask(i) = isKey(expectedKeys, makeRunKey(allRuns.Scenario(i), allRuns.Variant(i), ...
            allRuns.Run(i), allRuns.Seed(i)));
    end
end

function uniqueRuns = deduplicateRuns(allRuns)
    if isempty(allRuns)
        uniqueRuns = allRuns;
        return;
    end

    rowKeys = strings(height(allRuns), 1);
    for i = 1:height(allRuns)
        rowKeys(i) = string(makeRunKey(allRuns.Scenario(i), allRuns.Variant(i), ...
            allRuns.Run(i), allRuns.Seed(i)));
    end

    [~, uniqueIdx] = unique(rowKeys, 'stable');
    duplicateCount = height(allRuns) - numel(uniqueIdx);
    if duplicateCount > 0
        fprintf('Ignoring %d duplicate checkpoint rows.\n', duplicateCount);
    end
    uniqueRuns = allRuns(sort(uniqueIdx), :);
end

function [tStat, pValue] = pairedBetterPValue(sample1, sample2)
    [tStat, pTwoSided, ~] = pairedTTest(sample1, sample2);
    if ~isfinite(tStat)
        if tStat > 0
            pValue = 0;
        else
            pValue = 1;
        end
        return;
    end

    if tStat >= 0
        pValue = pTwoSided / 2;
    else
        pValue = 1 - pTwoSided / 2;
    end
end
