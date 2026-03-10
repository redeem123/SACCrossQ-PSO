function result = run_apex_full_loop_iteration()
%RUN_APEX_FULL_LOOP_ITERATION
% Automation runner for:
% 1) Full vs NoCrossScaleState benchmark on scenarios 1..3
% 2) If needed, one minimal evidence-driven ablation iteration
% 3) Re-benchmark with retained configuration

    clc;

    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repoRoot);
    addpath(genpath(fullfile(repoRoot, 'algorithms')));
    addpath(genpath(fullfile(repoRoot, 'shared')));
    addpath(genpath(fullfile(repoRoot, 'scripts')));
    addpath(fullfile(repoRoot, 'data'));

    outputsDir = fullfile(repoRoot, 'outputs', 'apex_full_loop');
    if exist(outputsDir, 'dir') ~= 7
        mkdir(outputsDir);
    end

    % Stable comparison settings
    settings.numRuns = 3;
    settings.alpha = 0.05;
    settings.scenarios = [0; 5; 10];
    settings.mapSize = [100, 100, 100];
    settings.startPoint = [10, 95, 10];
    settings.goalPoint = [97, 2, 10];
    settings.terrainFile = '';
    settings.globalPlanInterval = 1e5;
    settings.pathDeviationThreshold = 1e5;
    settings.obstacleChangeThreshold = 1e5;
    settings.timeStep = 0.1;
    settings.totalTime = 0.1;
    settings.seedOffsets = struct('full', 1, 'nocross', 2);
    settings = resolveTerrainEnvironment(settings);

    % Map user terminology:
    % "NoCrossScaleState" in this codebase corresponds to the No-CrossQ ablation model.
    fullCurrent = '/Users/hust-hwashin621m/Desktop/vietanhpaper-2/models/RRSACPSO/rrsacpso_perparticle.mat';
    noCrossScaleState = '/Users/hust-hwashin621m/Desktop/vietanhpaper-2/models/RRSACPSO/notlog/rrsacpso_abl_nocrossq.mat';

    assert(isfile(fullCurrent), 'Missing Full model: %s', fullCurrent);
    assert(isfile(noCrossScaleState), 'Missing NoCrossScaleState model: %s', noCrossScaleState);

    fprintf('\n=== APEX Full Loop Iteration ===\n');
    fprintf('Full model: %s\n', fullCurrent);
    fprintf('NoCrossScaleState model: %s\n', noCrossScaleState);
    fprintf('Runs per scenario: %d\n\n', settings.numRuns);

    baseline = runFullVsNoCross(fullCurrent, noCrossScaleState, settings);
    baseline.allSignificant = all([baseline.scenario.fullSignificantBetter]);

    result = struct();
    result.timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
    result.settings = settings;
    result.mapping = struct( ...
        'fullName', 'RRSACPSO Full', ...
        'noCrossScaleStateName', 'RRSACPSO NoCrossScaleState (No CrossQ ablation model)');
    result.baseline = baseline;
    result.iterationRan = false;
    result.iteration = struct();

    if baseline.allSignificant
        fprintf('Baseline result: Full is significantly better on all 3 scenarios. Stopping.\n');
        result.status = 'success_stop';
        result.retainedFullModel = fullCurrent;
        result.shouldContinueLoop = false;
    else
        fprintf('Baseline result: Full is NOT significantly better on all 3 scenarios.\n');
        fprintf('Running one focused ablation iteration on Full checkpoint choice...\n\n');

        result.iterationRan = true;
        iteration = runFocusedIteration(fullCurrent, noCrossScaleState, baseline, settings);
        result.iteration = iteration;
        result.retainedFullModel = iteration.retainedFullModel;
        result.shouldContinueLoop = iteration.evidenceImproved;

        if iteration.evidenceImproved
            result.status = 'improved_continue';
        else
            result.status = 'no_improvement_stop';
        end
    end

    jsonOut = fullfile(outputsDir, 'latest_result.json');
    fid = fopen(jsonOut, 'w');
    if fid ~= -1
        fwrite(fid, jsonencode(result, 'PrettyPrint', true), 'char');
        fclose(fid);
    end

    matOut = fullfile(outputsDir, 'latest_result.mat');
    save(matOut, 'result');

    fprintf('\nSaved result JSON: %s\n', jsonOut);
    fprintf('Saved result MAT: %s\n', matOut);
end

function comp = runFullVsNoCross(fullModelPath, nocrossModelPath, settings)
    comp = struct();
    comp.fullModelPath = fullModelPath;
    comp.nocrossModelPath = nocrossModelPath;
    scenarioTemplate = struct( ...
        'scenarioIdx', [], ...
        'dangerZones', [], ...
        'fullPathLengths', [], ...
        'noCrossScaleStatePathLengths', [], ...
        'meanFull', [], ...
        'meanNoCrossScaleState', [], ...
        'deltaFullMinusNoCross', [], ...
        'tStat', [], ...
        'df', [], ...
        'pValue', [], ...
        'fullBetter', [], ...
        'significant', [], ...
        'fullSignificantBetter', []);
    comp.scenario = repmat(scenarioTemplate, numel(settings.scenarios), 1);

    for scenarioIdx = 1:numel(settings.scenarios)
        numDangerZones = settings.scenarios(scenarioIdx);
        fprintf('\n--- Scenario %d (%d danger zones) ---\n', scenarioIdx, numDangerZones);

        [terrainGrid, terrainX, terrainY] = generateFixedTerrain( ...
            settings.mapSize, scenarioIdx, settings.terrainFile);
        dangerZones = generateDangerZones(numDangerZones, settings.mapSize, terrainGrid, terrainX, terrainY);
        terrainMap = defaultTerrainColormap();

        commonBefore = { ...
            settings.startPoint, ...
            settings.goalPoint, ...
            dangerZones, ...
            terrainGrid, terrainX, terrainY, ...
            settings.mapSize ...
        };

        commonAfter = { ...
            settings.globalPlanInterval, ...
            settings.pathDeviationThreshold, ...
            settings.obstacleChangeThreshold, ...
            settings.timeStep, ...
            settings.totalTime, ...
            terrainMap ...
        };

        fullPathLengths = zeros(settings.numRuns, 1);
        noCrossPathLengths = zeros(settings.numRuns, 1);

        for run = 1:settings.numRuns
            seedBase = run * 1000 + scenarioIdx * 100;

            rng(seedBase + settings.seedOffsets.full);
            fullPathLen = quietPathLength(commonBefore, commonAfter, ...
                'RRSACPSO_Pretrained', fullModelPath, 'per-particle');
            fullPathLengths(run) = fullPathLen;

            rng(seedBase + settings.seedOffsets.nocross);
            noCrossPathLen = quietPathLength(commonBefore, commonAfter, ...
                'RRSACPSO_AblationRun', nocrossModelPath, 'ablation_no_crossq');
            noCrossPathLengths(run) = noCrossPathLen;

            fprintf('  Run %2d/%2d: Full=%.4f | NoCrossScaleState=%.4f\n', ...
                run, settings.numRuns, fullPathLen, noCrossPathLen);
        end

        [tStat, pValue, df] = pairedTTest(fullPathLengths, noCrossPathLengths);
        meanFull = mean(fullPathLengths);
        meanNoCross = mean(noCrossPathLengths);
        fullBetter = meanFull < meanNoCross;
        significant = pValue < settings.alpha;

        comp.scenario(scenarioIdx) = struct( ...
            'scenarioIdx', scenarioIdx, ...
            'dangerZones', numDangerZones, ...
            'fullPathLengths', fullPathLengths, ...
            'noCrossScaleStatePathLengths', noCrossPathLengths, ...
            'meanFull', meanFull, ...
            'meanNoCrossScaleState', meanNoCross, ...
            'deltaFullMinusNoCross', meanFull - meanNoCross, ...
            'tStat', tStat, ...
            'df', df, ...
            'pValue', pValue, ...
            'fullBetter', fullBetter, ...
            'significant', significant, ...
            'fullSignificantBetter', fullBetter && significant);

        verdict = 'not significant';
        if fullBetter && significant
            verdict = 'FULL significantly better';
        elseif (~fullBetter) && significant
            verdict = 'NoCrossScaleState significantly better';
        end

        fprintf('  Scenario %d summary: meanFull=%.4f, meanNoCross=%.4f, delta=%.4f, p=%.6f -> %s\n', ...
            scenarioIdx, meanFull, meanNoCross, meanFull - meanNoCross, pValue, verdict);
    end
end

function iteration = runFocusedIteration(currentFull, nocrossPath, baseline, settings)
    % Minimal candidate set: only Full checkpoint variant swaps.
    candidates = {
        currentFull;
        '/Users/hust-hwashin621m/Desktop/vietanhpaper-2/models/RRSACPSO/checkpoints/rrsacpso_perparticle_ep250.mat';
        '/Users/hust-hwashin621m/Desktop/vietanhpaper-2/models/RRSACPSO/checkpoints/rrsacpso_perparticle_ep200.mat';
        '/Users/hust-hwashin621m/Desktop/vietanhpaper-2/models/RRSACPSO/notlog/rrsacpso_perparticle.mat'
    };
    candidates = unique(candidates(cellfun(@isfile, candidates)));

    % Pick the weakest scenario from baseline to isolate change effect.
    sigBetter = [baseline.scenario.fullSignificantBetter];
    if any(~sigBetter)
        unresolved = find(~sigBetter, 1, 'first');
    else
        [~, unresolved] = max([baseline.scenario.pValue]);
    end

    targetScenario = unresolved;
    targetDangerZones = settings.scenarios(targetScenario);

    fprintf('Ablation target scenario: %d (%d danger zones)\n', targetScenario, targetDangerZones);
    fprintf('Candidates to ablate (%d):\n', numel(candidates));
    for i = 1:numel(candidates)
        fprintf('  [%d] %s\n', i, candidates{i});
    end

    ablation = struct();
    ablation.targetScenario = targetScenario;
    ablation.targetDangerZones = targetDangerZones;
    ablation.runs = 3;
    entryTemplate = struct( ...
        'candidate', '', ...
        'meanCandidate', [], ...
        'meanBaselineFull', [], ...
        'deltaCandidateMinusBaseline', [], ...
        'tStat', [], ...
        'df', [], ...
        'pValue', [], ...
        'supported', false);
    ablation.entries = repmat(entryTemplate, numel(candidates), 1);

    baselineFullVec = runSingleModelScenario(currentFull, targetScenario, targetDangerZones, ablation.runs, settings, settings.seedOffsets.full);

    bestCandidate = currentFull;
    bestDelta = 0;
    bestSupported = false;

    for i = 1:numel(candidates)
        cPath = candidates{i};
        cVec = runSingleModelScenario(cPath, targetScenario, targetDangerZones, ablation.runs, settings, settings.seedOffsets.full + 10);

        [tStat, pValue, df] = pairedTTest(cVec, baselineFullVec);
        meanC = mean(cVec);
        meanB = mean(baselineFullVec);
        delta = meanC - meanB;

        supported = (meanC < meanB) && (pValue < settings.alpha);
        ablation.entries(i) = struct( ...
            'candidate', cPath, ...
            'meanCandidate', meanC, ...
            'meanBaselineFull', meanB, ...
            'deltaCandidateMinusBaseline', delta, ...
            'tStat', tStat, ...
            'df', df, ...
            'pValue', pValue, ...
            'supported', supported);

        fprintf('  Candidate %d: delta=%.4f p=%.6f -> %s\n', ...
            i, delta, pValue, ternary(supported, 'SUPPORTED', 'not supported'));

        if supported && (delta < bestDelta)
            bestCandidate = cPath;
            bestDelta = delta;
            bestSupported = true;
        end
    end

    retained = currentFull;
    if bestSupported
        retained = bestCandidate;
    end

    rerun = runFullVsNoCross(retained, nocrossPath, settings);
    rerun.allSignificant = all([rerun.scenario.fullSignificantBetter]);

    oldScore = evidenceScore(baseline);
    newScore = evidenceScore(rerun);
    improved = newScore > oldScore;

    iteration = struct();
    iteration.testedChangeSet = 'Full checkpoint selection only (no algorithm code changes)';
    iteration.ablation = ablation;
    iteration.supportedCandidateFound = bestSupported;
    iteration.retainedFullModel = retained;
    iteration.rerun = rerun;
    iteration.evidenceScoreBefore = oldScore;
    iteration.evidenceScoreAfter = newScore;
    iteration.evidenceImproved = improved;
end

function score = evidenceScore(comp)
    sigWins = sum([comp.scenario.fullSignificantBetter]);
    meanDelta = mean([comp.scenario.deltaFullMinusNoCross]);
    score = sigWins * 100 - meanDelta; % more sig wins, lower (more negative) delta is better
end

function v = runSingleModelScenario(modelPath, scenarioIdx, numDangerZones, numRuns, settings, seedOffset)
    [terrainGrid, terrainX, terrainY] = generateFixedTerrain( ...
        settings.mapSize, scenarioIdx, settings.terrainFile);
    dangerZones = generateDangerZones(numDangerZones, settings.mapSize, terrainGrid, terrainX, terrainY);
    terrainMap = defaultTerrainColormap();

    commonBefore = { ...
        settings.startPoint, settings.goalPoint, dangerZones, ...
        terrainGrid, terrainX, terrainY, settings.mapSize ...
    };
    commonAfter = { ...
        settings.globalPlanInterval, settings.pathDeviationThreshold, ...
        settings.obstacleChangeThreshold, settings.timeStep, settings.totalTime, terrainMap ...
    };

    v = zeros(numRuns, 1);
    for run = 1:numRuns
        seedBase = run * 1000 + scenarioIdx * 100;
        rng(seedBase + seedOffset);
        pathLen = quietPathLength(commonBefore, commonAfter, ...
            'RRSACPSO_Pretrained', modelPath, 'per-particle');
        v(run) = pathLen;
    end
end

function pathLen = quietPathLength(commonBefore, commonAfter, algName, modelPath, mode)
    cmd = '[~, pathLen] = directGlobalPlanning(commonBefore{:}, algName, modelPath, mode, commonAfter{:});';
    evalc(cmd); %#ok<EVLC>
end

function out = defaultTerrainColormap()
    out = [
        0.2 0.4 0.1;
        0.3 0.6 0.2;
        0.6 0.5 0.3;
        0.7 0.6 0.4;
        0.8 0.7 0.6;
        0.9 0.9 0.9
    ];
end

function s = ternary(cond, a, b)
    if cond
        s = a;
    else
        s = b;
    end
end
