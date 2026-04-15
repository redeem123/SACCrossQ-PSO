function result = run_afsacpso_autosearch_loop()
%RUN_APEXPSO_AUTOSEARCH_LOOP Search SAC-level 3-component candidates.
%
% Search policy:
%   1) Screen delta-prior and temporal state encoders with RankResidual + SimBa.
%   2) Promote the strongest pilot candidates to a larger mid-stage run.
%   3) If a candidate clears the mid-stage gate, launch the full 30x1000x100
%      fair-seeded validation automatically.
%   4) Continue expanding the neighborhood until the objective is met.
%
% Objective:
%   Full model must be significantly better than:
%     - LOO-NoCrossScaleState
%     - LOO-NoRankResidualControl
%     - LOO-NoSimBaBackbone

    clc;

    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    cd(repoRoot);
    addpath(repoRoot, '-begin');
    addpath(genpath(fullfile(repoRoot, 'algorithms')), '-begin');
    addpath(genpath(fullfile(repoRoot, 'shared')), '-begin');
    addpath(genpath(fullfile(repoRoot, 'scripts')), '-begin');
    addpath(fullfile(repoRoot, 'data'), '-begin');

    timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss'));
    requestedOutputDir = strtrim(getenv('APEXPSO_AUTOSEARCH_OUTPUT_DIR'));
    if isempty(requestedOutputDir)
        rootOutputDir = fullfile(repoRoot, 'outputs', 'research', ['autosearch_' timestamp]);
    else
        rootOutputDir = requestedOutputDir;
    end
    ensureDir(rootOutputDir);

    candidates = initialCandidates();
    seenIds = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    leaderboard = table();
    objectiveMet = false;
    bestResult = struct();
    bestScore = -inf;
    roundIdx = 1;

    while ~objectiveMet
        fprintf('\n=== AutoSearch Round %d ===\n', roundIdx);

        roundResults = cell(0, 1);
        for i = 1:numel(candidates)
            candidate = candidates(i);
            if isKey(seenIds, candidate.id)
                continue;
            end
            seenIds(candidate.id) = true;

            fprintf('Evaluating candidate %s\n', candidate.id);
            evalResult = evaluateCandidate(repoRoot, rootOutputDir, candidate, ...
                roundIdx, 'pilot', 5, 120, 40);
            roundResults{end + 1, 1} = evalResult; %#ok<AGROW>
            leaderboard = appendLeaderboard(leaderboard, evalResult);

            if evalResult.score > bestScore
                bestScore = evalResult.score;
                bestResult = evalResult;
            end
            if evalResult.objectiveMet
                objectiveMet = true;
            end
            writeProgress(rootOutputDir, timestamp, roundIdx, objectiveMet, bestResult, bestScore, leaderboard);

            if objectiveMet
                break;
            end
        end

        writeLeaderboard(leaderboard, fullfile(rootOutputDir, sprintf('leaderboard_round%d.csv', roundIdx)));

        if objectiveMet
            break;
        end

        if isempty(roundResults)
            candidates = diversifyCandidates(bestResult, roundIdx);
            roundIdx = roundIdx + 1;
            continue;
        end

        promoted = promoteCandidates(roundResults, 2);
        midResults = cell(0, 1);
        for i = 1:numel(promoted)
            candidate = promoted(i);
            evalResult = evaluateCandidate(repoRoot, rootOutputDir, candidate, ...
                roundIdx, 'mid', 10, 300, 40);
            midResults{end + 1, 1} = evalResult; %#ok<AGROW>
            leaderboard = appendLeaderboard(leaderboard, evalResult);

            if evalResult.score > bestScore
                bestScore = evalResult.score;
                bestResult = evalResult;
            end
            if evalResult.objectiveMet
                objectiveMet = true;
            end
            writeProgress(rootOutputDir, timestamp, roundIdx, objectiveMet, bestResult, bestScore, leaderboard);

            if objectiveMet
                break;
            end
        end

        writeLeaderboard(leaderboard, fullfile(rootOutputDir, sprintf('leaderboard_round%d_mid.csv', roundIdx)));

        if objectiveMet
            break;
        end

        if ~isempty(midResults)
            midScores = cellfun(@(x) x.score, midResults);
            [~, bestIdx] = max(midScores);
            bestMid = midResults{bestIdx};
            if qualifiesForFinal(bestMid)
                finalResult = evaluateCandidate(repoRoot, rootOutputDir, bestMid.candidate, ...
                    roundIdx, 'final', 30, 1000, 100);
                leaderboard = appendLeaderboard(leaderboard, finalResult);
                if finalResult.score > bestScore
                    bestScore = finalResult.score;
                    bestResult = finalResult;
                end
                objectiveMet = finalResult.objectiveMet;
                writeProgress(rootOutputDir, timestamp, roundIdx, objectiveMet, bestResult, bestScore, leaderboard);
                writeLeaderboard(leaderboard, fullfile(rootOutputDir, sprintf('leaderboard_round%d_final.csv', roundIdx)));
                if objectiveMet
                    break;
                end
            end

            candidates = refineCandidates(bestMid.candidate);
        else
            candidates = diversifyCandidates(bestResult, roundIdx);
        end

        roundIdx = roundIdx + 1;
    end

    result = struct();
    result.timestamp = timestamp;
    result.rootOutputDir = rootOutputDir;
    result.objectiveMet = objectiveMet;
    result.bestResult = bestResult;
    result.bestScore = bestScore;

    save(fullfile(rootOutputDir, 'autosearch_result.mat'), 'result');
    writejson(fullfile(rootOutputDir, 'autosearch_result.json'), ...
        makeSerializableStatus(timestamp, rootOutputDir, roundIdx, objectiveMet, bestResult, bestScore, leaderboard));

    if objectiveMet
        markObjectiveMet(rootOutputDir, bestResult);
        fprintf('\nObjective met. Best candidate: %s\n', safeBestCandidateId(bestResult));
    else
        fprintf('\nObjective not met. Best candidate so far: %s\n', safeBestCandidateId(bestResult));
    end
end

function candidates = initialCandidates()
    candidates = [
        makeCandidate('delta18_flat_a96_b2_c160_b2', 'delta18', 18, false, 96, 2, 160, 2)
        makeCandidate('delta18_branch_a96_b2_c160_b2', 'delta18', 18, true, 96, 2, 160, 2)
        makeCandidate('delta18_flat_a128_b3_c192_b3', 'delta18', 18, false, 128, 3, 192, 3)
        makeCandidate('delta27_flat_a96_b2_c160_b2', 'delta27', 27, false, 96, 2, 160, 2)
        makeCandidate('delta27_branch_a96_b2_c160_b2', 'delta27', 27, true, 96, 2, 160, 2)
        makeCandidate('delta27_flat_a128_b3_c192_b3', 'delta27', 27, false, 128, 3, 192, 3)
        makeCandidate('delta36_flat_a96_b2_c160_b2', 'delta36', 36, false, 96, 2, 160, 2)
        makeCandidate('delta36_branch_a96_b2_c160_b2', 'delta36', 36, true, 96, 2, 160, 2)
        makeCandidate('delta36_flat_a128_b3_c192_b3', 'delta36', 36, false, 128, 3, 192, 3)
        makeCandidate('delta45_flat_a96_b2_c160_b2', 'delta45', 45, false, 96, 2, 160, 2)
        makeCandidate('delta45_branch_a96_b2_c160_b2', 'delta45', 45, true, 96, 2, 160, 2)
        makeCandidate('temporal45_flat_a96_b2_c160_b2', 'temporalstats45', 45, false, 96, 2, 160, 2)
        makeCandidate('temporal45_branch_a96_b2_c160_b2', 'temporalstats45', 45, true, 96, 2, 160, 2)
    ];
end

function candidate = makeCandidate(id, encoderMode, stateSize, useCrossScaleBranching, actorWidth, actorBlocks, criticWidth, criticBlocks)
    candidate = struct( ...
        'id', id, ...
        'encoderMode', encoderMode, ...
        'stateSize', stateSize, ...
        'useCrossScaleBranching', logical(useCrossScaleBranching), ...
        'simbaActorWidth', actorWidth, ...
        'simbaActorBlocks', actorBlocks, ...
        'simbaCriticWidth', criticWidth, ...
        'simbaCriticBlocks', criticBlocks);
end

function evalResult = evaluateCandidate(repoRoot, rootOutputDir, candidate, roundIdx, stageName, numRuns, maxIterations, popSize)
    outputDir = fullfile(rootOutputDir, sprintf('round%d_%s_%s', roundIdx, stageName, candidate.id));
    ensureDir(outputDir);

    setenv('APEXPSO_RESEARCH_PARALLEL', '1');
    setenv('APEXPSO_RESEARCH_WORKERS', '14');
    setenv('APEXPSO_RESEARCH_VARIANTS', 'apex_full,loo_no_crossscale_state,loo_no_rank_residual_control,loo_no_simba_backbone');
    setenv('APEXPSO_RESEARCH_OUTPUT_DIR', outputDir);
    setenv('APEXPSO_RESEARCH_POP_SIZE', num2str(popSize));
    setenv('APEXPSO_RESEARCH_USE_CROSSSCALE_STATE', '1');
    setenv('APEXPSO_RESEARCH_USE_CROSSSCALE_BRANCHING', num2str(double(candidate.useCrossScaleBranching)));
    setenv('APEXPSO_RESEARCH_USE_RESIDUAL_CRITIC', '0');
    setenv('APEXPSO_RESEARCH_USE_PRIORITIZED_REPLAY', '0');
    setenv('APEXPSO_RESEARCH_STATE_SIZE', num2str(candidate.stateSize));
    setenv('APEXPSO_RESEARCH_ENCODER_MODE', candidate.encoderMode);
    setenv('APEXPSO_RESEARCH_SIMBA_ACTOR_WIDTH', num2str(candidate.simbaActorWidth));
    setenv('APEXPSO_RESEARCH_SIMBA_ACTOR_BLOCKS', num2str(candidate.simbaActorBlocks));
    setenv('APEXPSO_RESEARCH_SIMBA_CRITIC_WIDTH', num2str(candidate.simbaCriticWidth));
    setenv('APEXPSO_RESEARCH_SIMBA_CRITIC_BLOCKS', num2str(candidate.simbaCriticBlocks));

    run_afsacpso_research_cycle(numRuns, maxIterations);

    summaryPath = fullfile(outputDir, 'afsacpso_variant_summary.csv');
    summary = readtable(summaryPath);
    coreRows = summary(strcmp(summary.Family, 'core_loo'), :);

    winnerIsFull = strcmp(coreRows.Winner, 'APEX-Full');
    sigWins = double(coreRows.FullSignificantlyBetter);
    deltas = coreRows.MeanDeltaFullMinusVariant;
    score = sum(-deltas) + 25 * sum(winnerIsFull) + 250 * sum(sigWins);

    evalResult = struct();
    evalResult.candidate = candidate;
    evalResult.outputDir = outputDir;
    evalResult.round = roundIdx;
    evalResult.stage = stageName;
    evalResult.numRuns = numRuns;
    evalResult.maxIterations = maxIterations;
    evalResult.popSize = popSize;
    evalResult.summaryPath = summaryPath;
    evalResult.fullBetterCount = sum(winnerIsFull);
    evalResult.significantCount = sum(sigWins);
    evalResult.score = score;
    evalResult.objectiveMet = all(sigWins == 1);
    evalResult.coreSummary = coreRows;
end

function promoted = promoteCandidates(results, limit)
    if isempty(results)
        promoted = struct([]);
        return;
    end
    if iscell(results)
        results = [results{:}]';
    end
    [~, order] = sort([results.score], 'descend');
    keep = order(1:min(limit, numel(order)));
    promoted = [results(keep).candidate]';
end

function flag = qualifiesForFinal(evalResult)
    rows = evalResult.coreSummary;
    winners = strcmp(rows.Winner, 'APEX-Full');
    flag = sum(winners) >= 6 && sum(rows.FullSignificantlyBetter) >= 1;
end

function candidates = refineCandidates(bestCandidate)
    actorWidthChoices = unique([max(64, bestCandidate.simbaActorWidth - 32), ...
        bestCandidate.simbaActorWidth, bestCandidate.simbaActorWidth + 32]);
    criticWidthChoices = unique([max(128, bestCandidate.simbaCriticWidth - 32), ...
        bestCandidate.simbaCriticWidth, bestCandidate.simbaCriticWidth + 32]);
    actorBlockChoices = unique([max(2, bestCandidate.simbaActorBlocks - 1), ...
        bestCandidate.simbaActorBlocks, min(4, bestCandidate.simbaActorBlocks + 1)]);
    criticBlockChoices = unique([max(2, bestCandidate.simbaCriticBlocks - 1), ...
        bestCandidate.simbaCriticBlocks, min(4, bestCandidate.simbaCriticBlocks + 1)]);

    modeOrder = {'delta18', 'delta27', 'delta36', 'delta45', 'temporalstats45'};
    modeIdx = find(strcmp(modeOrder, bestCandidate.encoderMode), 1);
    modeIdx = max(1, min(numel(modeOrder), modeIdx));
    modeChoices = unique(modeOrder(max(1, modeIdx - 1):min(numel(modeOrder), modeIdx + 1)));

    candidates = struct([]);
    for i = 1:numel(modeChoices)
        stateSize = modeToStateSize(modeChoices{i});
        candidates = appendStructItem(candidates, makeCandidate( ...
            makeCandidateId(modeChoices{i}, bestCandidate.useCrossScaleBranching, ...
                bestCandidate.simbaActorWidth, bestCandidate.simbaActorBlocks, ...
                bestCandidate.simbaCriticWidth, bestCandidate.simbaCriticBlocks), ...
            modeChoices{i}, stateSize, bestCandidate.useCrossScaleBranching, ...
            bestCandidate.simbaActorWidth, bestCandidate.simbaActorBlocks, ...
            bestCandidate.simbaCriticWidth, bestCandidate.simbaCriticBlocks));

        for aw = actorWidthChoices
            candidates = appendStructItem(candidates, makeCandidate( ...
                makeCandidateId(modeChoices{i}, bestCandidate.useCrossScaleBranching, ...
                    aw, bestCandidate.simbaActorBlocks, ...
                    bestCandidate.simbaCriticWidth, bestCandidate.simbaCriticBlocks), ...
                modeChoices{i}, stateSize, bestCandidate.useCrossScaleBranching, ...
                aw, bestCandidate.simbaActorBlocks, ...
                bestCandidate.simbaCriticWidth, bestCandidate.simbaCriticBlocks));
        end
        for cw = criticWidthChoices
            candidates = appendStructItem(candidates, makeCandidate( ...
                makeCandidateId(modeChoices{i}, bestCandidate.useCrossScaleBranching, ...
                    bestCandidate.simbaActorWidth, bestCandidate.simbaActorBlocks, ...
                    cw, bestCandidate.simbaCriticBlocks), ...
                modeChoices{i}, stateSize, bestCandidate.useCrossScaleBranching, ...
                bestCandidate.simbaActorWidth, bestCandidate.simbaActorBlocks, ...
                cw, bestCandidate.simbaCriticBlocks));
        end
        for ab = actorBlockChoices
            candidates = appendStructItem(candidates, makeCandidate( ...
                makeCandidateId(modeChoices{i}, bestCandidate.useCrossScaleBranching, ...
                    bestCandidate.simbaActorWidth, ab, ...
                    bestCandidate.simbaCriticWidth, bestCandidate.simbaCriticBlocks), ...
                modeChoices{i}, stateSize, bestCandidate.useCrossScaleBranching, ...
                bestCandidate.simbaActorWidth, ab, ...
                bestCandidate.simbaCriticWidth, bestCandidate.simbaCriticBlocks));
        end
        for cb = criticBlockChoices
            candidates = appendStructItem(candidates, makeCandidate( ...
                makeCandidateId(modeChoices{i}, bestCandidate.useCrossScaleBranching, ...
                    bestCandidate.simbaActorWidth, bestCandidate.simbaActorBlocks, ...
                    bestCandidate.simbaCriticWidth, cb), ...
                modeChoices{i}, stateSize, bestCandidate.useCrossScaleBranching, ...
                bestCandidate.simbaActorWidth, bestCandidate.simbaActorBlocks, ...
                bestCandidate.simbaCriticWidth, cb));
        end
    end
    candidates = dedupeCandidates(candidates);
end

function candidates = diversifyCandidates(bestResult, roundIdx)
    if nargin < 2
        roundIdx = 1;
    end
    if isempty(fieldnames(bestResult))
        candidates = initialCandidates();
        return;
    end

    bestCandidate = bestResult.candidate;
    widthRadius = 32 * max(1, roundIdx);
    actorWidthChoices = unique([96, 128, bestCandidate.simbaActorWidth, ...
        max(64, bestCandidate.simbaActorWidth - widthRadius), ...
        bestCandidate.simbaActorWidth + widthRadius]);
    criticWidthChoices = unique([160, 192, bestCandidate.simbaCriticWidth, ...
        max(128, bestCandidate.simbaCriticWidth - widthRadius), ...
        bestCandidate.simbaCriticWidth + widthRadius]);
    actorBlockChoices = unique([2, bestCandidate.simbaActorBlocks, min(4, bestCandidate.simbaActorBlocks + 1)]);
    criticBlockChoices = unique([2, bestCandidate.simbaCriticBlocks, min(4, bestCandidate.simbaCriticBlocks + 1)]);
    branchChoices = unique([false, true, bestCandidate.useCrossScaleBranching]);
    modeChoices = {'delta18', 'delta27', 'delta36', 'delta45', 'temporalstats45'};

    candidates = struct([]);
    for i = 1:numel(modeChoices)
        stateSize = modeToStateSize(modeChoices{i});
        for branch = branchChoices
            candidates = appendStructItem(candidates, makeCandidate( ...
                makeCandidateId(modeChoices{i}, branch, bestCandidate.simbaActorWidth, ...
                    bestCandidate.simbaActorBlocks, bestCandidate.simbaCriticWidth, bestCandidate.simbaCriticBlocks), ...
                modeChoices{i}, stateSize, branch, bestCandidate.simbaActorWidth, ...
                bestCandidate.simbaActorBlocks, bestCandidate.simbaCriticWidth, ...
                bestCandidate.simbaCriticBlocks));
        end
    end

    stateSize = modeToStateSize(bestCandidate.encoderMode);
    for aw = actorWidthChoices
        for cw = criticWidthChoices
            candidates = appendStructItem(candidates, makeCandidate( ...
                makeCandidateId(bestCandidate.encoderMode, bestCandidate.useCrossScaleBranching, ...
                    aw, bestCandidate.simbaActorBlocks, cw, bestCandidate.simbaCriticBlocks), ...
                bestCandidate.encoderMode, stateSize, bestCandidate.useCrossScaleBranching, ...
                aw, bestCandidate.simbaActorBlocks, cw, bestCandidate.simbaCriticBlocks));
        end
    end
    for ab = actorBlockChoices
        for cb = criticBlockChoices
            candidates = appendStructItem(candidates, makeCandidate( ...
                makeCandidateId(bestCandidate.encoderMode, bestCandidate.useCrossScaleBranching, ...
                    bestCandidate.simbaActorWidth, ab, bestCandidate.simbaCriticWidth, cb), ...
                bestCandidate.encoderMode, stateSize, bestCandidate.useCrossScaleBranching, ...
                bestCandidate.simbaActorWidth, ab, bestCandidate.simbaCriticWidth, cb));
        end
    end
    candidates = dedupeCandidates(candidates);
end

function out = dedupeCandidates(candidates)
    if isempty(candidates)
        out = candidates;
        return;
    end
    ids = string({candidates.id});
    [~, keep] = unique(ids, 'stable');
    out = candidates(sort(keep));
end

function id = makeCandidateId(modeName, useCrossScaleBranching, actorWidth, actorBlocks, criticWidth, criticBlocks)
    if useCrossScaleBranching
        layoutTag = 'branch';
    else
        layoutTag = 'flat';
    end
    id = sprintf('%s_%s_a%d_b%d_c%d_b%d', modeName, layoutTag, ...
        actorWidth, actorBlocks, criticWidth, criticBlocks);
end

function out = appendLeaderboard(leaderboard, evalResult)
    row = table( ...
        string(evalResult.candidate.id), ...
        string(evalResult.candidate.encoderMode), ...
        double(evalResult.candidate.useCrossScaleBranching), ...
        string(evalResult.stage), ...
        evalResult.numRuns, ...
        evalResult.maxIterations, ...
        evalResult.popSize, ...
        evalResult.fullBetterCount, ...
        evalResult.significantCount, ...
        evalResult.score, ...
        string(evalResult.outputDir), ...
        'VariableNames', {'Candidate', 'EncoderMode', 'UseCrossScaleBranching', ...
            'Stage', 'Runs', 'MaxIterations', 'PopSize', 'FullBetterCount', ...
            'SignificantCount', 'Score', 'OutputDir'});
    if isempty(leaderboard) || width(leaderboard) == 0
        out = row;
    else
        out = [leaderboard; row];
    end
end

function out = appendStructItem(items, item)
    if isempty(items) || isempty(fieldnames(items))
        out = item;
    else
        out = [items; item];
    end
end

function writeProgress(rootOutputDir, timestamp, roundIdx, objectiveMet, bestResult, bestScore, leaderboard)
    writeLeaderboard(leaderboard, fullfile(rootOutputDir, 'leaderboard_live.csv'));
    writejson(fullfile(rootOutputDir, 'autosearch_status.json'), ...
        makeSerializableStatus(timestamp, rootOutputDir, roundIdx, objectiveMet, bestResult, bestScore, leaderboard));
end

function writeLeaderboard(leaderboard, path)
    writetable(leaderboard, path);
end

function ensureDir(pathStr)
    if exist(pathStr, 'dir') ~= 7
        mkdir(pathStr);
    end
end

function writejson(pathStr, value)
    fid = fopen(pathStr, 'w');
    if fid == -1
        return;
    end
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid, jsonencode(value, 'PrettyPrint', true), 'char');
end

function status = makeSerializableStatus(timestamp, rootOutputDir, roundIdx, objectiveMet, bestResult, bestScore, leaderboard)
    status = struct();
    status.timestamp = timestamp;
    status.rootOutputDir = rootOutputDir;
    status.round = roundIdx;
    status.objectiveMet = objectiveMet;
    status.bestCandidateId = safeBestCandidateId(bestResult);
    status.bestStage = safeBestStage(bestResult);
    status.bestScore = bestScore;
    status.numLeaderboardRows = height(leaderboard);
    status.leaderboardLivePath = fullfile(rootOutputDir, 'leaderboard_live.csv');
end

function markObjectiveMet(rootOutputDir, bestResult)
    flagPath = fullfile(rootOutputDir, 'objective_met.flag');
    fid = fopen(flagPath, 'w');
    if fid == -1
        return;
    end
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, 'Objective met at %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
    fprintf(fid, 'Best candidate: %s\n', safeBestCandidateId(bestResult));
end

function id = safeBestCandidateId(bestResult)
    if isempty(fieldnames(bestResult))
        id = '';
    else
        id = char(string(bestResult.candidate.id));
    end
end

function stage = safeBestStage(bestResult)
    if isempty(fieldnames(bestResult))
        stage = '';
    else
        stage = char(string(bestResult.stage));
    end
end

function stateSize = modeToStateSize(modeName)
    switch lower(modeName)
        case 'delta18'
            stateSize = 18;
        case 'delta27'
            stateSize = 27;
        case 'delta36'
            stateSize = 36;
        case {'delta45', 'temporalstats45'}
            stateSize = 45;
        otherwise
            error('Unsupported mode: %s', modeName);
    end
end
