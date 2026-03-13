function performTTest(allRunResults, algorithms, scenarioIdx)
    % Perform pairwise inferential statistics on per-run global fitness.
    % Primary endpoint: fitnessComponents.totalFitness.
    % Fallback order: actualBestFitness -> pathLength.

    fprintf('\n=== STATISTICAL ANALYSIS FOR SCENARIO %d ===\n', scenarioIdx);

    numRuns = length(allRunResults);
    numAlgorithms = length(algorithms);

    [metricMatrix, metricSourceMatrix, metricName, sourceCounts] = ...
        extractMetricMatrix(allRunResults, algorithms);
    algorithmNames = getAlgorithmNames(algorithms);

    fprintf('Metric tested: %s\n', metricName);
    fprintf('Runs available: %d\n', numRuns);

    if sourceCounts.actualBestFitness > 0 || sourceCounts.pathLength > 0
        warning('performTTest:MetricFallback', ...
            ['Scenario %d metric fallback used for %d/%d values ', ...
             '(actualBestFitness=%d, pathLength=%d).'], ...
            scenarioIdx, ...
            sourceCounts.actualBestFitness + sourceCounts.pathLength, ...
            sourceCounts.totalFitness + sourceCounts.actualBestFitness + sourceCounts.pathLength, ...
            sourceCounts.actualBestFitness, ...
            sourceCounts.pathLength);
    end

    summaryResults = repmat(emptySummaryResult(), numAlgorithms, 1);
    meanValues = NaN(numAlgorithms, 1);

    fprintf('\n=== DESCRIPTIVE STATISTICS (%s) ===\n', metricName);
    fprintf('%-25s | %-7s | %-10s | %-10s | %-10s\n', 'Algorithm', 'N', 'Mean', 'Std', 'Median');
    fprintf('------------------------------------------------------------------------\n');
    for algIdx = 1:numAlgorithms
        values = metricMatrix(:, algIdx);
        validValues = values(isfinite(values));

        meanValues(algIdx) = safeMean(validValues);

        summaryResults(algIdx).scenario_id = scenarioIdx;
        summaryResults(algIdx).metric_name = metricName;
        summaryResults(algIdx).algorithm = algorithmNames{algIdx};
        summaryResults(algIdx).n_valid = numel(validValues);
        summaryResults(algIdx).mean = safeMean(validValues);
        summaryResults(algIdx).std = safeStd(validValues);
        summaryResults(algIdx).median = safeMedian(validValues);
        summaryResults(algIdx).min = safeMin(validValues);
        summaryResults(algIdx).max = safeMax(validValues);

        fprintf('%-25s | %7d | %10.4f | %10.4f | %10.4f\n', ...
            summaryResults(algIdx).algorithm, ...
            summaryResults(algIdx).n_valid, ...
            summaryResults(algIdx).mean, ...
            summaryResults(algIdx).std, ...
            summaryResults(algIdx).median);
    end

    pairIndices = zeros(0, 2);
    if numAlgorithms >= 2
        pairIndices = nchoosek(1:numAlgorithms, 2);
    end

    numPairs = size(pairIndices, 1);
    pairwiseResults = repmat(emptyPairwiseResult(), numPairs, 1);
    rawPT = ones(numPairs, 1);
    rawPWilcoxon = ones(numPairs, 1);

    fprintf('\n=== PAIRWISE INFERENCE (PAIRED T + WILCOXON) ===\n');
    if numPairs == 0
        fprintf('Not enough algorithms for pairwise comparisons.\n');
    else
        fprintf('%-25s vs %-25s | N | meanDiff | p_t(raw) | p_w(raw)\n', 'Algorithm 1', 'Algorithm 2');
        fprintf('-------------------------------------------------------------------------------------------\n');
    end

    for pairIdx = 1:numPairs
        i = pairIndices(pairIdx, 1);
        j = pairIndices(pairIdx, 2);

        sample1 = metricMatrix(:, i);
        sample2 = metricMatrix(:, j);
        validMask = isfinite(sample1) & isfinite(sample2);
        x = sample1(validMask);
        y = sample2(validMask);
        differences = x - y;

        nPairs = numel(differences);
        meanDiff = safeMean(differences);

        tStat = NaN;
        pRawT = 1;
        ciLow = NaN;
        ciHigh = NaN;
        cohenDz = NaN;
        if nPairs >= 2
            [tStat, pRawT] = pairedTTestOnly(x, y);
            [ciLow, ciHigh] = pairedMeanCI(differences, 0.95);
            cohenDz = pairedCohensDz(differences);
        end

        [wilcoxonW, pRawWilcoxon, rankBiserial] = wilcoxonSignedRankTwoSided(differences);

        pairwiseResults(pairIdx).scenario_id = scenarioIdx;
        pairwiseResults(pairIdx).metric_name = metricName;
        pairwiseResults(pairIdx).algorithm_1 = algorithmNames{i};
        pairwiseResults(pairIdx).algorithm_2 = algorithmNames{j};
        pairwiseResults(pairIdx).n_pairs = nPairs;
        pairwiseResults(pairIdx).mean_1 = safeMean(x);
        pairwiseResults(pairIdx).mean_2 = safeMean(y);
        pairwiseResults(pairIdx).mean_diff = meanDiff;
        pairwiseResults(pairIdx).ci95_low = ciLow;
        pairwiseResults(pairIdx).ci95_high = ciHigh;
        pairwiseResults(pairIdx).t_stat = tStat;
        pairwiseResults(pairIdx).p_raw_t = pRawT;
        pairwiseResults(pairIdx).p_holm_t = NaN;
        pairwiseResults(pairIdx).sig_holm_t = false;
        pairwiseResults(pairIdx).cohen_dz = cohenDz;
        pairwiseResults(pairIdx).wilcoxon_W = wilcoxonW;
        pairwiseResults(pairIdx).p_raw_wilcoxon = pRawWilcoxon;
        pairwiseResults(pairIdx).p_holm_wilcoxon = NaN;
        pairwiseResults(pairIdx).sig_holm_wilcoxon = false;
        pairwiseResults(pairIdx).rank_biserial_r = rankBiserial;

        rawPT(pairIdx) = pRawT;
        rawPWilcoxon(pairIdx) = pRawWilcoxon;

        fprintf('%-25s vs %-25s | %2d | %8.4f | %8.6f | %8.6f\n', ...
            algorithmNames{i}, algorithmNames{j}, nPairs, meanDiff, pRawT, pRawWilcoxon);
    end

    pHolmT = holmBonferroni(rawPT);
    pHolmWilcoxon = holmBonferroni(rawPWilcoxon);

    for pairIdx = 1:numPairs
        pairwiseResults(pairIdx).p_holm_t = pHolmT(pairIdx);
        pairwiseResults(pairIdx).sig_holm_t = pHolmT(pairIdx) < 0.05;
        pairwiseResults(pairIdx).p_holm_wilcoxon = pHolmWilcoxon(pairIdx);
        pairwiseResults(pairIdx).sig_holm_wilcoxon = pHolmWilcoxon(pairIdx) < 0.05;
    end

    winsHolmT = zeros(numAlgorithms, 1);
    for pairIdx = 1:numPairs
        if ~pairwiseResults(pairIdx).sig_holm_t
            continue;
        end

        i = pairIndices(pairIdx, 1);
        j = pairIndices(pairIdx, 2);

        if pairwiseResults(pairIdx).mean_diff < 0
            winsHolmT(i) = winsHolmT(i) + 1;
        elseif pairwiseResults(pairIdx).mean_diff > 0
            winsHolmT(j) = winsHolmT(j) + 1;
        end
    end

    [friedmanChi2, friedmanP, avgRanks, friedmanRuns, rankMatrix] = ...
        computeFriedmanStatistics(metricMatrix);

    friedmanDf = max(numAlgorithms - 1, 0);
    for algIdx = 1:numAlgorithms
        summaryResults(algIdx).average_rank = avgRanks(algIdx);
        summaryResults(algIdx).significant_wins_holm_t = winsHolmT(algIdx);
        summaryResults(algIdx).friedman_chi_square = friedmanChi2;
        summaryResults(algIdx).friedman_df = friedmanDf;
        summaryResults(algIdx).friedman_p = friedmanP;
        summaryResults(algIdx).n_friedman_runs = friedmanRuns;
    end

    fprintf('\n=== HOLM-CORRECTED SIGNIFICANCE SUMMARY ===\n');
    numSigT = sum([pairwiseResults.sig_holm_t]);
    numSigW = sum([pairwiseResults.sig_holm_wilcoxon]);
    fprintf('Significant pairs (Holm-corrected, paired t-test): %d/%d\n', numSigT, numPairs);
    fprintf('Significant pairs (Holm-corrected, Wilcoxon):      %d/%d\n', numSigW, numPairs);

    fprintf('\n=== OVERALL RANKING (MEAN METRIC + HOLM T-TEST WINS) ===\n');
    [sortedMeans, order] = sort(meanValues, 'ascend');
    fprintf('Rank | %-25s | Mean | Holm Wins\n', 'Algorithm');
    fprintf('---------------------------------------------------------------\n');
    for rankIdx = 1:numAlgorithms
        algIdx = order(rankIdx);
        fprintf('%-4d | %-25s | %10.4f | %9d\n', ...
            rankIdx, algorithmNames{algIdx}, sortedMeans(rankIdx), winsHolmT(algIdx));
    end

    if friedmanRuns >= 2 && numAlgorithms >= 2
        fprintf('\nFriedman omnibus (run-level blocks): chi2(%d)=%.4f, p=%.6f (N=%d runs)\n', ...
            friedmanDf, friedmanChi2, friedmanP, friedmanRuns);
    else
        fprintf('\nFriedman omnibus skipped: insufficient complete run blocks.\n');
    end

    results = struct();
    results.scenarioIdx = scenarioIdx;
    results.metricName = metricName;
    results.algorithmNames = algorithmNames;
    results.metricMatrix = metricMatrix;
    results.metricSourceMatrix = metricSourceMatrix;
    results.sourceCounts = sourceCounts;
    results.pairIndices = pairIndices;
    results.pairwiseResults = pairwiseResults;
    results.summaryResults = summaryResults;
    results.holm = struct( ...
        'p_raw_t', rawPT, ...
        'p_holm_t', pHolmT, ...
        'p_raw_wilcoxon', rawPWilcoxon, ...
        'p_holm_wilcoxon', pHolmWilcoxon);
    results.friedman = struct( ...
        'chi_square', friedmanChi2, ...
        'df', friedmanDf, ...
        'p_value', friedmanP, ...
        'n_runs', friedmanRuns, ...
        'rank_matrix', rankMatrix, ...
        'average_ranks', avgRanks);

    saveTTestResults(results, scenarioIdx);
end

function [metricMatrix, metricSourceMatrix, metricName, sourceCounts] = extractMetricMatrix(allRunResults, algorithms)
    metricName = 'global_fitness_total';

    numRuns = length(allRunResults);
    numAlgorithms = length(algorithms);
    metricMatrix = NaN(numRuns, numAlgorithms);
    metricSourceMatrix = cell(numRuns, numAlgorithms);

    sourceCounts = struct( ...
        'totalFitness', 0, ...
        'actualBestFitness', 0, ...
        'pathLength', 0, ...
        'missing', 0);

    for runIdx = 1:numRuns
        for algIdx = 1:numAlgorithms
            algField = algorithms{algIdx}.fieldName;
            metricSourceMatrix{runIdx, algIdx} = 'missing';

            if ~isfield(allRunResults{runIdx}, algField)
                sourceCounts.missing = sourceCounts.missing + 1;
                continue;
            end

            entry = allRunResults{runIdx}.(algField);
            [metricValue, sourceLabel] = extractMetricValue(entry);
            metricMatrix(runIdx, algIdx) = metricValue;
            metricSourceMatrix{runIdx, algIdx} = sourceLabel;

            if isfield(sourceCounts, sourceLabel)
                sourceCounts.(sourceLabel) = sourceCounts.(sourceLabel) + 1;
            else
                sourceCounts.missing = sourceCounts.missing + 1;
            end
        end
    end
end

function [metricValue, sourceLabel] = extractMetricValue(entry)
    metricValue = NaN;
    sourceLabel = 'missing';

    if isfield(entry, 'fitnessComponents') && ~isempty(entry.fitnessComponents)
        components = entry.fitnessComponents;
        if isfield(components, 'totalFitness')
            candidate = toScalar(components.totalFitness);
            if isfinite(candidate)
                metricValue = candidate;
                sourceLabel = 'totalFitness';
                return;
            end
        end
    end

    if isfield(entry, 'actualBestFitness')
        candidate = toScalar(entry.actualBestFitness);
        if isfinite(candidate)
            metricValue = candidate;
            sourceLabel = 'actualBestFitness';
            return;
        end
    end

    if isfield(entry, 'pathLength')
        candidate = toScalar(entry.pathLength);
        if isfinite(candidate)
            metricValue = candidate;
            sourceLabel = 'pathLength';
            return;
        end
    end
end

function [tStat, pValue] = pairedTTestOnly(sample1, sample2)
    [tStat, pValue, ~] = pairedTTest(sample1, sample2);
end

function [ciLow, ciHigh] = pairedMeanCI(differences, confidence)
    ciLow = NaN;
    ciHigh = NaN;

    n = numel(differences);
    if n < 2
        return;
    end

    alpha = 1 - confidence;
    meanDiff = mean(differences);
    stdDiff = std(differences, 0);

    if stdDiff == 0
        ciLow = meanDiff;
        ciHigh = meanDiff;
        return;
    end

    se = stdDiff / sqrt(n);
    tCrit = tCriticalValue(1 - alpha / 2, n - 1);
    halfWidth = tCrit * se;

    ciLow = meanDiff - halfWidth;
    ciHigh = meanDiff + halfWidth;
end

function tCrit = tCriticalValue(targetCDF, df)
    if df <= 0
        tCrit = NaN;
        return;
    end

    low = 0;
    high = 1;
    while tcdf(high, df) < targetCDF
        high = high * 2;
        if high > 1e6
            break;
        end
    end

    for iter = 1:80
        mid = 0.5 * (low + high);
        if tcdf(mid, df) < targetCDF
            low = mid;
        else
            high = mid;
        end
    end

    tCrit = 0.5 * (low + high);
end

function dz = pairedCohensDz(differences)
    if numel(differences) < 2
        dz = NaN;
        return;
    end

    stdDiff = std(differences, 0);
    meanDiff = mean(differences);

    if stdDiff == 0
        if meanDiff == 0
            dz = 0;
        else
            dz = sign(meanDiff) * Inf;
        end
    else
        dz = meanDiff / stdDiff;
    end
end

function [Wplus, pValue, rankBiserial] = wilcoxonSignedRankTwoSided(differences)
    differences = differences(isfinite(differences));
    differences = differences(differences ~= 0);

    if isempty(differences)
        Wplus = 0;
        pValue = 1;
        rankBiserial = 0;
        return;
    end

    absDiff = abs(differences);
    ranks = averageRanks(absDiff(:));
    ranks = ranks(:);

    isPositive = differences(:) > 0;
    isNegative = differences(:) < 0;

    Wplus = sum(ranks(isPositive));
    Wminus = sum(ranks(isNegative));

    n = numel(differences);
    denom = n * (n + 1) / 2;
    rankBiserial = (Wplus - Wminus) / denom;

    if n < 2
        pValue = 1;
        return;
    end

    muW = n * (n + 1) / 4;
    tieCorrection = wilcoxonTieCorrection(absDiff);
    sigmaSq = n * (n + 1) * (2 * n + 1) / 24 - tieCorrection;

    if sigmaSq <= 0
        pValue = 1;
        return;
    end

    sigmaW = sqrt(sigmaSq);
    z = (abs(Wplus - muW) - 0.5) / sigmaW;
    pValue = 2 * (1 - normcdf(z));
    pValue = min(max(pValue, 0), 1);
end

function correction = wilcoxonTieCorrection(absDiff)
    sortedValues = sort(absDiff(:));
    correction = 0;

    if isempty(sortedValues)
        return;
    end

    count = 1;
    for idx = 2:numel(sortedValues)
        if sortedValues(idx) == sortedValues(idx - 1)
            count = count + 1;
        else
            if count > 1
                correction = correction + (count^3 - count) / 48;
            end
            count = 1;
        end
    end

    if count > 1
        correction = correction + (count^3 - count) / 48;
    end
end

function pAdj = holmBonferroni(pRaw)
    if isempty(pRaw)
        pAdj = pRaw;
        return;
    end

    p = pRaw(:);
    p(~isfinite(p)) = 1;

    m = numel(p);
    [pSorted, order] = sort(p, 'ascend');

    adjustedSorted = zeros(m, 1);
    for idx = 1:m
        adjustedSorted(idx) = min(1, pSorted(idx) * (m - idx + 1));
    end

    for idx = 2:m
        adjustedSorted(idx) = max(adjustedSorted(idx), adjustedSorted(idx - 1));
    end

    pAdj = zeros(m, 1);
    pAdj(order) = adjustedSorted;
end

function [chiSquare, pValue, avgRanks, nBlocks, rankMatrix] = computeFriedmanStatistics(metricMatrix)
    [nRuns, numAlgorithms] = size(metricMatrix);
    completeMask = false(nRuns, 1);
    for runIdx = 1:nRuns
        completeMask(runIdx) = all(isfinite(metricMatrix(runIdx, :)));
    end

    completeData = metricMatrix(completeMask, :);
    nBlocks = size(completeData, 1);
    rankMatrix = NaN(nBlocks, numAlgorithms);
    avgRanks = NaN(1, numAlgorithms);

    chiSquare = NaN;
    pValue = NaN;

    if nBlocks == 0
        return;
    end

    for blockIdx = 1:nBlocks
        rankMatrix(blockIdx, :) = averageRanks(completeData(blockIdx, :));
    end

    avgRanks = mean(rankMatrix, 1);

    if nBlocks < 2 || numAlgorithms < 2
        return;
    end

    rankSums = sum(rankMatrix, 1);
    chiSquare = (12 / (nBlocks * numAlgorithms * (numAlgorithms + 1))) * ...
        sum(rankSums .^ 2) - 3 * nBlocks * (numAlgorithms + 1);
    chiSquare = max(chiSquare, 0);

    pValue = 1 - chi2cdfLocal(chiSquare, numAlgorithms - 1);
end

function p = chi2cdfLocal(x, df)
    if df <= 0
        p = NaN;
        return;
    end

    if x <= 0
        p = 0;
        return;
    end

    p = gammainc(x / 2, df / 2, 'lower');
end

function ranks = averageRanks(values)
    values = values(:)';
    n = numel(values);
    ranks = NaN(1, n);

    if n == 0
        return;
    end

    [sortedValues, sortedOrder] = sort(values, 'ascend');
    sortedRanks = zeros(1, n);

    startIdx = 1;
    while startIdx <= n
        endIdx = startIdx;
        while endIdx < n && sortedValues(endIdx + 1) == sortedValues(startIdx)
            endIdx = endIdx + 1;
        end

        avgRank = (startIdx + endIdx) / 2;
        sortedRanks(startIdx:endIdx) = avgRank;
        startIdx = endIdx + 1;
    end

    ranks(sortedOrder) = sortedRanks;
end

function names = getAlgorithmNames(algorithms)
    numAlgorithms = length(algorithms);
    names = cell(numAlgorithms, 1);
    for algIdx = 1:numAlgorithms
        names{algIdx} = algorithms{algIdx}.displayName;
    end
end

function scalar = toScalar(value)
    scalar = NaN;
    if ~isnumeric(value) || isempty(value)
        return;
    end

    value = value(:);
    scalar = double(value(end));
end

function value = safeMean(values)
    if isempty(values)
        value = NaN;
    else
        value = mean(values);
    end
end

function value = safeStd(values)
    if numel(values) < 2
        value = NaN;
    else
        value = std(values, 0);
    end
end

function value = safeMedian(values)
    if isempty(values)
        value = NaN;
    else
        value = median(values);
    end
end

function value = safeMin(values)
    if isempty(values)
        value = NaN;
    else
        value = min(values);
    end
end

function value = safeMax(values)
    if isempty(values)
        value = NaN;
    else
        value = max(values);
    end
end

function row = emptyPairwiseResult()
    row = struct( ...
        'scenario_id', NaN, ...
        'metric_name', '', ...
        'algorithm_1', '', ...
        'algorithm_2', '', ...
        'n_pairs', NaN, ...
        'mean_1', NaN, ...
        'mean_2', NaN, ...
        'mean_diff', NaN, ...
        'ci95_low', NaN, ...
        'ci95_high', NaN, ...
        't_stat', NaN, ...
        'p_raw_t', NaN, ...
        'p_holm_t', NaN, ...
        'sig_holm_t', false, ...
        'cohen_dz', NaN, ...
        'wilcoxon_W', NaN, ...
        'p_raw_wilcoxon', NaN, ...
        'p_holm_wilcoxon', NaN, ...
        'sig_holm_wilcoxon', false, ...
        'rank_biserial_r', NaN);
end

function row = emptySummaryResult()
    row = struct( ...
        'scenario_id', NaN, ...
        'metric_name', '', ...
        'algorithm', '', ...
        'n_valid', NaN, ...
        'mean', NaN, ...
        'std', NaN, ...
        'median', NaN, ...
        'min', NaN, ...
        'max', NaN, ...
        'average_rank', NaN, ...
        'significant_wins_holm_t', NaN, ...
        'friedman_chi_square', NaN, ...
        'friedman_df', NaN, ...
        'friedman_p', NaN, ...
        'n_friedman_runs', NaN);
end
