function sync_ablation_tables_to_tex(validationDir)
%SYNC_ABLATION_TABLES_TO_TEX Sync ablation/training tables into main.tex files.

    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    if nargin < 1 || strlength(string(validationDir)) == 0
        validationDir = fullfile(repoRoot, 'outputs', 'results', 'ablation_validation');
    end
    validationDir = char(validationDir);

    ablationCsv = fullfile(validationDir, 'ablation_table_reconstructed.csv');
    trainCsv = fullfile(validationDir, 'training_log_metrics.csv');
    pairCsv = fullfile(validationDir, 'ablation_pairwise_stats_full_vs_variants.csv');

    mustExist(ablationCsv);
    mustExist(trainCsv);
    mustExist(pairCsv);

    % Force comma parsing here; auto-detection occasionally mis-parses SourceFile paths.
    Tabl = readtable(ablationCsv, 'Delimiter', ',', 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');
    Ttr = readtable(trainCsv, 'VariableNamingRule', 'preserve');
    Tpw = readtable(pairCsv, 'VariableNamingRule', 'preserve');

    texFiles = {
        fullfile(repoRoot, 'sage_latex_template_4_unzipped', 'main.tex');
        fullfile(repoRoot, 'paper', 'main.tex')
    };

    for i = 1:numel(texFiles)
        syncSingleTex(texFiles{i}, Tabl, Ttr, Tpw);
    end

    fprintf('Synced ablation/training tables into %d TeX files.\n', numel(texFiles));
end

function mustExist(pathIn)
    if exist(pathIn, 'file') ~= 2
        error('Missing required file: %s', pathIn);
    end
end

function syncSingleTex(texPath, Tabl, Ttr, Tpw)
    txt = fileread(texPath);

    s1 = scenarioLine(Tabl, 1);
    s2 = scenarioLine(Tabl, 2);
    s3 = scenarioLine(Tabl, 3);

    txt = regexprep(txt, '^Scenario 1 & .*\\\\\s*$', s1, 'lineanchors');
    txt = regexprep(txt, '^Scenario 2 & .*\\\\\s*$', s2, 'lineanchors');
    txt = regexprep(txt, '^Scenario 3 & .*\\\\\s*$', s3, 'lineanchors');

    txt = regexprep(txt, '^Full \(Baseline\) & .*\\\\\s*$', trainingLine(Ttr, 'Full (Baseline)'), 'lineanchors');
    txt = regexprep(txt, '^No Attention & .*\\\\\s*$', trainingLine(Ttr, 'No Attention'), 'lineanchors');
    txt = regexprep(txt, '^No CrossQ & .*\\\\\s*$', trainingLine(Ttr, 'No CrossQ'), 'lineanchors');

    statsBlock = buildPairwiseStatsTable(Tpw);
    beginMarker = '% BEGIN AUTO_ABLATION_PAIRWISE_STATS';
    endMarker = '% END AUTO_ABLATION_PAIRWISE_STATS';

    if contains(txt, beginMarker) && contains(txt, endMarker)
        pattern = [regexptranslate('escape', beginMarker) '[\s\S]*?' regexptranslate('escape', endMarker)];
        replacement = sprintf('%s\n%s\n%s', beginMarker, statsBlock, endMarker);
        txt = regexprep(txt, pattern, replacement, 'once');
    else
        anchor = '\subsection{Training Efficiency and Convergence}';
        insertText = sprintf('\n%s\n%s\n%s\n\n', beginMarker, statsBlock, endMarker);
        txt = strrep(txt, anchor, [insertText anchor]);
    end

    fid = fopen(texPath, 'w');
    if fid == -1
        error('Cannot open %s for writing', texPath);
    end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid, txt, 'char');

    fprintf('Updated: %s\n', texPath);
end

function out = scenarioLine(T, s)
    scenario = getCol(T, {'Scenario'});
    variant = getCol(T, {'Variant'});
    meanFit = getCol(T, {'MeanFitness'});
    stdFit = getCol(T, {'StdFitness'});

    variants = {'Full (Baseline)','No Attention','No CrossQ'};
    means = zeros(1, numel(variants));
    stds = zeros(1, numel(variants));
    for i = 1:numel(variants)
        idx = find(scenario == s & strcmp(variant, variants{i}), 1, 'first');
        if isempty(idx)
            error('Missing variant %s in scenario %d', variants{i}, s);
        end
        means(i) = meanFit(idx);
        stds(i) = stdFit(idx);
    end

    [~, bestMeanIdx] = min(means);
    [~, bestStdIdx] = min(stds);

    cells = cell(1, numel(variants));
    for i = 1:numel(variants)
        m = sprintf('%.1f', means(i));
        sd = sprintf('%.1f', stds(i));
        if i == bestMeanIdx
            m = ['\\textbf{' m '}'];
        end
        if i == bestStdIdx
            sd = ['\\textbf{' sd '}'];
        end
        cells{i} = sprintf('%s $\\pm$ %s', m, sd);
    end

    out = sprintf('Scenario %d & %s & %s & %s \\\\', s, cells{1}, cells{2}, cells{3});
end

function out = trainingLine(T, variantName)
    variant = getCol(T, {'Variant'});
    meanRewardLast20 = getCol(T, {'MeanRewardLast20'});
    rewardStdLast50 = getCol(T, {'RewardStdLast50'});
    finalFitness = getCol(T, {'FinalFitness'});
    fitnessStdLast50 = getCol(T, {'FitnessStdLast50'});
    totalHours = getCol(T, {'TotalTrainingHours'});

    idx = find(strcmp(variant, variantName), 1, 'first');
    if isempty(idx)
        error('Missing training row for %s', variantName);
    end

    out = sprintf('%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.2f \\\\', ...
        variantName, meanRewardLast20(idx), rewardStdLast50(idx), ...
        finalFitness(idx), fitnessStdLast50(idx), totalHours(idx));
end

function block = buildPairwiseStatsTable(T)
    scenario = getCol(T, {'Scenario'});
    variant = getCol(T, {'Variant'});
    pHolmT = getCol(T, {'P_Holm_T'});
    ciLow = getCol(T, {'CI95_Low'});
    ciHigh = getCol(T, {'CI95_High'});
    dz = getCol(T, {'Cohen_dz'});

    variants = {'No Attention','No CrossQ'};
    lines = {};
    lines{end+1} = '\\begin{table}[t]';
    lines{end+1} = '\\centering';
    lines{end+1} = '\\small';
    lines{end+1} = '\\begin{tabular}{lcccc}';
    lines{end+1} = '\\hline';
    lines{end+1} = 'Scenario & Variant & $p_{Holm}$ (t) & 95\\% CI & $d_z$ \\\\';
    lines{end+1} = '\\hline';

    for s = 1:3
        for v = 1:numel(variants)
            idx = find(scenario == s & strcmp(variant, variants{v}), 1, 'first');
            if isempty(idx)
                continue;
            end
            lines{end+1} = sprintf('S%d & %s & %.3f & [%.1f, %.1f] & %.3f \\\\', ...
                s, variants{v}, pHolmT(idx), ciLow(idx), ciHigh(idx), dz(idx));
        end
    end

    lines{end+1} = '\\hline';
    lines{end+1} = '\\end{tabular}';
    lines{end+1} = '\\caption{Supplementary pairwise statistics (Full baseline vs each ablation variant) using Holm-corrected paired t-tests.}';
    lines{end+1} = '\\label{tab:ablation_pairwise_stats}';
    lines{end+1} = '\\end{table}';

    block = strjoin(lines, newline);
end

function col = getCol(T, names)
    vars = T.Properties.VariableNames;
    col = [];

    for i = 1:numel(names)
        idx = find(strcmp(vars, names{i}), 1, 'first');
        if ~isempty(idx)
            col = T.(vars{idx});
            return;
        end
    end

    normVars = regexprep(lower(vars), '[^a-z0-9]', '');
    for i = 1:numel(names)
        key = regexprep(lower(names{i}), '[^a-z0-9]', '');
        idx = find(strcmp(normVars, key), 1, 'first');
        if ~isempty(idx)
            col = T.(vars{idx});
            return;
        end
    end

    error('Missing table column. Candidates=%s. Available=%s', strjoin(names, '|'), strjoin(vars, ', '));
end
