function analyze_ablation_results()
    % STATISTICAL ANALYSIS OF ABLATION STUDY RESULTS
    %
    % This script performs rigorous statistical analysis on APEXPSO ablation study
    % results to determine significance of performance differences.
    %
    % USAGE:
    %   1. After running comparisons, extract fitness values for each algorithm
    %   2. Input data into the data structure below
    %   3. Run this script: analyze_ablation_results()
    %   4. Review statistical output and generated plots
    %
    % OUTPUT:
    %   - Statistical significance tests (t-tests, effect sizes)
    %   - Performance comparison table with confidence intervals
    %   - Box plots and bar charts with error bars
    %   - Publication-ready figures saved to outputs/figures/

    close all;
    clc;

    %% ========== INPUT YOUR DATA HERE ==========

    % TODO: Replace these with your actual results from 10 runs
    % Each row is one independent run

    % Example data structure (10 runs × 5 algorithms)
    % Replace with your actual data from run_comparison results

    data = struct();

    % IMPORTANT: Input your actual fitness values here!
    % Each algorithm should have 10 values (one per independent run)

    data.Full = [1608.0719, 1590.23, 1620.45, 1595.78, 1612.34, ...
                 1605.89, 1615.67, 1600.12, 1610.45, 1598.76];  % Replace with actual

    data.Global = [1583.7753, 1575.12, 1590.34, 1580.56, 1588.90, ...
                   1577.45, 1585.23, 1582.78, 1589.01, 1584.56];  % Replace with actual

    data.Subgroup5 = [1577.5973, 1570.23, 1585.67, 1572.89, 1580.12, ...
                      1575.45, 1582.34, 1574.90, 1578.23, 1576.89];  % Replace with actual

    data.NoAttention = [1644.7081, 1638.45, 1650.12, 1642.78, 1648.90, ...
                        1640.23, 1646.78, 1643.56, 1649.23, 1641.67];  % Replace with actual

    data.SimpleReward = [1613.1731, 1605.89, 1620.45, 1610.23, 1615.78, ...
                         1608.34, 1618.90, 1612.56, 1614.23, 1609.67];  % Replace with actual

    % Algorithm names for display
    algorithmNames = {'Full (Baseline)', 'Global', '5-Subgroup', 'No Attention', 'Simple Reward'};

    %% ========== DESCRIPTIVE STATISTICS ==========

    fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
    fprintf('║          APEXPSO ABLATION STUDY - STATISTICAL ANALYSIS          ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════════╝\n\n');

    fprintf('Descriptive Statistics (Global Path Fitness - Lower is Better)\n');
    fprintf('════════════════════════════════════════════════════════════════════\n');
    fprintf('%-20s %10s %10s %10s %10s %10s\n', 'Algorithm', 'Mean', 'Std', 'Min', 'Max', '95% CI');
    fprintf('────────────────────────────────────────────────────────────────────\n');

    results = struct();
    algorithms = fieldnames(data);

    for i = 1:length(algorithms)
        alg = algorithms{i};
        values = data.(alg);

        results.(alg).mean = mean(values);
        results.(alg).std = std(values);
        results.(alg).sem = std(values) / sqrt(length(values));
        results.(alg).min = min(values);
        results.(alg).max = max(values);
        results.(alg).n = length(values);

        % 95% Confidence Interval
        tval = tinv(0.975, length(values)-1);  % t-value for 95% CI
        ci_margin = tval * results.(alg).sem;
        results.(alg).ci95_lower = results.(alg).mean - ci_margin;
        results.(alg).ci95_upper = results.(alg).mean + ci_margin;

        fprintf('%-20s %10.2f %10.2f %10.2f %10.2f [%.2f, %.2f]\n', ...
            algorithmNames{i}, results.(alg).mean, results.(alg).std, ...
            results.(alg).min, results.(alg).max, ...
            results.(alg).ci95_lower, results.(alg).ci95_upper);
    end
    fprintf('════════════════════════════════════════════════════════════════════\n\n');

    %% ========== PAIRWISE COMPARISONS ==========

    fprintf('Pairwise Statistical Comparisons (vs. Full Baseline)\n');
    fprintf('════════════════════════════════════════════════════════════════════\n');
    fprintf('%-20s %10s %10s %12s %15s\n', 'Algorithm', 'Δ Mean', 'Δ %%', 'p-value', 'Significance');
    fprintf('────────────────────────────────────────────────────────────────────\n');

    baseline = data.Full;
    baselineMean = results.Full.mean;

    for i = 2:length(algorithms)  % Skip Full (it's the baseline)
        alg = algorithms{i};
        values = data.(alg);

        % Two-sample t-test
        [h, p] = ttest2(baseline, values);

        % Effect size (Cohen's d)
        pooledStd = sqrt((results.Full.std^2 + results.(alg).std^2) / 2);
        cohensD = abs(results.(alg).mean - baselineMean) / pooledStd;

        % Delta from baseline
        delta = results.(alg).mean - baselineMean;
        deltaPercent = (delta / baselineMean) * 100;

        % Significance level
        if p < 0.001
            sig = '***';
        elseif p < 0.01
            sig = '**';
        elseif p < 0.05
            sig = '*';
        else
            sig = 'n.s.';
        end

        fprintf('%-20s %+10.2f %+9.2f%% %12.4f %10s (d=%.2f)\n', ...
            algorithmNames{i}, delta, deltaPercent, p, sig, cohensD);

        results.(alg).p_value = p;
        results.(alg).cohens_d = cohensD;
        results.(alg).delta = delta;
    end

    fprintf('────────────────────────────────────────────────────────────────────\n');
    fprintf('Significance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant\n');
    fprintf('Effect size (Cohen''s d): small=0.2, medium=0.5, large=0.8\n');
    fprintf('════════════════════════════════════════════════════════════════════\n\n');

    %% ========== KEY COMPARISONS ==========

    fprintf('Key Ablation Study Comparisons\n');
    fprintf('════════════════════════════════════════════════════════════════════\n');

    % Parameter Granularity Study
    fprintf('\n1. PARAMETER GRANULARITY (Action Space Ablation)\n');
    fprintf('   ────────────────────────────────────────────────────────────\n');
    [h_global, p_global] = ttest2(data.Subgroup5, data.Global);
    fprintf('   5-Subgroup (15D) vs. Global (3D):    Δ=%.2f, p=%.4f %s\n', ...
        results.Subgroup5.mean - results.Global.mean, p_global, ...
        p_global < 0.05 ? '✓ Significant' : '✗ Not significant');

    [h_per, p_per] = ttest2(data.Subgroup5, data.Full);
    fprintf('   5-Subgroup (15D) vs. Per-Particle (120D): Δ=%.2f, p=%.4f %s\n', ...
        results.Subgroup5.mean - results.Full.mean, p_per, ...
        p_per < 0.05 ? '✓ Significant' : '✗ Not significant');

    % Feature Ablation Study
    fprintf('\n2. FEATURE ABLATION STUDY\n');
    fprintf('   ────────────────────────────────────────────────────────────\n');
    [h_att, p_att] = ttest2(data.Full, data.NoAttention);
    fprintf('   Attention Impact (Full vs. No Attention): Δ=%.2f, p=%.4f %s\n', ...
        results.NoAttention.mean - results.Full.mean, p_att, ...
        p_att < 0.05 ? '✓ Significant' : '✗ Not significant');

    [h_rwd, p_rwd] = ttest2(data.Full, data.SimpleReward);
    fprintf('   Reward Impact (Multi-obj vs. Simple):     Δ=%.2f, p=%.4f %s\n', ...
        results.SimpleReward.mean - results.Full.mean, p_rwd, ...
        p_rwd < 0.05 ? '✓ Significant' : '✗ Not significant');

    fprintf('════════════════════════════════════════════════════════════════════\n\n');

    %% ========== RANKING ==========

    fprintf('Performance Ranking (Lower Fitness = Better)\n');
    fprintf('════════════════════════════════════════════════════════════════════\n');

    means = [results.Full.mean, results.Global.mean, results.Subgroup5.mean, ...
             results.NoAttention.mean, results.SimpleReward.mean];
    [sortedMeans, idx] = sort(means);

    medals = {'🥇', '🥈', '🥉', '4th', '5th'};
    for i = 1:length(idx)
        fprintf('%s %-20s  %.2f ± %.2f\n', medals{i}, ...
            algorithmNames{idx(i)}, sortedMeans(i), ...
            results.(algorithms{idx(i)}).std);
    end
    fprintf('════════════════════════════════════════════════════════════════════\n\n');

    %% ========== VISUALIZATION ==========

    fprintf('Generating publication-quality figures...\n');

    % Create output directory
    outputDir = 'outputs/figures';
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Figure 1: Bar chart with error bars
    fig1 = figure('Position', [100, 100, 900, 600]);

    meansArray = [results.Full.mean, results.Global.mean, results.Subgroup5.mean, ...
                  results.NoAttention.mean, results.SimpleReward.mean];
    semsArray = [results.Full.sem, results.Global.sem, results.Subgroup5.sem, ...
                 results.NoAttention.sem, results.SimpleReward.sem];

    bar(meansArray, 'FaceColor', [0.2 0.4 0.7]);
    hold on;
    errorbar(1:5, meansArray, semsArray*1.96, 'k.', 'LineWidth', 1.5);  % 95% CI

    set(gca, 'XTickLabel', algorithmNames, 'FontSize', 11);
    xtickangle(15);
    ylabel('Mean Global Path Fitness (Lower is Better)', 'FontSize', 12, 'FontWeight', 'bold');
    title('APEXPSO Ablation Study Results', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;

    % Add best performer marker
    [~, bestIdx] = min(meansArray);
    text(bestIdx, meansArray(bestIdx) - 20, '★ BEST', ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');

    saveas(fig1, fullfile(outputDir, 'ablation_bar_chart.png'));
    fprintf('✓ Saved: %s\n', fullfile(outputDir, 'ablation_bar_chart.png'));

    % Figure 2: Box plot
    fig2 = figure('Position', [150, 150, 900, 600]);

    dataMatrix = [data.Full', data.Global', data.Subgroup5', data.NoAttention', data.SimpleReward'];
    boxplot(dataMatrix, 'Labels', algorithmNames, 'Widths', 0.6);
    ylabel('Global Path Fitness (Lower is Better)', 'FontSize', 12, 'FontWeight', 'bold');
    title('APEXPSO Ablation Study - Distribution Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xtickangle(15);

    saveas(fig2, fullfile(outputDir, 'ablation_boxplot.png'));
    fprintf('✓ Saved: %s\n', fullfile(outputDir, 'ablation_boxplot.png'));

    % Figure 3: Delta from baseline
    fig3 = figure('Position', [200, 200, 900, 600]);

    deltas = [0, results.Global.delta, results.Subgroup5.delta, ...
              results.NoAttention.delta, results.SimpleReward.delta];
    colors = [0.7 0.7 0.7; 0.2 0.7 0.2; 0.2 0.7 0.2; 0.9 0.2 0.2; 0.9 0.5 0.2];

    bar(deltas, 'FaceColor', 'flat', 'CData', colors);
    hold on;
    plot([0.5 5.5], [0 0], 'k--', 'LineWidth', 1.5);

    set(gca, 'XTickLabel', algorithmNames, 'FontSize', 11);
    xtickangle(15);
    ylabel('Δ from Full Baseline (Negative = Better)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Performance Difference from Baseline', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;

    saveas(fig3, fullfile(outputDir, 'ablation_delta_chart.png'));
    fprintf('✓ Saved: %s\n', fullfile(outputDir, 'ablation_delta_chart.png'));

    fprintf('\n✓ Analysis complete! Figures saved to: %s\n', outputDir);

    %% ========== SAVE RESULTS ==========

    resultsFile = fullfile(outputDir, 'ablation_statistical_results.mat');
    save(resultsFile, 'results', 'data', 'algorithmNames');
    fprintf('✓ Results saved to: %s\n\n', resultsFile);

    %% ========== PUBLICATION-READY TABLE ==========

    fprintf('════════════════════════════════════════════════════════════════════\n');
    fprintf('LATEX TABLE (Copy to your paper)\n');
    fprintf('════════════════════════════════════════════════════════════════════\n\n');

    fprintf('\\begin{table}[h]\n');
    fprintf('\\centering\n');
    fprintf('\\caption{APEXPSO Ablation Study Results}\n');
    fprintf('\\begin{tabular}{lcccc}\n');
    fprintf('\\toprule\n');
    fprintf('Algorithm & Mean ± SD & 95\\%% CI & $\\Delta$ vs. Baseline & p-value \\\\\n');
    fprintf('\\midrule\n');

    for i = 1:length(algorithms)
        alg = algorithms{i};
        if i == 1
            fprintf('%s & %.2f ± %.2f & [%.2f, %.2f] & — & — \\\\\n', ...
                algorithmNames{i}, results.(alg).mean, results.(alg).std, ...
                results.(alg).ci95_lower, results.(alg).ci95_upper);
        else
            sig = '';
            if results.(alg).p_value < 0.001
                sig = '***';
            elseif results.(alg).p_value < 0.01
                sig = '**';
            elseif results.(alg).p_value < 0.05
                sig = '*';
            end
            fprintf('%s & %.2f ± %.2f & [%.2f, %.2f] & %.2f & %.4f%s \\\\\n', ...
                algorithmNames{i}, results.(alg).mean, results.(alg).std, ...
                results.(alg).ci95_lower, results.(alg).ci95_upper, ...
                results.(alg).delta, results.(alg).p_value, sig);
        end
    end

    fprintf('\\bottomrule\n');
    fprintf('\\end{tabular}\n');
    fprintf('\\label{tab:ablation}\n');
    fprintf('\\end{table}\n\n');

    fprintf('════════════════════════════════════════════════════════════════════\n');

end
