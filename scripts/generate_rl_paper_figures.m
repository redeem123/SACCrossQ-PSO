function generate_rl_paper_figures(sourceDir)
%GENERATE_RL_PAPER_FIGURES Generate RL comparison figures from latest scenario CSVs.
%
% Usage:
%   generate_rl_paper_figures();
%   generate_rl_paper_figures('/abs/path/to/results_dir');

    close all;

    if nargin < 1 || strlength(string(sourceDir)) == 0
        repoRoot = fileparts(fileparts(mfilename('fullpath')));
        sourceDir = fullfile(repoRoot, 'outputs', 'results', 'rl_based');
    end
    sourceDir = char(sourceDir);

    csv1 = readtable(resolveLatestScenarioFile(sourceDir, 'TTest_AlgorithmSummary', 1), 'VariableNamingRule', 'preserve');
    csv2 = readtable(resolveLatestScenarioFile(sourceDir, 'TTest_AlgorithmSummary', 2), 'VariableNamingRule', 'preserve');
    csv3 = readtable(resolveLatestScenarioFile(sourceDir, 'TTest_AlgorithmSummary', 3), 'VariableNamingRule', 'preserve');

    % Extract algorithm names (shortened for plotting)
    algNames = csv1.Algorithm;
    shortNames = cell(size(algNames));
    for i = 1:length(algNames)
        name = algNames{i};
        if contains(name, 'RLAMPSO')
            if contains(name, 'Global')
                shortNames{i} = 'RLAMPSO-G';
            elseif contains(name, '5-Subgroup')
                shortNames{i} = 'RLAMPSO-5S';
            else
                shortNames{i} = 'RLAMPSO-PP';
            end
        elseif contains(name, 'DQN')
            if contains(name, 'Global')
                shortNames{i} = 'DQN-G';
            elseif contains(name, '5-Subgroup')
                shortNames{i} = 'DQN-5S';
            else
                shortNames{i} = 'DQN-PP';
            end
        elseif contains(name, 'AFSACPSO')
            if contains(name, 'Global')
                shortNames{i} = 'AFSACPSO-G';
            elseif contains(name, '5-Subgroup')
                shortNames{i} = 'AFSACPSO-5S';
            else
                shortNames{i} = 'AFSACPSO-PP';
            end
        else
            shortNames{i} = name;
        end
    end

    %% Figure 1: Bar chart comparison across scenarios
    figure('Position', [100, 100, 1400, 500], 'Color', 'w');

    means = [csv1.Mean_GlobalFitness, csv2.Mean_GlobalFitness, csv3.Mean_GlobalFitness];
    stds = [csv1.Std_GlobalFitness, csv2.Std_GlobalFitness, csv3.Std_GlobalFitness];

    x = 1:length(shortNames);
    width = 0.25;

    hold on;
    bar(x - width, means(:,1), width, 'FaceColor', [0.2 0.4 0.8]);
    bar(x, means(:,2), width, 'FaceColor', [0.8 0.4 0.2]);
    bar(x + width, means(:,3), width, 'FaceColor', [0.2 0.8 0.4]);

    errorbar(x - width, means(:,1), stds(:,1), 'k.', 'LineWidth', 1);
    errorbar(x, means(:,2), stds(:,2), 'k.', 'LineWidth', 1);
    errorbar(x + width, means(:,3), stds(:,3), 'k.', 'LineWidth', 1);

    set(gca, 'XTick', x, 'XTickLabel', shortNames, 'FontSize', 11);
    xtickangle(45);
    ylabel('Mean Fitness (lower is better)', 'FontSize', 13, 'FontWeight', 'bold');
    legend({'Scenario 1 (0 DZ)', 'Scenario 2 (5 DZ)', 'Scenario 3 (10 DZ)'}, ...
        'Location', 'northwest', 'FontSize', 11);
    grid on;
    set(gca, 'GridAlpha', 0.3);

    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [14, 5]);
    set(gcf, 'PaperPosition', [0, 0, 14, 5]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'Renderer', 'painters');
    print(gcf, fullfile(sourceDir, 'RL_Comparison_BarChart'), '-dpdf', '-r300');

    %% Figure 2: Computation time comparison
    figure('Position', [100, 100, 1200, 500], 'Color', 'w');

    times = [csv1.Mean_TotalGlobalPlanTime, csv2.Mean_TotalGlobalPlanTime, csv3.Mean_TotalGlobalPlanTime];
    timeStds = [csv1.Std_TotalGlobalPlanTime, csv2.Std_TotalGlobalPlanTime, csv3.Std_TotalGlobalPlanTime];

    hold on;
    bar(x - width, times(:,1), width, 'FaceColor', [0.6 0.2 0.8]);
    bar(x, times(:,2), width, 'FaceColor', [0.8 0.6 0.2]);
    bar(x + width, times(:,3), width, 'FaceColor', [0.2 0.8 0.8]);

    errorbar(x - width, times(:,1), timeStds(:,1), 'k.', 'LineWidth', 1);
    errorbar(x, times(:,2), timeStds(:,2), 'k.', 'LineWidth', 1);
    errorbar(x + width, times(:,3), timeStds(:,3), 'k.', 'LineWidth', 1);

    set(gca, 'XTick', x, 'XTickLabel', shortNames, 'FontSize', 11);
    xtickangle(45);
    ylabel('Computation Time (seconds)', 'FontSize', 13, 'FontWeight', 'bold');
    legend({'Scenario 1', 'Scenario 2', 'Scenario 3'}, 'Location', 'northwest', 'FontSize', 11);
    grid on;
    set(gca, 'GridAlpha', 0.3);

    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [12, 5]);
    set(gcf, 'PaperPosition', [0, 0, 12, 5]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'Renderer', 'painters');
    print(gcf, fullfile(sourceDir, 'RL_Computation_Time'), '-dpdf', '-r300');

    %% Figure 3: Performance table summary (saved as text for LaTeX)
    fid = fopen(fullfile(sourceDir, 'RL_Table_Summary.txt'), 'w');

    fprintf(fid, '\\begin{table}[htbp]\n');
    fprintf(fid, '\\centering\n');
    fprintf(fid, '\\caption{RL-based Algorithm Performance Comparison}\n');
    fprintf(fid, '\\label{tab:rl_comparison}\n');
    fprintf(fid, '\\begin{tabular}{lccccc}\n');
    fprintf(fid, '\\hline\n');
    fprintf(fid, 'Algorithm & \\multicolumn{3}{c}{Mean Fitness $\\pm$ Std} & Time (s) \\\\n');
    fprintf(fid, '& Scenario 1 & Scenario 2 & Scenario 3 & Avg \\\\n');
    fprintf(fid, '\\hline\n');

    for i = 1:length(shortNames)
        avgTime = mean([csv1.Mean_TotalGlobalPlanTime(i), csv2.Mean_TotalGlobalPlanTime(i), csv3.Mean_TotalGlobalPlanTime(i)]);
        fprintf(fid, '%s & %.1f$\\pm$%.1f & %.1f$\\pm$%.1f & %.1f$\\pm$%.1f & %.1f \\\\n', ...
            shortNames{i}, ...
            csv1.Mean_GlobalFitness(i), csv1.Std_GlobalFitness(i), ...
            csv2.Mean_GlobalFitness(i), csv2.Std_GlobalFitness(i), ...
            csv3.Mean_GlobalFitness(i), csv3.Std_GlobalFitness(i), ...
            avgTime);
    end

    fprintf(fid, '\\hline\n');
    fprintf(fid, '\\end{tabular}\n');
    fprintf(fid, '\\end{table}\n');

    fclose(fid);
    fprintf('RL paper figures generated from: %s\n', sourceDir);
end

function filePath = resolveLatestScenarioFile(sourceDir, prefix, scenario)
    pattern = fullfile(sourceDir, sprintf('%s_Scenario%d_*.csv', prefix, scenario));
    files = dir(pattern);
    if isempty(files)
        error('No matching files for %s scenario %d in %s', prefix, scenario, sourceDir);
    end

    [~, idx] = max([files.datenum]);
    filePath = fullfile(files(idx).folder, files(idx).name);
end
