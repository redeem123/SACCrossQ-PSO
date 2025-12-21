function generate_additional_figures()
    % GENERATE_ADDITIONAL_FIGURES
    % Generates:
    % 1. Alpha (Temperature) Evolution during training.
    % 2. Cost Component Breakdown for Scenario 3 (Stacked Bar).
    % 3. Path Length vs Fitness Trade-off for Scenario 3.

    clc; close all;
    
    outputDir = fullfile('outputs', 'results', 'training_comparison');
    if ~exist(outputDir, 'dir'), mkdir(outputDir); end
    
    %% 1. Alpha Evolution Plot
    fprintf('Generating Alpha Evolution plot...\n');
    modelDir = fullfile('models', 'APEXPSO');
    files = {'apexpso_perparticle_log.mat', 'apexpso_abl_noatt_log.mat', 'apexpso_abl_simplerwd_log.mat', 'apexpso_abl_nocrossq_log.mat'};
    names = {'Full (Baseline)', 'No Attention', 'Simple Reward', 'No CrossQ'};
    colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250; 0.4940 0.1840 0.5560];
    
    fig1 = figure('Position', [100, 100, 800, 500]);
    hold on; grid on;
    
    for i = 1:length(files)
        path = fullfile(modelDir, files{i});
        if exist(path, 'file')
            d = load(path);
            if isfield(d.logData, 'alphaValue')
                alpha = d.logData.alphaValue;
                plot(smooth(alpha, 10), 'LineWidth', 2, 'Color', colors(i,:), 'DisplayName', names{i});
            end
        end
    end
    
    xlabel('Training Episode', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Temperature (\alpha)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Entropy Temperature Adaptation', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'northeast');
    xlim([0 200]);
    
    exportgraphics(fig1, fullfile(outputDir, 'Training_Alpha_Comparison.pdf'), 'ContentType', 'vector');
    
    %% 2. Scenario 3 Cost Breakdown (Data Hardcoded from CSV for robust simplicity)
    % Data from TTest_AlgorithmSummary_Scenario3_2025-12-15_09-21-30.csv
    % Weights: L=1, Turn=20, Climb=200/250?, Coll=1000? 
    % Actually, fitness = Path + Turn + Climb + Coll. 
    % The CSV columns are: Mean_PathLength, Mean_TurningPenalty, Mean_ClimbingPenalty, Mean_HeightPenalty, Mean_CollisionPenalty
    % We need to check if the CSV "Penalty" columns are raw values or weighted.
    % Usually "Penalty" implies the term added to fitness.
    
    % Let's read the CSV to be sure.
    csvPath = fullfile('outputs', 'results', 'result_ablation', 'TTest_AlgorithmSummary_Scenario3_2025-12-15_09-21-30.csv');
    opts = detectImportOptions(csvPath);
    opts.VariableNamingRule = 'preserve';
    T = readtable(csvPath, opts);
    
    % Extract rows for relevant algorithms
    algs = {'APEXPSO Run: Full (Baseline)', 'APEXPSO Run: No Attention', 'APEXPSO Run: Simple Reward', 'APEXPSO Run: No CrossQ'};
    shortNames = {'Full (Baseline)', 'No Attention', 'Simple Reward', 'No CrossQ'};
    
    % Data matrices
    pathL = zeros(1, 4);
    climbP = zeros(1, 4);
    turnP = zeros(1, 4);
    collP = zeros(1, 4);
    heightP = zeros(1, 4);
    
    % Fitness Function Weights (approximate from paper context)
    % Fitness = Path + 10*Turn + 10*Climb + Coll... 
    % Wait, the CSV headers are "Mean_ClimbingPenalty". If the code saves the *Weighted* penalty, we can stack them directly.
    % If it saves the raw value, we need weights.
    % Given "Mean_GlobalFitness" is ~1700 and Path is ~160, the penalties must make up the rest.
    % Let's verify: Fitness - Path = 1756 - 160 = 1596.
    % CSV: Climb=4.69, Height=229, Turn=2.16. This doesn't sum to 1596.
    % Ah, likely the "Penalty" columns in CSV are RAW values (e.g. degrees, meters), not weighted fitness components.
    % We need to infer the weights or use the "Fitness" vs "Path" difference as a "Total Penalty" block.
    % Actually, "Mean_HeightPenalty" is ~229. 
    % Let's assume the "Weighted Penalties" are what matters for the stacked bar.
    % We can plot: [Path Length] vs [Non-Path Penalties] (Fitness - Path Length).
    
    totalFit = zeros(1, 4);
    
    for i = 1:length(algs)
        rowIdx = strcmp(T.Algorithm, algs{i});
        if any(rowIdx)
            pathL(i) = T.Mean_PathLength(rowIdx);
            totalFit(i) = T.Mean_GlobalFitness(rowIdx);
        end
    end
    
    nonPathPenalty = totalFit - pathL;
    
    %% Figure 2: Stacked Bar (Path vs Penalties)
    fprintf('Generating Cost Breakdown plot...\n');
    fig2 = figure('Position', [150, 150, 800, 600]);
    b = bar([pathL', nonPathPenalty'], 'stacked');
    b(1).FaceColor = [0.2 0.6 0.8]; % Path
    b(2).FaceColor = [0.8 0.3 0.3]; % Penalties
    
    xticklabels(shortNames);
    xtickangle(15);
    ylabel('Global Fitness Score', 'FontSize', 12, 'FontWeight', 'bold');
    legend({'Path Cost', 'Constraint Penalties'}, 'Location', 'northwest');
    title('Scenario 3: Fitness Composition', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    
    % Add text labels
    for i = 1:4
        text(i, pathL(i)/2, sprintf('%.0f', pathL(i)), 'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold');
        text(i, pathL(i) + nonPathPenalty(i)/2, sprintf('%.0f', nonPathPenalty(i)), 'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold');
        text(i, totalFit(i) + 50, sprintf('%.0f', totalFit(i)), 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    
    exportgraphics(fig2, fullfile(outputDir, 'Ablation_S3_CostBreakdown.pdf'), 'ContentType', 'vector');
    
    %% 3. Figure: Dual-Axis Comparison (Fitness vs Length)
    fprintf('Generating Fitness vs Length plot...\n');
    fig3 = figure('Position', [200, 200, 800, 600]);
    yyaxis left
    b1 = bar(1:4, totalFit, 0.4, 'FaceColor', [0.8500 0.3250 0.0980]); % Orange
    ylabel('Global Fitness (Lower is Better)', 'FontSize', 12, 'FontWeight', 'bold');
    ylim([1500 1900]);
    
    yyaxis right
    hold on;
    % Shift x-axis slightly for grouped effect if using bars, or just use points/lines
    % Let's use a line with markers for Path Length to distinguish
    p1 = plot(1:4, pathL, '-o', 'LineWidth', 3, 'MarkerSize', 10, 'Color', [0 0.4470 0.7410], 'MarkerFaceColor', 'w');
    ylabel('Path Length (Meters)', 'FontSize', 12, 'FontWeight', 'bold');
    ylim([140 200]);
    
    ax = gca;
    ax.YAxis(1).Color = [0.8500 0.3250 0.0980];
    ax.YAxis(2).Color = [0 0.4470 0.7410];
    
    xticks(1:4);
    xticklabels(shortNames);
    xtickangle(15);
    title('Scenario 3: The "Cheating" Trade-off', 'FontSize', 14, 'FontWeight', 'bold');
    
    legend([b1, p1], {'Global Fitness', 'Path Length'}, 'Location', 'north');
    grid on;
    
    exportgraphics(fig3, fullfile(outputDir, 'Ablation_S3_LengthVsFitness.pdf'), 'ContentType', 'vector');
    
    fprintf('Done! Figures saved to %s\n', outputDir);
end
