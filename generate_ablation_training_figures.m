function generate_ablation_training_figures()
    % GENERATE_ABLATION_TRAINING_FIGURES
    % Reads _log.mat files from models/APEXPSO and generates publication-quality
    % figures and tables for the "Training Efficiency and Convergence" section.

    clc; close all;
    
    % Configuration
    modelDir = fullfile('models', 'APEXPSO');
    outputDir = fullfile('outputs', 'results', 'training_comparison');
    if ~exist(outputDir, 'dir'), mkdir(outputDir); end
    
    % Files to process (Map display name to filename)
    models = containers.Map();
    models('Full (Baseline)') = 'apexpso_perparticle_log.mat';
    models('No Attention') = 'apexpso_abl_noatt_log.mat';
    models('Simple Reward') = 'apexpso_abl_simplerwd_log.mat';
    models('No CrossQ') = 'apexpso_abl_nocrossq_log.mat';
    
    % Colors for plotting
    colors = containers.Map();
    colors('Full (Baseline)') = [0 0.4470 0.7410];      % Blue
    colors('No Attention') = [0.8500 0.3250 0.0980];    % Orange
    colors('Simple Reward') = [0.9290 0.6940 0.1250];   % Yellow
    colors('No CrossQ') = [0.4940 0.1840 0.5560];       % Purple
    
    % Order for legend/tables
    orderedNames = {'Full (Baseline)', 'No Attention', 'Simple Reward', 'No CrossQ'};
    
    fprintf('Loading training logs from %s...\n', modelDir);
    data = struct();
    
    for i = 1:length(orderedNames)
        name = orderedNames{i};
        filename = models(name);
        filePath = fullfile(modelDir, filename);
        
        if exist(filePath, 'file')
            loaded = load(filePath);
            if isfield(loaded, 'logData')
                fieldName = strrep(name, ' ', '_');
                fieldName = strrep(fieldName, '(', '');
                fieldName = strrep(fieldName, ')', '');
                data.(fieldName) = loaded.logData;
                fprintf('  ✓ Loaded %s\n', name);
            else
                warning('File %s does not contain logData. Skipping.', filename);
            end
        else
            warning('File %s not found. Skipping.', filename);
        end
    end
    
    %% 1. Figure: Episode Rewards Comparison (Smoothed)
    fprintf('Generating Episode Reward Comparison plot...\n');
    fig1 = figure('Position', [100, 100, 800, 500]);
    hold on;
    grid on;
    
    windowSize = 10; % Moving average window
    
    for i = 1:length(orderedNames)
        name = orderedNames{i};
        fieldName = strrep(name, ' ', '_');
        fieldName = strrep(fieldName, '(', '');
        fieldName = strrep(fieldName, ')', '');
        
        if ~isfield(data, fieldName), continue; end
        
        rewards = data.(fieldName).episodeRewards;
        episodes = 1:length(rewards);
        
        % Smooth data
        smoothed = smooth(rewards, windowSize);
        
        % Plot smoothed line
        plot(episodes, smoothed, 'LineWidth', 2, 'Color', colors(name), 'DisplayName', name);
        
        % Optional: Plot faint raw data
        % plot(episodes, rewards, 'Color', [colors(name) 0.2], 'HandleVisibility', 'off');
    end
    
    xlabel('Training Episode', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Total Reward (Smoothed)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Training Convergence: Episode Rewards', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'southeast', 'FontSize', 10);
    xlim([0 200]); % Assuming 200 episodes
    
    exportgraphics(fig1, fullfile(outputDir, 'Training_EpisodeRewards_Comparison.pdf'), 'ContentType', 'vector');
    
    %% 2. Figure: Episode Fitness Comparison (Lower is Better)
    fprintf('Generating Episode Fitness Comparison plot...\n');
    fig2 = figure('Position', [150, 150, 800, 500]);
    hold on;
    grid on;
    
    for i = 1:length(orderedNames)
        name = orderedNames{i};
        fieldName = strrep(name, ' ', '_');
        fieldName = strrep(fieldName, '(', '');
        fieldName = strrep(fieldName, ')', '');
        
        if ~isfield(data, fieldName), continue; end
        
        % Use Episode Fitness
        fitness = data.(fieldName).episodeFitness;
        episodes = 1:length(fitness);
        smoothed = smooth(fitness, windowSize);
        
        plot(episodes, smoothed, 'LineWidth', 2, 'Color', colors(name), 'DisplayName', name);
    end
    
    xlabel('Training Episode', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Best Fitness (Lower is Better)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Training Convergence: Path Fitness', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'northeast', 'FontSize', 10);
    xlim([0 200]);
    
    exportgraphics(fig2, fullfile(outputDir, 'Training_Fitness_Comparison.pdf'), 'ContentType', 'vector');
    
    %% 4. Generate LaTeX Table
    fprintf('Generating LaTeX Summary Table...\n');
    
    tableFile = fullfile(outputDir, 'training_summary_table.tex');
    fid = fopen(tableFile, 'w');
    
    fprintf(fid, '\\begin{table}[ht]\n');
    fprintf(fid, '\\centering\n');
    fprintf(fid, '\\small\n');
    fprintf(fid, '\\begin{tabular}{lcccc}\n');
    fprintf(fid, '\\hline\n');
    fprintf(fid, '\\textbf{Variant} & \\textbf{Final Reward} & \\textbf{Final Fitness} & \\textbf{Training Time (h)} & \\textbf{Samples} \\\\\n');
    fprintf(fid, '\\hline\n');
    
    for i = 1:length(orderedNames)
        name = orderedNames{i};
        fieldName = strrep(name, ' ', '_');
        fieldName = strrep(fieldName, '(', '');
        fieldName = strrep(fieldName, ')', '');
        
        if isfield(data, fieldName)
            d = data.(fieldName);
            
            % Metrics (average of last 10 episodes)
            last10 = max(1, length(d.episodeRewards)-9):length(d.episodeRewards);
            finalReward = mean(d.episodeRewards(last10));
            finalRewardStd = std(d.episodeRewards(last10));
            
            % Use Mean Fitness (Last 10) instead of Best Fitness
            finalFit = mean(d.episodeFitness(last10));
            finalFitStd = std(d.episodeFitness(last10));
            
            % Training time
            if isfield(d, 'startTime') && isfield(d, 'endTime')
                try
                    duration = seconds(datetime(d.endTime) - datetime(d.startTime));
                    hours = duration / 3600;
                catch
                    hours = 0; 
                end
            else
                hours = sum(d.episodeDuration) / 3600;
            end
            
            totalSamples = d.totalSamples(end);
            
            fprintf(fid, '%s & %.1f $\\pm$ %.1f & %.1f $\\pm$ %.1f & %.2f & %d \\\\\n', ...
                name, finalReward, finalRewardStd, finalFit, finalFitStd, hours, totalSamples);
        else
            fprintf(fid, '%s & - & - & - & - \\\\\n', name);
        end
    end
    
    fprintf(fid, '\\hline\n');
    fprintf(fid, '\\end{tabular}\n');
    fprintf(fid, '\\caption{Training efficiency comparison of APEXPSO ablation variants.}\n');
    fprintf(fid, '\\label{tab:training_efficiency}\n');
    fprintf(fid, '\\end{table}\n');
    
    fclose(fid);
    fprintf('✓ Table saved to %s\n', tableFile);
    fprintf('Done! Check %s for results.\n', outputDir);
end
