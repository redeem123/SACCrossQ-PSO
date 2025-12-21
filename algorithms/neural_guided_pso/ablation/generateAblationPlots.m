function generateAblationPlots(fitnessMatrix, configs, scenarios)
    % Generate visualization plots for ablation study
    
    figure('Position', [100, 100, 1400, 800], 'Color', 'w');
    
    % Group configurations by ablation type
    architectureIdx = [1, 2, 3, 4];  % Baseline, Full, Shallow, Linear
    featureIdx = [2, 5, 6, 7];       % Full features, Basic, Position, Environment
    trainingIdx = [2, 8, 9];         % Online, No training, Batch
    influenceIdx = [2, 10, 11, 12];  % Adaptive, Low, Medium, High
    
    configNames = cellfun(@(x) x.name, configs, 'UniformOutput', false);
    
    % Plot 1: Architecture ablation
    subplot(2, 2, 1);
    bar(fitnessMatrix(architectureIdx, :));
    set(gca, 'XTickLabel', strrep(configNames(architectureIdx), '_', ' '));
    ylabel('Mean Fitness');
    title('Architecture Ablation');
    legend(arrayfun(@(x) sprintf('Scenario %d', x), 1:size(fitnessMatrix, 2), ...
           'UniformOutput', false), 'Location', 'best');
    xtickangle(45);
    
    % Plot 2: Feature set ablation
    subplot(2, 2, 2);
    bar(fitnessMatrix(featureIdx, :));
    set(gca, 'XTickLabel', strrep(configNames(featureIdx), '_', ' '));
    ylabel('Mean Fitness');
    title('Feature Set Ablation');
    xtickangle(45);
    
    % Plot 3: Training strategy ablation
    subplot(2, 2, 3);
    bar(fitnessMatrix(trainingIdx, :));
    set(gca, 'XTickLabel', strrep(configNames(trainingIdx), '_', ' '));
    ylabel('Mean Fitness');
    title('Training Strategy Ablation');
    xtickangle(45);
    
    % Plot 4: Influence mechanism ablation
    subplot(2, 2, 4);
    bar(fitnessMatrix(influenceIdx, :));
    set(gca, 'XTickLabel', strrep(configNames(influenceIdx), '_', ' '));
    ylabel('Mean Fitness');
    title('Influence Mechanism Ablation');
    xtickangle(45);
    
    sgtitle('Dynamic PSO Neural Guidance Ablation Study');
    
    % Save figure
    resultsDir = getResultsDir();
    filename = fullfile(resultsDir, sprintf('Ablation_Study_Results_%s', datestr(now, 'yyyy-mm-dd_HH-MM-SS')));
    print(gcf, filename, '-dpdf', '-r300');
    fprintf('Ablation plots saved as: %s.pdf\n', filename);
end

