function generateConvergencePlot(results, algorithms, scenarioIdx)
    % Generate convergence plot for all algorithms
    
    h_fig = figure('Position', [100, 100, 1200, 800], 'Color', 'w');
    hold on;
    
    % Colors for each algorithm
    colors = [
        1.0, 0.0, 0.0;  % Red
        0.0, 0.0, 1.0;  % Blue  
        0.0, 0.8, 0.0;  % Green
        1.0, 0.5, 0.0;  % Orange
        0.5, 0.0, 0.5;  % Purple
        0.0, 0.8, 0.8;  % Cyan
        1.0, 0.0, 1.0;  % Magenta
        0.8, 0.8, 0.0;  % Yellow
        0.5, 0.5, 0.5;  % Gray
        0.0, 0.0, 0.0;  % Black
        1.0, 0.4, 0.7;  % Pink
        0.2, 0.6, 0.2;  % Dark Green
        0.6, 0.2, 0.8;  % Dark Purple
        0.8, 0.6, 0.2;  % Brown
    ];
    
    legendLabels = {};
    plotHandles = [];
    
    for i = 1:length(algorithms)
        alg = algorithms{i};
        algName = alg.fieldName;
        
        if isfield(results, algName) && isfield(results.(algName), 'convergenceHistory')
            convergence = results.(algName).convergenceHistory;
            
            if ~isempty(convergence)
                iterations = 1:length(convergence);
                color = colors(mod(i-1, size(colors, 1)) + 1, :);
                
                h = plot(iterations, convergence, '-', 'Color', color, 'LineWidth', 2, 'MarkerSize', 4);
                plotHandles = [plotHandles, h];
                legendLabels{end+1} = alg.displayName;
            end
        end
    end
    
    xlabel('Iteration', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Best Fitness', 'FontSize', 14, 'FontWeight', 'bold');
    % title(sprintf('Scenario %d: Convergence Comparison', scenarioIdx), 'FontSize', 16, 'FontWeight', 'bold');
    
    if ~isempty(plotHandles)
        legend(plotHandles, legendLabels, 'Location', 'northeast', 'FontSize', 10);
    end
    
    grid on;
    set(gca, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1.5);
    set(gca, 'YScale', 'log'); % Log scale for better visualization
    
    % Set up figure for PDF export
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [12, 8]);
    set(gcf, 'PaperPosition', [0, 0, 12, 8]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'Renderer', 'painters');
    
    % Save to PDF
    resultsDir = getResultsDir();
    filename = fullfile(resultsDir, sprintf('Convergence_Comparison_Scenario%d', scenarioIdx));
    print(gcf, filename, '-dpdf', '-r300');

    fprintf('Convergence plot saved as: %s.pdf\n', filename);
    hold off;
end

