function generateParameterTrackingPlot(results, algorithms, scenarioIdx)
    % Generate parameter tracking plot for adaptive algorithms
    
    % Define which algorithms have parameter adaptation
    adaptiveAlgorithms = {
        'RLAMPSO_Online_Global', 'RLAMPSO [36]';
        'DQN_PSO_Online_Global', 'DQN-PSO [34]';
        'PPO_PSO_Online_Global', 'PPO-PSO';
        'MPSORL', 'MPSORL';
        'SACPSO_Global', 'SACPSO-Global';
        'AFSACPSO_Online', 'AFSACPSO';
        'SAEPSO_ParamsOnly', 'SAEPSO* [11]';
        'UAPSO', 'UAPSO [25]';
    };
    
    % Count how many adaptive algorithms we have data for
    validAlgorithms = {};
    for i = 1:size(adaptiveAlgorithms, 1)
        algFieldName = adaptiveAlgorithms{i, 1};
        if algorithmRequested(algorithms, algFieldName) && isfield(results, algFieldName) && hasParameterSeries(results.(algFieldName))
            validAlgorithms{end+1, 1} = algFieldName;
            validAlgorithms{end, 2} = adaptiveAlgorithms{i, 2};
        end
    end
    
    if isempty(validAlgorithms)
        fprintf('No parameter adaptation data found for plotting.\n');
        return;
    end
    
    % Create figure with subplots for w, c1, c2
    h_fig = figure('Position', [100, 100, 1400, 1000], 'Color', 'w');
    
    % Colors for algorithms
    colors = [
        1.0, 0.0, 0.0;  % Red
        0.0, 0.0, 1.0;  % Blue  
        0.0, 0.8, 0.0;  % Green
        1.0, 0.5, 0.0;  % Orange
        0.5, 0.0, 0.5;  % Purple
        0.0, 0.8, 0.8;  % Cyan
        1.0, 0.0, 1.0;  % Magenta
    ];
    
    % Subplot 1: Inertia Weight (w)
    subplot(3, 1, 1);
    hold on;
    legendLabels = {};
    plotHandles = [];
    
    for i = 1:size(validAlgorithms, 1)
        algFieldName = validAlgorithms{i, 1};
        algDisplayName = validAlgorithms{i, 2};
        
        [w_data, w_lower, w_upper, hasSamples] = extractSeries(results.(algFieldName), 'w');
        if ~isempty(w_data)
            iterations = 1:length(w_data);
            color = colors(mod(i-1, size(colors, 1)) + 1, :);
            if hasSamples
                fill([iterations, fliplr(iterations)], ...
                    [w_lower(:)', fliplr(w_upper(:)')], ...
                    color, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            end
            h = plot(iterations, w_data, '-', 'Color', color, 'LineWidth', 2);
            plotHandles = [plotHandles, h];
            legendLabels{end+1} = algDisplayName;
        end
    end
    
    xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Inertia Weight (w)', 'FontSize', 12, 'FontWeight', 'bold');
    % title(sprintf('Scenario %d: Parameter Adaptation - Inertia Weight', scenarioIdx), 'FontSize', 14, 'FontWeight', 'bold');
    yline(0.1, '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');
    yline(0.9, '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');
    ylim([0.1, 0.9]);
    grid on;
    if ~isempty(plotHandles)
        legend(plotHandles, legendLabels, 'Location', 'best', 'FontSize', 8);
    end
    set(gca, 'FontSize', 10);
    hold off;
    
    % Subplot 2: Cognitive Coefficient (c1)
    subplot(3, 1, 2);
    hold on;
    plotHandles = [];
    
    for i = 1:size(validAlgorithms, 1)
        algFieldName = validAlgorithms{i, 1};
        
        [c1_data, c1_lower, c1_upper, hasSamples] = extractSeries(results.(algFieldName), 'c1');
        if ~isempty(c1_data)
            iterations = 1:length(c1_data);
            color = colors(mod(i-1, size(colors, 1)) + 1, :);
            if hasSamples
                fill([iterations, fliplr(iterations)], ...
                    [c1_lower(:)', fliplr(c1_upper(:)')], ...
                    color, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            end
            h = plot(iterations, c1_data, '-', 'Color', color, 'LineWidth', 2);
            plotHandles = [plotHandles, h];
        end
    end
    
    xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Cognitive Coefficient (c1)', 'FontSize', 12, 'FontWeight', 'bold');
    % title(sprintf('Scenario %d: Parameter Adaptation - Cognitive Coefficient', scenarioIdx), 'FontSize', 14, 'FontWeight', 'bold');
    yline(0.5, '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');
    yline(2.5, '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');
    ylim([0.5, 2.5]);
    grid on;
    set(gca, 'FontSize', 10);
    hold off;
    
    % Subplot 3: Social Coefficient (c2)
    subplot(3, 1, 3);
    hold on;
    plotHandles = [];
    
    for i = 1:size(validAlgorithms, 1)
        algFieldName = validAlgorithms{i, 1};
        
        [c2_data, c2_lower, c2_upper, hasSamples] = extractSeries(results.(algFieldName), 'c2');
        if ~isempty(c2_data)
            iterations = 1:length(c2_data);
            color = colors(mod(i-1, size(colors, 1)) + 1, :);
            if hasSamples
                fill([iterations, fliplr(iterations)], ...
                    [c2_lower(:)', fliplr(c2_upper(:)')], ...
                    color, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            end
            h = plot(iterations, c2_data, '-', 'Color', color, 'LineWidth', 2);
            plotHandles = [plotHandles, h];
        end
    end
    
    xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Social Coefficient (c2)', 'FontSize', 12, 'FontWeight', 'bold');
    % title(sprintf('Scenario %d: Parameter Adaptation - Social Coefficient', scenarioIdx), 'FontSize', 14, 'FontWeight', 'bold');
    yline(0.5, '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');
    yline(2.5, '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');
    ylim([0.5, 2.5]);
    grid on;
    set(gca, 'FontSize', 10);
    hold off;
    
    % Adjust subplot spacing
    % sgtitle(sprintf('Parameter Evolution (Scenario %d)', scenarioIdx), 'FontSize', 16, 'FontWeight', 'bold');
    
    % Set up figure for PDF export
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [14, 10]);
    set(gcf, 'PaperPosition', [0, 0, 14, 10]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'Renderer', 'painters');
    
    % Save to PDF
    resultsDir = getResultsDir();
    filename = fullfile(resultsDir, sprintf('Parameter_Tracking_Scenario%d', scenarioIdx));
    print(gcf, filename, '-dpdf', '-r300');

    fprintf('Parameter tracking plot saved as: %s.pdf\n', filename);
end

function requested = algorithmRequested(algorithms, fieldName)
    requested = false;
    for i = 1:numel(algorithms)
        if strcmp(algorithms{i}.fieldName, fieldName)
            requested = true;
            return;
        end
    end
end

function hasData = hasParameterSeries(algResults)
    hasData = false;
    if isfield(algResults, 'w_history') || isfield(algResults, 'w_samples')
        hasData = true;
        return;
    end
    if isfield(algResults, 'avgInertiaHistory') || isfield(algResults, 'avgC1History') || isfield(algResults, 'avgC2History')
        hasData = true;
        return;
    end
    if isfield(algResults, 'parameterHistory')
        if isfield(algResults.parameterHistory, 'w') || isfield(algResults.parameterHistory, 'w_samples') || ...
                isfield(algResults.parameterHistory, 'avgInertia') || isfield(algResults.parameterHistory, 'avgC1') || ...
                isfield(algResults.parameterHistory, 'avgC2')
            hasData = true;
        end
    end
end

function [meanSeries, lowerBand, upperBand, hasSamples] = extractSeries(algResults, baseName)
    meanSeries = [];
    lowerBand = [];
    upperBand = [];
    hasSamples = false;

    sampleField = [baseName '_samples'];
    historyField = [baseName '_history'];
    samples = [];

    if isfield(algResults, sampleField)
        samples = algResults.(sampleField);
    elseif isfield(algResults, 'parameterHistory') && isfield(algResults.parameterHistory, sampleField)
        samples = algResults.parameterHistory.(sampleField);
    end

    if ~isempty(samples)
        meanSeries = mean(samples, 2);
        spread = std(samples, 0, 2);
        lowerBand = meanSeries - spread;
        upperBand = meanSeries + spread;
        meanSeries = meanSeries(:)';
        lowerBand = lowerBand(:);
        upperBand = upperBand(:);
        hasSamples = true;
        return;
    end

    if isfield(algResults, historyField)
        meanSeries = algResults.(historyField);
    elseif strcmp(baseName, 'w') && isfield(algResults, 'avgInertiaHistory')
        meanSeries = algResults.avgInertiaHistory;
    elseif strcmp(baseName, 'c1') && isfield(algResults, 'avgC1History')
        meanSeries = algResults.avgC1History;
    elseif strcmp(baseName, 'c2') && isfield(algResults, 'avgC2History')
        meanSeries = algResults.avgC2History;
    elseif isfield(algResults, 'parameterHistory') && isfield(algResults.parameterHistory, baseName)
        meanSeries = algResults.parameterHistory.(baseName);
    elseif strcmp(baseName, 'w') && isfield(algResults, 'parameterHistory') && isfield(algResults.parameterHistory, 'avgInertia')
        meanSeries = algResults.parameterHistory.avgInertia;
    elseif strcmp(baseName, 'c1') && isfield(algResults, 'parameterHistory') && isfield(algResults.parameterHistory, 'avgC1')
        meanSeries = algResults.parameterHistory.avgC1;
    elseif strcmp(baseName, 'c2') && isfield(algResults, 'parameterHistory') && isfield(algResults.parameterHistory, 'avgC2')
        meanSeries = algResults.parameterHistory.avgC2;
    end
    meanSeries = meanSeries(:)';
end

