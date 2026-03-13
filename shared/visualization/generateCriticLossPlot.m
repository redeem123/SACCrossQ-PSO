function generateCriticLossPlot(results, algorithms, scenarioIdx)
    % Generate critic loss / TD error plot for learning-based algorithms.

    validAlgorithms = {};
    maxLen = 0;
    for i = 1:numel(algorithms)
        alg = algorithms{i};
        if isfield(results, alg.fieldName) && isfield(results.(alg.fieldName), 'criticLossHistory')
            losses = results.(alg.fieldName).criticLossHistory;
            if ~isempty(losses)
                validAlgorithms{end + 1, 1} = alg.fieldName; %#ok<AGROW>
                validAlgorithms{end, 2} = alg.displayName;
                maxLen = max(maxLen, numel(losses));
            end
        end
    end

    if isempty(validAlgorithms)
        fprintf('No critic loss history found for plotting.\n');
        return;
    end

    smoothWindow = max(5, floor(maxLen * 0.05));
    if mod(smoothWindow, 2) == 0
        smoothWindow = smoothWindow + 1;
    end

    h_fig = figure('Position', [100, 100, 1200, 800], 'Color', 'w');
    hold on;

    colors = [
        1.0, 0.0, 0.0;
        0.0, 0.0, 1.0;
        0.0, 0.8, 0.0;
        1.0, 0.5, 0.0;
        0.5, 0.0, 0.5;
        0.0, 0.8, 0.8;
        1.0, 0.0, 1.0;
        0.8, 0.8, 0.0;
        0.5, 0.5, 0.5;
        0.0, 0.0, 0.0;
    ];

    legendLabels = {};
    plotHandles = [];

    for i = 1:size(validAlgorithms, 1)
        algFieldName = validAlgorithms{i, 1};
        algDisplayName = validAlgorithms{i, 2};
        losses = results.(algFieldName).criticLossHistory(:);
        if isempty(losses)
            continue;
        end

        smoothLosses = smoothSeries(losses, smoothWindow);
        iterations = 1:numel(losses);
        color = colors(mod(i - 1, size(colors, 1)) + 1, :);
        h = plot(iterations, smoothLosses, '-', 'Color', color, 'LineWidth', 2);
        plotHandles = [plotHandles, h];
        legendLabels{end + 1} = algDisplayName;
    end

    xlabel('Iteration', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Critic Loss / TD Error (smoothed)', 'FontSize', 14, 'FontWeight', 'bold');

    if ~isempty(plotHandles)
        legend(plotHandles, legendLabels, 'Location', 'best', 'FontSize', 10);
    end

    grid on;
    set(gca, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1.5);

    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [12, 8]);
    set(gcf, 'PaperPosition', [0, 0, 12, 8]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'Renderer', 'painters');

    resultsDir = getResultsDir();
    filename = fullfile(resultsDir, sprintf('Critic_Loss_Scenario%d', scenarioIdx));
    print(gcf, filename, '-dpdf', '-r300');

    fprintf('Critic loss plot saved as: %s.pdf\n', filename);
    hold off;
end

function smoothValues = smoothSeries(values, window)
    if numel(values) < 2
        smoothValues = values;
        return;
    end
    window = max(1, min(window, numel(values)));
    smoothValues = movmean(values, window, 'omitnan');
end
