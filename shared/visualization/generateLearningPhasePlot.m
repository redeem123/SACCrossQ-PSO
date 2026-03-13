function generateLearningPhasePlot(results, algorithms, scenarioIdx)
    % Generate learning-phase indicators for adaptive algorithms.

    validAlgorithms = {};
    for i = 1:numel(algorithms)
        alg = algorithms{i};
        if isfield(results, alg.fieldName) && ...
           isfield(results.(alg.fieldName), 'w_history') && ...
           isfield(results.(alg.fieldName), 'c1_history') && ...
           isfield(results.(alg.fieldName), 'c2_history')
            validAlgorithms{end+1, 1} = alg.fieldName; %#ok<AGROW>
            validAlgorithms{end, 2} = alg.displayName;
        end
    end

    if isempty(validAlgorithms)
        fprintf('No parameter history found for learning-phase plot.\n');
        return;
    end

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
    ];

    smoothWindow = 10;

    h_fig = figure('Position', [100, 100, 1400, 900], 'Color', 'w');

    % Subplot 1: parameter change magnitude
    subplot(2, 1, 1);
    hold on;
    legendLabels = {};
    plotHandles = [];

    for i = 1:size(validAlgorithms, 1)
        algFieldName = validAlgorithms{i, 1};
        algDisplayName = validAlgorithms{i, 2};
        data = results.(algFieldName);

        w = data.w_history(:);
        c1 = data.c1_history(:);
        c2 = data.c2_history(:);
        len = min([numel(w), numel(c1), numel(c2)]);
        if len < 2
            continue;
        end
        w = w(1:len);
        c1 = c1(1:len);
        c2 = c2(1:len);

        delta = zeros(len, 1);
        for t = 2:len
            dw = w(t) - w(t-1);
            dc1 = c1(t) - c1(t-1);
            dc2 = c2(t) - c2(t-1);
            delta(t) = sqrt(dw * dw + dc1 * dc1 + dc2 * dc2);
        end
        deltaSmooth = movmean(delta, smoothWindow);

        color = colors(mod(i-1, size(colors, 1)) + 1, :);
        h = plot(1:len, deltaSmooth, '-', 'Color', color, 'LineWidth', 2);
        plotHandles = [plotHandles, h];
        legendLabels{end+1} = algDisplayName;
    end

    xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('||\Delta CP||', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    if ~isempty(plotHandles)
        legend(plotHandles, legendLabels, 'Location', 'best', 'FontSize', 8);
    end
    set(gca, 'FontSize', 10);
    hold off;

    % Subplot 2: improvement per iteration
    subplot(2, 1, 2);
    hold on;
    plotHandles = [];

    for i = 1:size(validAlgorithms, 1)
        algFieldName = validAlgorithms{i, 1};
        data = results.(algFieldName);
        if ~isfield(data, 'convergenceHistory')
            continue;
        end

        conv = data.convergenceHistory(:);
        len = numel(conv);
        if len < 2
            continue;
        end
        improvement = zeros(len, 1);
        for t = 2:len
            improvement(t) = max(0, conv(t-1) - conv(t));
        end
        improvementSmooth = movmean(improvement, smoothWindow);

        color = colors(mod(i-1, size(colors, 1)) + 1, :);
        h = plot(1:len, improvementSmooth, '-', 'Color', color, 'LineWidth', 2);
        plotHandles = [plotHandles, h];
    end

    xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('\Delta Fitness', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 10);
    hold off;

    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [14, 9]);
    set(gcf, 'PaperPosition', [0, 0, 14, 9]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'Renderer', 'painters');

    resultsDir = getResultsDir();
    filename = fullfile(resultsDir, sprintf('Learning_Phase_Scenario%d', scenarioIdx));
    print(gcf, filename, '-dpdf', '-r300');

    fprintf('Learning-phase plot saved as: %s.pdf\n', filename);
end
