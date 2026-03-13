function generateRewardPlot(results, algorithms, scenarioIdx)
    % Generate reward comparison plot for learning-based algorithms.

    validAlgorithms = {};
    maxLen = 0;
    for i = 1:numel(algorithms)
        alg = algorithms{i};
        algResults = resolveAlgorithmResults(results, alg.fieldName);
        if ~isempty(algResults) && isfield(algResults, 'rewardHistory')
            rewards = algResults.rewardHistory;
            if ~isempty(rewards)
                validAlgorithms{end + 1, 1} = alg.fieldName; %#ok<AGROW>
                validAlgorithms{end, 2} = alg.displayName;
                maxLen = max(maxLen, numel(rewards));
            end
        end
    end

    if isempty(validAlgorithms)
        fprintf('No reward history found for plotting.\n');
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
        algResults = resolveAlgorithmResults(results, algFieldName);
        rewards = algResults.rewardHistory(:);
        if isempty(rewards)
            continue;
        end

        if isDQNAlgorithm(algFieldName, algDisplayName)
            scale = max(abs(rewards));
            if scale > 0
                rewards = rewards / scale;
            end
        end

        smoothRewards = smoothSeries(rewards, smoothWindow);
        iterations = 1:numel(rewards);
        color = colors(mod(i - 1, size(colors, 1)) + 1, :);
        h = plot(iterations, smoothRewards, '-', 'Color', color, 'LineWidth', 2);
        plotHandles = [plotHandles, h];
        legendLabels{end + 1} = algDisplayName;
    end

    xlabel('Iteration', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Reward (smoothed)', 'FontSize', 14, 'FontWeight', 'bold');

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
    filename = fullfile(resultsDir, sprintf('Reward_Comparison_Scenario%d', scenarioIdx));
    print(gcf, filename, '-dpdf', '-r300');

    fprintf('Reward plot saved as: %s.pdf\n', filename);
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

function tf = isDQNAlgorithm(fieldName, displayName)
    tf = contains(fieldName, 'DQN_PSO') || contains(displayName, 'DQN-PSO');
end

function algResults = resolveAlgorithmResults(results, fieldName)
    algResults = [];
    candidateFields = {fieldName};

    if strcmp(fieldName, 'RRSACPSO')
        candidateFields{end + 1} = 'RRSACPSO_Online'; %#ok<AGROW>
    elseif strcmp(fieldName, 'RRSACPSO_Online')
        candidateFields{end + 1} = 'RRSACPSO'; %#ok<AGROW>
    end

    for idx = 1:numel(candidateFields)
        candidate = candidateFields{idx};
        if isfield(results, candidate)
            algResults = results.(candidate);
            return;
        end
    end
end
