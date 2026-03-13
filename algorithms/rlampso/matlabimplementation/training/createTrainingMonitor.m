function monitorFig = createTrainingMonitor(config)
    % Create live training monitor figure for RLAMPSO parallel training
    %
    % Displays:
    %   - Episode fitness over time (all workers)
    %   - Average fitness (smoothed)
    %   - Reward metric (cumulative fitness improvement)
    %   - Training progress (episodes completed)
    %
    % Usage:
    %   monitorFig = createTrainingMonitor(config);
    %   updateTrainingMonitor(monitorFig, episodeData);

    % Create figure
    monitorFig = figure('Name', 'RLAMPSO Training Monitor', ...
                        'NumberTitle', 'off', ...
                        'Position', [100, 100, 1200, 700], ...
                        'Color', 'w');

    % Title
    annotation('textbox', [0.3, 0.95, 0.4, 0.04], ...
               'String', sprintf('RLAMPSO Training Monitor - %s Mode', config.mode), ...
               'FontSize', 14, 'FontWeight', 'bold', ...
               'HorizontalAlignment', 'center', ...
               'EdgeColor', 'none');

    % Create 4 subplots

    % Subplot 1: Episode Fitness Over Time
    subplot(2, 2, 1);
    hold on; grid on;
    xlabel('Episode');
    ylabel('Fitness (lower is better)');
    title('Episode Fitness Progress');
    xlim([0, config.numEpisodes]);
    ylim([0, 5000]);

    % Plot lines (will be updated)
    plot(0, 0, 'b.', 'MarkerSize', 3, 'Tag', 'raw_fitness');  % Raw fitness points
    plot(0, 0, 'r-', 'LineWidth', 2, 'Tag', 'avg_fitness');   % Moving average
    plot([0, config.numEpisodes], [1000, 1000], 'g--', 'LineWidth', 1.5, 'Tag', 'success_threshold');
    legend({'Episode Fitness', 'Moving Avg (50 ep)', 'Success Threshold (1000)'}, 'Location', 'northeast');

    % Subplot 2: Reward Metric (Fitness Improvement)
    subplot(2, 2, 2);
    hold on; grid on;
    xlabel('Episode');
    ylabel('Reward (Fitness Improvement)');
    title('Cumulative Reward (Initial - Current Fitness)');
    xlim([0, config.numEpisodes]);
    ylim([-500, 2000]);

    plot(0, 0, 'm-', 'LineWidth', 2, 'Tag', 'reward_metric');
    plot([0, config.numEpisodes], [0, 0], 'k--', 'LineWidth', 1, 'Tag', 'zero_line');
    legend({'Reward (Improvement)', 'No Change'}, 'Location', 'southeast');

    % Subplot 3: Training Progress
    subplot(2, 2, 3);
    hold on; grid on;
    xlabel('Time (minutes)');
    ylabel('Episodes Completed');
    title('Training Progress');
    xlim([0, 180]);  % 3 hours max
    ylim([0, config.numEpisodes]);

    plot(0, 0, 'b-', 'LineWidth', 2, 'Tag', 'progress_line');
    % Expected progress line (linear)
    plot([0, 150], [0, config.numEpisodes], 'r--', 'LineWidth', 1, 'Tag', 'expected_progress');
    legend({'Actual', 'Expected (2.5 hrs)'}, 'Location', 'southeast');

    % Subplot 4: Statistics Text
    subplot(2, 2, 4);
    axis off;

    statsText = sprintf([
        'Training Statistics\n\n' ...
        'Episodes Completed: 0 / %d (0.0%%)\n' ...
        'Current Fitness: N/A\n' ...
        'Best Fitness: N/A\n' ...
        'Avg Last 50: N/A\n' ...
        'Reward (Improvement): N/A\n' ...
        'Time Elapsed: 0.0 min\n' ...
        'Est. Time Remaining: %.1f min\n\n' ...
        'Workers: %d\n' ...
        'Episodes/Worker: %d\n' ...
        'GPU: %s\n' ...
        'Mode: %s\n'], ...
        config.numEpisodes, ...
        150.0, ...  % Estimated total time
        config.numParallelWorkers, ...
        ceil(config.numEpisodes / config.numParallelWorkers), ...
        iif(config.useGPU, 'ON', 'OFF'), ...
        config.mode);

    text(0.05, 0.95, statsText, ...
         'FontSize', 11, 'FontName', 'FixedWidth', ...
         'VerticalAlignment', 'top', ...
         'Tag', 'stats_text');

    % Store data in figure UserData
    monitorFig.UserData = struct();
    monitorFig.UserData.config = config;
    monitorFig.UserData.episodeFitness = [];
    monitorFig.UserData.episodeTimes = [];
    monitorFig.UserData.startTime = tic;
    monitorFig.UserData.lastUpdate = 0;

    drawnow;
end

function result = iif(condition, trueVal, falseVal)
    % Inline if function
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end
