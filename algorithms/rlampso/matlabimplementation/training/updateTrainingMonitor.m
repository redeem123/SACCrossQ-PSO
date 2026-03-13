function updateTrainingMonitor(monitorFig, episodeFitness, episodeTimes, currentEpisode)
    % Update live training monitor with new episode data
    %
    % Inputs:
    %   monitorFig: Figure handle from createTrainingMonitor()
    %   episodeFitness: Array of fitness values for completed episodes
    %   episodeTimes: Array of time per episode (seconds)
    %   currentEpisode: Current episode number
    %
    % Usage:
    %   updateTrainingMonitor(monitorFig, stats.episodeFitness, stats.episodeTimes, 50);

    if ~isvalid(monitorFig)
        return;  % Figure was closed
    end

    % Get config and stored data
    config = monitorFig.UserData.config;
    startTime = monitorFig.UserData.startTime;

    % Update stored data
    monitorFig.UserData.episodeFitness = episodeFitness;
    monitorFig.UserData.episodeTimes = episodeTimes;

    % Throttle updates (max every 2 seconds)
    currentTime = toc(startTime);
    if currentTime - monitorFig.UserData.lastUpdate < 2
        return;
    end
    monitorFig.UserData.lastUpdate = currentTime;

    % Calculate statistics
    numEpisodes = length(episodeFitness);
    if numEpisodes == 0
        return;
    end

    % Moving average (50 episodes)
    windowSize = min(50, numEpisodes);
    movingAvg = zeros(numEpisodes, 1);
    for i = 1:numEpisodes
        startIdx = max(1, i - windowSize + 1);
        movingAvg(i) = mean(episodeFitness(startIdx:i));
    end

    % Reward: fitness improvement over time (initial fitness - current fitness)
    initialFitness = episodeFitness(1);
    rewardMetric = zeros(numEpisodes, 1);
    for i = 1:numEpisodes
        startIdx = max(1, i - windowSize + 1);
        rewardMetric(i) = initialFitness - mean(episodeFitness(startIdx:i));
    end

    % ===== UPDATE SUBPLOT 1: Episode Fitness =====
    subplot(2, 2, 1);

    % Update raw fitness points
    h = findobj(gca, 'Tag', 'raw_fitness');
    if ~isempty(h)
        set(h, 'XData', 1:numEpisodes, 'YData', episodeFitness);
    end

    % Update moving average line
    h = findobj(gca, 'Tag', 'avg_fitness');
    if ~isempty(h)
        set(h, 'XData', 1:numEpisodes, 'YData', movingAvg);
    end

    % Update ylim if needed
    maxFitness = min(max(episodeFitness), 5000);
    ylim([0, max(maxFitness * 1.1, 500)]);

    % ===== UPDATE SUBPLOT 2: Reward Metric (Fitness Improvement) =====
    subplot(2, 2, 2);

    h = findobj(gca, 'Tag', 'reward_metric');
    if ~isempty(h)
        set(h, 'XData', 1:numEpisodes, 'YData', rewardMetric);
    end

    % ===== UPDATE SUBPLOT 3: Training Progress =====
    subplot(2, 2, 3);

    timeElapsedMin = currentTime / 60;

    h = findobj(gca, 'Tag', 'progress_line');
    if ~isempty(h)
        set(h, 'XData', [0, timeElapsedMin], 'YData', [0, numEpisodes]);
    end

    % Update xlim if needed
    if timeElapsedMin > 180
        xlim([0, timeElapsedMin * 1.1]);
    end

    % ===== UPDATE SUBPLOT 4: Statistics Text =====
    subplot(2, 2, 4);

    % Calculate statistics
    currentFitness = episodeFitness(end);
    bestFitness = min(episodeFitness);
    avgLast50 = mean(episodeFitness(max(1, end-49):end));
    currentReward = initialFitness - avgLast50;  % Reward = improvement from initial

    % Estimate remaining time
    avgTimePerEpisode = mean(episodeTimes(max(1, end-19):end));  % Last 20 episodes
    remainingEpisodes = config.numEpisodes - numEpisodes;
    estimatedTimeRemaining = (remainingEpisodes * avgTimePerEpisode) / 60;  % minutes

    % Update text
    statsText = sprintf([
        'Training Statistics\n\n' ...
        'Episodes Completed: %d / %d (%.1f%%)\n' ...
        'Current Fitness: %.2f\n' ...
        'Best Fitness: %.2f\n' ...
        'Avg Last 50: %.2f\n' ...
        'Reward (Improvement): %.2f\n' ...
        'Time Elapsed: %.1f min\n' ...
        'Est. Time Remaining: %.1f min\n\n' ...
        'Workers: %d\n' ...
        'Episodes/Worker: %d\n' ...
        'GPU: %s\n' ...
        'Mode: %s\n'], ...
        numEpisodes, config.numEpisodes, 100 * numEpisodes / config.numEpisodes, ...
        currentFitness, ...
        bestFitness, ...
        avgLast50, ...
        currentReward, ...
        timeElapsedMin, ...
        estimatedTimeRemaining, ...
        config.numParallelWorkers, ...
        ceil(config.numEpisodes / config.numParallelWorkers), ...
        iif(config.useGPU, 'ON', 'OFF'), ...
        config.mode);

    h = findobj(gca, 'Tag', 'stats_text');
    if ~isempty(h)
        set(h, 'String', statsText);
    end

    drawnow limitrate;  % Limit draw rate for performance
end

function result = iif(condition, trueVal, falseVal)
    % Inline if function
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end
