function [trainedAgent, stats] = trainCECFunction_Parallel(functionID, numEpisodes, config, preloadedAgent)
    % Sequential training on a single CEC benchmark function
    %
    % Paper training procedure: run multiple episodes of PSO with DDPG online learning.

    if nargin < 4, preloadedAgent = []; end

    if isempty(preloadedAgent)
        agent = initializeAgentAuto(config, '');
    else
        agent = preloadedAgent;
    end

    dimensions = 30;
    [~, optimalValue] = CECBenchmarks(zeros(dimensions, 1), functionID);

    stats = struct();
    stats.episodeFitness = zeros(numEpisodes, 1);
    stats.episodeTimes   = zeros(numEpisodes, 1);
    stats.episodeSuccess = false(numEpisodes, 1);

    for episode = 1:numEpisodes
        t0 = tic;
        [bestFitness, ~, agent] = runRLAMPSOonCEC(functionID, dimensions, config, agent);
        dt = toc(t0);

        stats.episodeFitness(episode) = bestFitness;
        stats.episodeTimes(episode)   = dt;
        stats.episodeSuccess(episode) = abs(bestFitness - optimalValue) < 1e-6;

        % Paper Eq. 14: noise is constant N(0, 0.5), no decay

        avgF = mean(stats.episodeFitness(max(1,episode-9):episode));
        fprintf('  [Ep %3d/%d] F%d | Fitness: %.4e | Avg: %.4e | Time: %5.1fs | Buffer: %d\n', ...
            episode, numEpisodes, functionID, bestFitness, avgF, dt, length(agent.replayBuffer));
    end

    stats.finalFitness = stats.episodeFitness(end);
    stats.bestFitness  = min(stats.episodeFitness);
    stats.avgFitness   = mean(stats.episodeFitness);
    stats.avgTime      = mean(stats.episodeTimes);
    stats.successRate  = sum(stats.episodeSuccess) / numEpisodes;

    trainedAgent = agent;
    fprintf('\nF%d done: Best=%.4e | Avg=%.4e | Success=%.1f%%\n', ...
        functionID, stats.bestFitness, stats.avgFitness, stats.successRate*100);
end
