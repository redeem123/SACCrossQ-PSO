function [trainedAgent, stats] = trainRLAMPSOonBenchmarks(fitnessFcn, dimensions, lowerBound, upperBound, numEpisodes, config, preloadedAgent, optimalValue)
    % Multi-episode DDPG training on a single benchmark function.
    %
    % Generic replacement for trainCECFunction_Parallel — works with any
    % fitness function handle from getBenchmarkFunction or CECBenchmarks.
    %
    % Inputs:
    %   fitnessFcn     — @(x) handle, x is Dx1 column vector
    %   dimensions     — search space dimensionality
    %   lowerBound     — scalar lower bound
    %   upperBound     — scalar upper bound
    %   numEpisodes    — number of training episodes
    %   config         — RLAMPSO_Config struct
    %   preloadedAgent — pre-initialized DDPG agent ([] for fresh init)
    %   optimalValue   — known global minimum (NaN to disable early-stop)

    if nargin < 7, preloadedAgent = []; end
    if nargin < 8, optimalValue = NaN; end

    if isempty(preloadedAgent)
        agent = initializeAgentAuto(config, '');
    else
        agent = preloadedAgent;
    end

    stats = struct();
    stats.episodeFitness = zeros(numEpisodes, 1);
    stats.episodeTimes   = zeros(numEpisodes, 1);
    stats.episodeSuccess = false(numEpisodes, 1);

    for episode = 1:numEpisodes
        t0 = tic;
        [bestFitness, ~, agent] = runRLAMPSOonBenchmark( ...
            fitnessFcn, dimensions, lowerBound, upperBound, optimalValue, config, agent);
        dt = toc(t0);

        stats.episodeFitness(episode) = bestFitness;
        stats.episodeTimes(episode)   = dt;
        if ~isnan(optimalValue)
            stats.episodeSuccess(episode) = abs(bestFitness - optimalValue) < 1e-6;
        end

        avgF = mean(stats.episodeFitness(max(1,episode-9):episode));
        fprintf('  [Ep %3d/%d] Fitness: %.4e | Avg: %.4e | Time: %5.1fs | Buffer: %d\n', ...
            episode, numEpisodes, bestFitness, avgF, dt, length(agent.replayBuffer));
    end

    stats.finalFitness = stats.episodeFitness(end);
    stats.bestFitness  = min(stats.episodeFitness);
    stats.avgFitness   = mean(stats.episodeFitness);
    stats.avgTime      = mean(stats.episodeTimes);
    stats.successRate  = sum(stats.episodeSuccess) / numEpisodes;

    trainedAgent = agent;
    fprintf('\nDone: Best=%.4e | Avg=%.4e | Success=%.1f%%\n', ...
        stats.bestFitness, stats.avgFitness, stats.successRate*100);
end
