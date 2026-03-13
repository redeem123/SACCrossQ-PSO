function trainingStats = trainDQNPSO(config)
%TRAINDQNPSO Offline training utility for DQN-PSO parameter adaptation.
%
%   trainingStats = trainDQNPSO(config) runs multiple UAV planning episodes
%   using the DQN-PSO global planner and stores the resulting model.
%
%   Required config fields:
%       numEpisodes   - Number of offline training episodes
%       savePath      - Destination .mat file for the trained model
%       paramMode     - 'global' (paper)
%       popSize       - PSO population
%       maxIterations - Max PSO iterations per episode
%       w, c1, c2     - Initial PSO coefficients
%       mapSize       - [X Y Z] map dimensions used to sample scenarios

    arguments
        config.numEpisodes (1,1) double {mustBePositive}
        config.savePath (1,1) string
        config.paramMode (1,1) string {mustBeMember(config.paramMode, ["global","5subgroup","per-particle"])}
        config.popSize (1,1) double {mustBePositive}
        config.maxIterations (1,1) double {mustBePositive}
        config.w (1,1) double
        config.c1 (1,1) double
        config.c2 (1,1) double
        config.mapSize (1,3) double {mustBePositive}
    end

    if isfield(config, 'alpha')
        alpha = config.alpha;
    else
        alpha = 0.1;
    end

    if isfield(config, 'gamma')
        gamma = config.gamma;
    else
        gamma = 0.9;
    end

    if isfield(config, 'epsilon')
        epsilon = config.epsilon;
    else
        epsilon = 0.2;
    end

    scenarioConfig = struct('mapSize', config.mapSize);
    numEpisodes = config.numEpisodes;

    trainingStats = struct();
    trainingStats.episodeFitness = zeros(numEpisodes, 1);
    trainingStats.episodePaths = cell(numEpisodes, 1);
    trainingStats.episodeConvergence = cell(numEpisodes, 1);
    trainingStats.paramMode = config.paramMode;
    trainingStats.totalTimeSeconds = 0;

    if ~strcmp(config.paramMode, "global")
        warning('DQN-PSO (paper) is homogeneous; forcing paramMode to global.');
        config.paramMode = "global";
    end

    dqnModel = struct();
    dqnModel.qNet = [];
    dqnModel.targetNet = [];
    dqnModel.gamma = gamma;
    dqnModel.epsilon = epsilon;
    dqnModel.paramMode = config.paramMode;
    dqnModel.lastEpisode = 0;

    trainingStart = tic;
    for episode = 1:numEpisodes
        [startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY, mapSize] = ...
            generateTrainingScenario(scenarioConfig, episode);

        [episodePath, convergenceHistory, episodeStats] = globalPathPlanningDQNPSO( ...
            startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY, mapSize, ...
            config.popSize, config.maxIterations, config.w, config.c1, config.c2, ...
            config.paramMode, dqnModel);

        if isfield(episodeStats, 'dqnModel')
            dqnModel = episodeStats.dqnModel;
        end
        dqnModel.lastEpisode = episode;

        trainingStats.episodeFitness(episode) = episodeStats.actualBestFitness;
        trainingStats.episodePaths{episode} = episodePath;
        trainingStats.episodeConvergence{episode} = convergenceHistory;
    end
    trainingStats.totalTimeSeconds = toc(trainingStart);
    trainingStats.finalModel = dqnModel;

    if strlength(config.savePath) > 0
        saveDir = fileparts(config.savePath);
        if ~isempty(saveDir) && ~exist(saveDir, 'dir')
            mkdir(saveDir);
        end
        dqnModelSaved = dqnModel; %#ok<NASGU>
        save(config.savePath, 'dqnModelSaved', 'trainingStats');
    end
end
