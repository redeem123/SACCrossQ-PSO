function [globalPath, algorithmSpecificStats] = callGlobalPlanningAlgorithm(algorithmName, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params)
    % Call the appropriate global planning algorithm with convergence and parameter tracking
    
    algorithmSpecificStats = struct();
    
    switch algorithmName
        case 'RLAMPSO'
            pretrainedPath = '';
            if isfield(params, 'pretrainedNetworkPath')
                pretrainedPath = params.pretrainedNetworkPath;
            end

            if isfield(params, 'configMode')
                config = RLAMPSO_Config(params.configMode);
            else
                config = RLAMPSO_Config('baseline');
            end

            if isfield(params, 'paramMode')
                config.paramMode = params.paramMode;
            else
                config.paramMode = '5subgroup';
            end

            if isfield(params, 'popSize')
                config.popSize = params.popSize;
            end
            if isfield(params, 'maxIterations')
                config.maxIterations = params.maxIterations;
            end

            if isfield(params, 'configOverrides') && isstruct(params.configOverrides)
                overrideFields = fieldnames(params.configOverrides);
                for iField = 1:numel(overrideFields)
                    fieldName = overrideFields{iField};
                    config.(fieldName) = params.configOverrides.(fieldName);
                end
            end

            % Update config based on paramMode (CRITICAL!)
            switch config.paramMode
                case 'global'
                    config.actionSize = 4;
                    config.numSubgroups = 1;
                case '5subgroup'
                    config.actionSize = 20;
                    config.numSubgroups = 5;
                case 'per-particle'
                    config.actionSize = config.popSize * 4;
                    config.numSubgroups = config.popSize;
            end

            [globalPath, convergence, paramHist, algorithmSpecificStats] = globalPathPlanningRLAMPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.initialW, params.initialC1, params.initialC2, pretrainedPath, config);
            algorithmSpecificStats.convergenceHistory = convergence;
            algorithmSpecificStats.parameterHistory = paramHist;
            
        case 'PSO'
            [globalPath, convergence, algorithmSpecificStats] = globalPathPlanningPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.w, params.c1, params.c2);
            algorithmSpecificStats.convergenceHistory = convergence;

        case 'SVPSO'
            [globalPath, sphericalStats, vectorStats, convergence, algorithmSpecificStats] = globalPathPlanningSVPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.w, params.c1, params.c2);
            algorithmSpecificStats.sphericalStats = sphericalStats;
            algorithmSpecificStats.vectorStats = vectorStats;
            algorithmSpecificStats.convergenceHistory = convergence;
   
        case 'PSO_TVAC'
            [globalPath, convergence, paramHist, algorithmSpecificStats] = globalPathPlanningPSOTVAC(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.initialW, params.initialC1, params.initialC2);
            algorithmSpecificStats.convergenceHistory = convergence;
            algorithmSpecificStats.parameterHistory = paramHist;

        case 'PSO_LDIW'
            [globalPath, convergence, paramHist, algorithmSpecificStats] = globalPathPlanningPSOLDIW( ...
                startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, ...
                params.popSize, params.maxIterations, params.initialW, params.initialC1, params.initialC2);
            algorithmSpecificStats.convergenceHistory = convergence;
            algorithmSpecificStats.parameterHistory = paramHist;

        case 'RLNNPSO'
            % RL-NNPSO: Reinforcement Learning based Neural-Guided PSO with TD3
            % Pass pre-trained network path if available (14th parameter)
            if isfield(params, 'pretrainedNetworkPath')
                [globalPath, convergence, algorithmSpecificStats] = globalPathPlanningRLNNPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.fixedW, params.fixedC1, params.fixedC2, params.pretrainedNetworkPath);
            else
                [globalPath, convergence, algorithmSpecificStats] = globalPathPlanningRLNNPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.fixedW, params.fixedC1, params.fixedC2);
            end
            algorithmSpecificStats.convergenceHistory = convergence;

        case 'SAEPSO'
            [globalPath, convergence, paramHist, algorithmSpecificStats] = globalPathPlanningSAEPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.c_min, params.c_max, params.w_min, params.w_max, params.useOtherImprovements);
            algorithmSpecificStats.convergenceHistory = convergence;
            algorithmSpecificStats.parameterHistory = paramHist;

        case 'ANNPSO'
            [globalPath, convergence, annStats, algorithmSpecificStats] = globalPathPlanningANNPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.w, params.c1, params.c2, params.hiddenNeurons);
            algorithmSpecificStats.convergenceHistory = convergence;
            algorithmSpecificStats.annStats = annStats;

        case 'IGPSO'
            [globalPath, convergence, igStats, algorithmSpecificStats] = globalPathPlanningIGPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.positiveSampleRatio, params.hiddenNeurons);
            algorithmSpecificStats.convergenceHistory = convergence;
            algorithmSpecificStats.igStats = igStats;

        case {'AFSACPSO', 'AFSACPSO_Online'}
            % AFSACPSO: pre-trained SAC policy + PSO + NM local search.
            config = AFSACPSO_Config();
            config.numEpisodes = 1;  % Single episode for deployment
            config.mapSize = mapSize;

            if isfield(params, 'popSize')
                config.popSize = params.popSize;
            end

            if isfield(params, 'maxIterations')
                config.maxIterations = params.maxIterations;
            end

            if isfield(params, 'paramMode')
                config.paramMode = params.paramMode;
            end
            % paramMode defaults from AFSACPSO_Config (no forced override)

            if isfield(params, 'configOverrides') && isstruct(params.configOverrides)
                overrideFields = fieldnames(params.configOverrides);
                for iField = 1:numel(overrideFields)
                    fieldName = overrideFields{iField};
                    config.(fieldName) = params.configOverrides.(fieldName);
                end
            end
            config = applyParamMode(config);

            % Paper state mode uses ns+3 dimensional state (popSize + 3)
            if isfield(config, 'stateMode') && strcmp(config.stateMode, 'paper')
                config.stateSize = config.popSize + 3;
            end

            [globalPath, bestFitness, fitnessHistory, agent, stateEncoder, paramHistory, learningStats] = ...
                globalPathPlanningAFSACPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config);

            intermediateWaypoints = globalPath(2:end-1, :);
            position = reshape(intermediateWaypoints', 1, []);
            numWaypoints = size(intermediateWaypoints, 1);
            [~, fitnessComponents] = evaluatePathFitness(position, startPoint, goalPoint, ...
                dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

            algorithmSpecificStats.convergenceHistory = fitnessHistory;
            algorithmSpecificStats.fitnessComponents = fitnessComponents;
            algorithmSpecificStats.finalFitness = bestFitness;
            algorithmSpecificStats.actualBestFitness = bestFitness;
            algorithmSpecificStats.parameterHistory = paramHistory;
            algorithmSpecificStats.rewardHistory = learningStats.rewardHistory;
            algorithmSpecificStats.criticLossHistory = learningStats.criticLossHistory;
            algorithmSpecificStats.agent = agent;
            algorithmSpecificStats.stateEncoder = stateEncoder;

        % ========================================================================
        % NEW ALGORITHMS FOR TOP-TIER JOURNAL COMPARISON
        % ========================================================================

        case 'UAPSO'
            % UAPSO: Self-Adapting Control Parameters (Isiet & Gadala, 2019)
            [globalPath, convergence, paramHist, algorithmSpecificStats] = globalPathPlanningUAPSO( ...
                startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, ...
                params.popSize, params.maxIterations, params.c_min, params.c_max);
            algorithmSpecificStats.convergenceHistory = convergence;
            algorithmSpecificStats.parameterHistory = paramHist;

        case 'MPSORL'
            [globalPath, convergence, algorithmSpecificStats] = globalPathPlanningMPSORL( ...
                startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, ...
                params.popSize, params.maxIterations, params.w, params.c1, params.c2, ...
                params.alpha, params.gamma, params.epsilon, params.learningPeriod, params.pop1Ratio);
            algorithmSpecificStats.convergenceHistory = convergence;

        case 'FIPS'
            % FIPS: Fully Informed Particle Swarm (Mendes et al., 2004)
            [globalPath, convergence, algorithmSpecificStats] = globalPathPlanningFIPS(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.w, params.c1, params.c2);
            algorithmSpecificStats.convergenceHistory = convergence;

        case 'CLPSO'
            % CLPSO: Comprehensive Learning PSO (Liang et al., IEEE TEVC 2006)
            [globalPath, convergence, algorithmSpecificStats] = globalPathPlanningCLPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.w, params.c);
            algorithmSpecificStats.convergenceHistory = convergence;

        case 'HPSO_TVAC'
            % HPSO-TVAC: Heterogeneous PSO with Time-Varying Acceleration Coefficients (Nickabadi et al., 2011)
            [globalPath, convergence, algorithmSpecificStats] = globalPathPlanningHPSOTVAC(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.w, params.c1, params.c2);
            algorithmSpecificStats.convergenceHistory = convergence;

        case 'DMS_PSO'
            % DMS-PSO: Dynamic Multi-Swarm PSO (Liang & Suganthan, 2005)
            [globalPath, convergence, algorithmSpecificStats] = globalPathPlanningDMSPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.w, params.c1, params.c2);
            algorithmSpecificStats.convergenceHistory = convergence;

        case 'LIPS'
            % LIPS: Learning PSO with Heterogeneous Social Interactions (Zhao et al., IEEE TEVC 2021)
            [globalPath, convergence, algorithmSpecificStats] = globalPathPlanningLIPS(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.w, params.c);
            algorithmSpecificStats.convergenceHistory = convergence;

        case 'DQN_PSO'
            paramMode = '5subgroup';
            if isfield(params, 'paramMode')
                paramMode = params.paramMode;
            end

            pretrainedModel = [];
            if isfield(params, 'pretrainedModelPath')
                pretrainedModel = params.pretrainedModelPath;
            end

            [globalPath, convergence, algorithmSpecificStats] = globalPathPlanningDQNPSO( ...
                startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, ...
                params.popSize, params.maxIterations, params.w, params.c1, params.c2, paramMode, pretrainedModel);
            algorithmSpecificStats.convergenceHistory = convergence;

        case 'DQN_PSO_Train'
            % Offline training for DQN-PSO parameter adaptation
            trainingStats = trainDQNPSO( ...
                numEpisodes=params.trainingEpisodes, ...
                savePath=string(params.savePath), ...
                paramMode=string(params.paramMode), ...
                popSize=params.popSize, ...
                maxIterations=params.maxIterations, ...
                w=params.w, ...
                c1=params.c1, ...
                c2=params.c2, ...
                mapSize=mapSize);
            pretrainedModel = trainingStats.finalModel;

            [globalPath, convergence, evalStats] = globalPathPlanningDQNPSO( ...
                startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, ...
                params.popSize, params.maxIterations, params.w, params.c1, params.c2, params.paramMode, pretrainedModel);

            algorithmSpecificStats = evalStats;
            algorithmSpecificStats.convergenceHistory = convergence;
            algorithmSpecificStats.trainingStats = trainingStats;
            algorithmSpecificStats.modelSavePath = params.savePath;

        case 'PPO_PSO'
            config = PPOPSO_Config('online');
            config.popSize = params.popSize;
            config.maxIterations = params.maxIterations;
            config.mapSize = mapSize;

            [globalPath, bestFitness, fitnessHistory, agent, stateTracker, paramHistory, learningStats] = ...
                globalPathPlanningPPOPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config);

            [bestFitness, fitnessComponents] = evaluatePathFromWaypoints(globalPath, startPoint, goalPoint, ...
                dangerZones, terrainGrid, terrainX, terrainY, bestFitness);

            algorithmSpecificStats.convergenceHistory = fitnessHistory;
            algorithmSpecificStats.fitnessComponents = fitnessComponents;
            algorithmSpecificStats.finalFitness = bestFitness;
            algorithmSpecificStats.actualBestFitness = bestFitness;
            algorithmSpecificStats.parameterHistory = paramHistory;
            algorithmSpecificStats.rewardHistory = learningStats.rewardHistory;
            algorithmSpecificStats.criticLossHistory = learningStats.criticLossHistory;
            algorithmSpecificStats.agent = agent;
            algorithmSpecificStats.stateTracker = stateTracker;

        otherwise
            error('Unknown algorithm: %s', algorithmName);
    end
end

function [bestFitness, fitnessComponents] = evaluatePathFromWaypoints(globalPath, startPoint, goalPoint, ...
    dangerZones, terrainGrid, terrainX, terrainY, fallbackFitness)
    fitnessComponents = struct();
    bestFitness = fallbackFitness;

    if isempty(globalPath) || size(globalPath, 1) < 2
        return;
    end

    intermediateWaypoints = globalPath(2:end-1, :);
    numWaypoints = size(intermediateWaypoints, 1);
    position = reshape(intermediateWaypoints', 1, []);
    [evaluatedFitness, fitnessComponents] = evaluatePathFitness(position, startPoint, goalPoint, ...
        dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

    if isfinite(evaluatedFitness)
        bestFitness = evaluatedFitness;
    end
end
