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
            
        case 'RLAMPGPSO'
            % Policy Gradient variant of RLAMPSO (located in algorithms/rlampso/)
            [globalPath, convergence, paramHist, algorithmSpecificStats] = globalPathPlanningRLAMPGPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, params.popSize, params.maxIterations, params.initialW, params.initialC1, params.initialC2);
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

        case 'APEXPSO'
            % APEX-PSO: Advanced Parameter Exploration CrossQ-SAC for PSO
            % State-of-the-art RL algorithm combining SAC, CrossQ, Transformer attention
            config = APEXPSO_Config();  % DEFAULT MODE: 150 episodes, 600 iterations
            config.mapSize = mapSize;   % Enforce map size from comparison script
            [globalPath, bestFitness, fitnessHistory, agent, stateEncoder, paramHistory] = globalPathPlanningAPEXPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config);
            
            % Calculate fitness components for metrics
            intermediateWaypoints = globalPath(2:end-1, :);
            position = reshape(intermediateWaypoints', 1, []);
            numWaypoints = size(intermediateWaypoints, 1);
            [~, fitnessComponents] = evaluatePathFitness(position, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);
            
            algorithmSpecificStats.convergenceHistory = fitnessHistory;
            algorithmSpecificStats.fitnessComponents = fitnessComponents;
            algorithmSpecificStats.finalFitness = bestFitness;
            algorithmSpecificStats.actualBestFitness = bestFitness;  % For metrics compatibility
            algorithmSpecificStats.parameterHistory = paramHistory;
            algorithmSpecificStats.agent = agent;
            algorithmSpecificStats.stateEncoder = stateEncoder;

        case 'APEXPSO_Online'
            % APEX-PSO Online Learning: Learns during actual PSO run (no pretraining)
            % Single episode with N iterations - learns while optimizing
            config = APEXPSO_Config('online');  % ONLINE MODE: 1 episode
            config.mapSize = mapSize;           % Enforce map size from comparison script

            % Override maxIterations from params (for testing with fewer iterations)
            if isfield(params, 'maxIterations')
                config.maxIterations = params.maxIterations;
                config.warmupPeriod = min(50, floor(config.maxIterations * 0.1));
            end

            if isfield(params, 'paramMode')
                config.paramMode = params.paramMode;
            else
                config.paramMode = 'per-particle';
            end

            % Update config based on paramMode (must update both actionSize AND usePerParticleActions)
            switch config.paramMode
                case 'global'
                    config.usePerParticleActions = false;
                    config.actionSize = 3;  % w, c1, c2 for all particles
                    config.targetEntropy = -3;  % Update target entropy for new action size
                case '5subgroup'
                    config.usePerParticleActions = false;
                    config.actionSize = 15;  % 5 groups × 3 params
                    config.targetEntropy = -15;
                case 'per-particle'
                    config.usePerParticleActions = true;
                    config.actionSize = 120;  % 40 particles × 3 params
                    config.targetEntropy = -120;
                otherwise
                    error('Unknown paramMode: %s', config.paramMode);
            end

            [globalPath, bestFitness, fitnessHistory, agent, stateEncoder, paramHistory] = globalPathPlanningAPEXPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config);
            
            % Calculate fitness components for metrics
            intermediateWaypoints = globalPath(2:end-1, :);
            position = reshape(intermediateWaypoints', 1, []);
            numWaypoints = size(intermediateWaypoints, 1);
            [~, fitnessComponents] = evaluatePathFitness(position, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

            algorithmSpecificStats.convergenceHistory = fitnessHistory;
            algorithmSpecificStats.fitnessComponents = fitnessComponents;
            algorithmSpecificStats.finalFitness = bestFitness;
            algorithmSpecificStats.actualBestFitness = bestFitness;  % For metrics compatibility
            algorithmSpecificStats.parameterHistory = paramHistory;
            algorithmSpecificStats.agent = agent;
            algorithmSpecificStats.stateEncoder = stateEncoder;

        case 'APEXPSO_Pretrained'
            config = APEXPSO_Config('pretrained');
            config.mapSize = mapSize;  % Enforce map size from comparison script
            config.pretrainedModelPath = params.pretrainedModelPath;

            if isfield(params, 'paramMode')
                config.paramMode = params.paramMode;
            else
                config.paramMode = 'per-particle';
            end

            % Update config based on paramMode (CRITICAL for loading correct model)
            switch config.paramMode
                case 'global'
                    config.usePerParticleActions = false;
                    config.actionSize = 3;
                case '5subgroup'
                    config.usePerParticleActions = false;
                    config.actionSize = 15;
                case 'per-particle'
                    config.usePerParticleActions = true;
                    config.actionSize = config.popSize * 3;
            end

            [globalPath, bestFitness, fitnessHistory, agent, stateEncoder, paramHistory] = globalPathPlanningAPEXPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config);
            
            % Calculate fitness components for metrics
            intermediateWaypoints = globalPath(2:end-1, :);
            position = reshape(intermediateWaypoints', 1, []);
            numWaypoints = size(intermediateWaypoints, 1);
            [~, fitnessComponents] = evaluatePathFitness(position, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

            algorithmSpecificStats.convergenceHistory = fitnessHistory;
            algorithmSpecificStats.fitnessComponents = fitnessComponents;
            algorithmSpecificStats.finalFitness = bestFitness;
            algorithmSpecificStats.actualBestFitness = bestFitness;
            algorithmSpecificStats.parameterHistory = paramHistory;
            algorithmSpecificStats.agent = agent;
            algorithmSpecificStats.stateEncoder = stateEncoder;

        case 'APEXPSO_Train'
            % APEX-PSO Training: Train model from scratch
            config = APEXPSO_Config('default');
            config.mapSize = mapSize;  % Enforce map size from comparison script
            config.numEpisodes = params.trainingEpisodes;
            config.savePath = params.savePath;

            % Set paramMode if provided
            if isfield(params, 'paramMode')
                config.paramMode = params.paramMode;
            else
                config.paramMode = 'per-particle';
            end

            % Update config based on paramMode
            switch config.paramMode
                case 'global'
                    config.usePerParticleActions = false;
                    config.actionSize = 3;
                case '5subgroup'
                    config.usePerParticleActions = false;
                    config.actionSize = 15;
                case 'per-particle'
                    config.usePerParticleActions = true;
                    config.actionSize = config.popSize * 3;
            end

            [globalPath, bestFitness, fitnessHistory, agent, stateEncoder, paramHistory] = globalPathPlanningAPEXPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config);
            
            % Calculate fitness components for metrics
            intermediateWaypoints = globalPath(2:end-1, :);
            position = reshape(intermediateWaypoints', 1, []);
            numWaypoints = size(intermediateWaypoints, 1);
            [~, fitnessComponents] = evaluatePathFitness(position, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

            algorithmSpecificStats.convergenceHistory = fitnessHistory;
            algorithmSpecificStats.fitnessComponents = fitnessComponents;
            algorithmSpecificStats.finalFitness = bestFitness;
            algorithmSpecificStats.actualBestFitness = bestFitness;
            algorithmSpecificStats.parameterHistory = paramHistory;
            algorithmSpecificStats.agent = agent;
            algorithmSpecificStats.stateEncoder = stateEncoder;

        case 'RLAMPSO_Train'
            % RLAMPSO Training: Train DDPG/TD3 agent offline, then evaluate in current scenario
            trainingConfig = RLAMPSO_Config(params.configMode);
            trainingConfig.paramMode = params.paramMode;
            trainingConfig.numEpisodes = params.trainingEpisodes;
            trainingConfig.popSize = params.popSize;
            trainingConfig.maxIterations = params.maxIterations;
            trainingConfig.savePath = params.savePath;
            trainingConfig.mapSize = mapSize;

            [trainedAgent, trainingStats] = trainRLAMPSO_Parallel(trainingConfig, params.savePath);

            evalConfig = trainingConfig;
            evalConfig.numEpisodes = 1;

            [globalPath, convergence, paramHistory, evalStats] = globalPathPlanningRLAMPSO( ...
                startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, ...
                params.popSize, params.maxIterations, params.initialW, params.initialC1, params.initialC2, ...
                params.savePath, evalConfig, trainedAgent);

            algorithmSpecificStats = evalStats;
            algorithmSpecificStats.convergenceHistory = convergence;
            algorithmSpecificStats.parameterHistory = paramHistory;
            algorithmSpecificStats.trainingStats = trainingStats;
            algorithmSpecificStats.modelSavePath = params.savePath;
            algorithmSpecificStats.trainedAgent = trainedAgent;

        case 'APEXPSO_AblationTrain'
            % APEX-PSO Ablation Training: Train with specific features disabled
            config = APEXPSO_Config(params.ablationMode);
            config.numEpisodes = params.trainingEpisodes;
            config.savePath = params.savePath;
            [globalPath, bestFitness, fitnessHistory, agent, stateEncoder, paramHistory] = globalPathPlanningAPEXPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config);
            
            % Calculate fitness components for metrics
            intermediateWaypoints = globalPath(2:end-1, :);
            position = reshape(intermediateWaypoints', 1, []);
            numWaypoints = size(intermediateWaypoints, 1);
            [~, fitnessComponents] = evaluatePathFitness(position, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

            algorithmSpecificStats.convergenceHistory = fitnessHistory;
            algorithmSpecificStats.fitnessComponents = fitnessComponents;
            algorithmSpecificStats.finalFitness = bestFitness;
            algorithmSpecificStats.actualBestFitness = bestFitness;
            algorithmSpecificStats.parameterHistory = paramHistory;
            algorithmSpecificStats.agent = agent;
            algorithmSpecificStats.stateEncoder = stateEncoder;

        case 'APEXPSO_AblationRun'
            % APEX-PSO Ablation Run: Test with pretrained ablated model
            config = APEXPSO_Config('pretrained');  % FIXED: Use 'pretrained' mode first
            config.pretrainedModelPath = params.pretrainedModelPath;

            % Apply ablation modifications
            switch params.ablationMode
                case 'ablation_no_attention'
                    config.useTransformerState = false;
                    config.stateSize = 15;
                case 'ablation_simple_reward'
                    config.useMultiObjectiveReward = false;
                    config.useCuriosityBonus = false;
            end

            [globalPath, bestFitness, fitnessHistory, agent, stateEncoder, paramHistory] = globalPathPlanningAPEXPSO(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, config);
            
            % Calculate fitness components for metrics
            intermediateWaypoints = globalPath(2:end-1, :);
            position = reshape(intermediateWaypoints', 1, []);
            numWaypoints = size(intermediateWaypoints, 1);
            [~, fitnessComponents] = evaluatePathFitness(position, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, numWaypoints);

            algorithmSpecificStats.convergenceHistory = fitnessHistory;
            algorithmSpecificStats.fitnessComponents = fitnessComponents;
            algorithmSpecificStats.finalFitness = bestFitness;
            algorithmSpecificStats.actualBestFitness = bestFitness;
            algorithmSpecificStats.parameterHistory = paramHistory;
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

        otherwise
            error('Unknown algorithm: %s', algorithmName);
    end
end

%% 1.. RLAM-PSO (DDPG-based)

