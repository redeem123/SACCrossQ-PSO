function [algorithmParams, commonParams] = parseAlgorithmParameters(algorithmName, remainingParams)
    % Parse algorithm-specific and common parameters

    switch algorithmName
        case 'RLAMPSO'
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.initialW = remainingParams{3};
            algorithmParams.initialC1 = remainingParams{4};
            algorithmParams.initialC2 = remainingParams{5};

            paramIdx = 6;
            % Parse optional pretrainedNetworkPath
            if length(remainingParams) >= paramIdx
                if ischar(remainingParams{paramIdx}) && ~isempty(remainingParams{paramIdx}) && contains(remainingParams{paramIdx}, {'/', '\'})
                    algorithmParams.pretrainedNetworkPath = remainingParams{paramIdx};
                    paramIdx = paramIdx + 1;
                elseif ischar(remainingParams{paramIdx}) && (isempty(remainingParams{paramIdx}) || strcmp(remainingParams{paramIdx}, ''))
                    % Empty path - skip it
                    algorithmParams.pretrainedNetworkPath = '';
                    paramIdx = paramIdx + 1;
                else
                    % Not a path - must be configMode, don't consume
                    algorithmParams.pretrainedNetworkPath = '';
                end
            else
                algorithmParams.pretrainedNetworkPath = '';
            end

            % Parse optional configMode
            if length(remainingParams) >= paramIdx && ischar(remainingParams{paramIdx})
                configMode = remainingParams{paramIdx};
                if any(strcmp(configMode, {'baseline', 'advanced', 'td3_only', 'transformer_only', 'fast'}))
                    algorithmParams.configMode = configMode;
                    paramIdx = paramIdx + 1;
                else
                    algorithmParams.configMode = 'baseline';
                    % Don't consume parameter if it's not a valid config mode
                end
            else
                algorithmParams.configMode = 'baseline';
            end

            % Parse optional paramMode
            if length(remainingParams) >= paramIdx && ischar(remainingParams{paramIdx})
                paramMode = remainingParams{paramIdx};
                if any(strcmp(paramMode, {'global', '5subgroup', 'per-particle'}))
                    algorithmParams.paramMode = paramMode;
                    paramIdx = paramIdx + 1;
                else
                    % Don't consume parameter if it's not a valid param mode
                end
            end

            if length(remainingParams) >= paramIdx && isstruct(remainingParams{paramIdx})
                algorithmParams.configOverrides = remainingParams{paramIdx};
                paramIdx = paramIdx + 1;
            end

            commonStartIdx = paramIdx;
    
        case 'ActorCriticPSO'
            % ActorCriticPSO: popSize, maxIterations, initialW, initialC1, initialC2, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.initialW = remainingParams{3};
            algorithmParams.initialC1 = remainingParams{4};
            algorithmParams.initialC2 = remainingParams{5};
            commonStartIdx = 6;
        case 'PSO'
            % PSO: popSize, maxIterations, w, c1, c2, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.w = remainingParams{3};
            algorithmParams.c1 = remainingParams{4};
            algorithmParams.c2 = remainingParams{5};
            commonStartIdx = 6;

        case 'SVPSO'
            % SVPSO: popSize, maxIterations, w, c1, c2, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.w = remainingParams{3};
            algorithmParams.c1 = remainingParams{4};
            algorithmParams.c2 = remainingParams{5};
            commonStartIdx = 6;
            
        case 'PSO_TVAC'
            % PSO_TVAC: popSize, maxIterations, initialW, initialC1, initialC2, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.initialW = remainingParams{3};
            algorithmParams.initialC1 = remainingParams{4};
            algorithmParams.initialC2 = remainingParams{5};
            commonStartIdx = 6;

        case 'PSO_LDIW'
            % PSO_LDIW: popSize, maxIterations, initialW, initialC1, initialC2, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.initialW = remainingParams{3};
            algorithmParams.initialC1 = remainingParams{4};
            algorithmParams.initialC2 = remainingParams{5};
            commonStartIdx = 6;

        case 'RLNNPSO'
            % RL-NNPSO (TD3-based velocity guidance): popSize, maxIterations, fixedW, fixedC1, fixedC2, pretrainedNetworkPath (optional), [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.fixedW = remainingParams{3};
            algorithmParams.fixedC1 = remainingParams{4};
            algorithmParams.fixedC2 = remainingParams{5};

            % Check if pre-trained network path is provided (6th parameter)
            if length(remainingParams) >= 6 && ischar(remainingParams{6})
                algorithmParams.pretrainedNetworkPath = remainingParams{6};
                commonStartIdx = 7;
            else
                commonStartIdx = 6;
            end

        case 'SAEPSO'
            % SAEPSO: popSize, maxIterations, c_min, c_max, w_min, w_max, useOtherImprovements, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.c_min = remainingParams{3};
            algorithmParams.c_max = remainingParams{4};
            algorithmParams.w_min = remainingParams{5};
            algorithmParams.w_max = remainingParams{6};

            % Check if useOtherImprovements parameter is provided (7th param)
            if length(remainingParams) >= 7 && islogical(remainingParams{7})
                algorithmParams.useOtherImprovements = remainingParams{7};
                commonStartIdx = 8;
            else
                % Default to true (full SAEPSO) for backward compatibility
                algorithmParams.useOtherImprovements = true;
                commonStartIdx = 7;
            end

        case 'UAPSO'
            % UAPSO: popSize, maxIterations, c_min, c_max, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.c_min = remainingParams{3};
            algorithmParams.c_max = remainingParams{4};
            commonStartIdx = 5;

        case 'PGPSO'
            % PGPSO: popSize, maxIterations, initialW, initialC1, initialC2, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.initialW = remainingParams{3};
            algorithmParams.initialC1 = remainingParams{4};
            algorithmParams.initialC2 = remainingParams{5};
            commonStartIdx = 6;
            
        case 'ANNPSO'
            % ANN-PSO: popSize, maxIterations, w, c1, c2, hiddenNeurons, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.w = remainingParams{3};
            algorithmParams.c1 = remainingParams{4};
            algorithmParams.c2 = remainingParams{5};
            algorithmParams.hiddenNeurons = remainingParams{6};
            commonStartIdx = 7;
            
        case 'IGPSO'
            % IGPSO: popSize, maxIterations, positiveSampleRatio, hiddenNeurons, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.positiveSampleRatio = remainingParams{3};
            algorithmParams.hiddenNeurons = remainingParams{4};
            commonStartIdx = 5;

        case 'AFSACPSO'
            % AFSACPSO: No algorithm-specific params (all in config), [common params]
            % Configuration handled internally by AFSACPSO_Config (AFSACPSO).
            algorithmParams = struct();  % Empty struct, no algorithm-specific params
            commonStartIdx = 1;

        case 'AFSACPSO_Online'
            % AFSACPSO online learning:
            %   [popSize], maxIterations, [paramMode], [configOverrides], [common params]
            paramIdx = 1;
            if length(remainingParams) >= 2 && isnumeric(remainingParams{1}) && isscalar(remainingParams{1}) && ...
                    isnumeric(remainingParams{2}) && isscalar(remainingParams{2})
                algorithmParams.popSize = remainingParams{1};
                algorithmParams.maxIterations = remainingParams{2};
                paramIdx = 3;
            else
                algorithmParams.maxIterations = remainingParams{1};
                paramIdx = 2;
            end

            if length(remainingParams) >= paramIdx && ischar(remainingParams{paramIdx}) && ...
                    any(strcmp(remainingParams{paramIdx}, {'global', '5subgroup', 'per-particle', 'rank-residual', 'attractor-field'}))
                algorithmParams.paramMode = remainingParams{paramIdx};
                paramIdx = paramIdx + 1;
            end
            % If no paramMode specified, let AFSACPSO_Config default stand

            if length(remainingParams) >= paramIdx && isstruct(remainingParams{paramIdx})
                algorithmParams.configOverrides = remainingParams{paramIdx};
                paramIdx = paramIdx + 1;
            end

            commonStartIdx = paramIdx;

        case 'PPO_PSO'
            % PPO-PSO: popSize, maxIterations, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            commonStartIdx = 3;

        % ========================================================================
        % NEW ALGORITHMS FOR TOP-TIER JOURNAL COMPARISON
        % ========================================================================

        case 'FIPS'
            % FIPS: popSize, maxIterations, w, c1, c2, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.w = remainingParams{3};
            algorithmParams.c1 = remainingParams{4};
            algorithmParams.c2 = remainingParams{5};
            commonStartIdx = 6;

        case 'CLPSO'
            % CLPSO: popSize, maxIterations, w, c, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.w = remainingParams{3};
            algorithmParams.c = remainingParams{4};
            commonStartIdx = 5;

        case 'HPSO_TVAC'
            % HPSO_TVAC: popSize, maxIterations, w, c1, c2, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.w = remainingParams{3};
            algorithmParams.c1 = remainingParams{4};
            algorithmParams.c2 = remainingParams{5};
            commonStartIdx = 6;

        case 'DMS_PSO'
            % DMS_PSO: popSize, maxIterations, w, c1, c2, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.w = remainingParams{3};
            algorithmParams.c1 = remainingParams{4};
            algorithmParams.c2 = remainingParams{5};
            commonStartIdx = 6;

        case 'LIPS'
            % LIPS: popSize, maxIterations, w, c, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.w = remainingParams{3};
            algorithmParams.c = remainingParams{4};
            commonStartIdx = 5;

        case 'MPSORL'
            % MPSORL: popSize, maxIterations, wMax, c1Max, c2Max, alpha, gamma, epsilon, learningPeriod, pop1Ratio, [common params]
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.w = remainingParams{3};
            algorithmParams.c1 = remainingParams{4};
            algorithmParams.c2 = remainingParams{5};

            paramIdx = 6;
            algorithmParams.alpha = remainingParams{paramIdx}; paramIdx = paramIdx + 1;
            algorithmParams.gamma = remainingParams{paramIdx}; paramIdx = paramIdx + 1;
            algorithmParams.epsilon = remainingParams{paramIdx}; paramIdx = paramIdx + 1;
            algorithmParams.learningPeriod = remainingParams{paramIdx}; paramIdx = paramIdx + 1;
            if length(remainingParams) >= paramIdx
                algorithmParams.pop1Ratio = remainingParams{paramIdx};
                paramIdx = paramIdx + 1;
            else
                algorithmParams.pop1Ratio = 0.4;
            end
            commonStartIdx = paramIdx;

        case 'DQN_PSO_Train'
            % DQN-PSO Training: episodes, savePath, [paramMode], popSize, maxIterations, w, c1, c2, [common params]
            algorithmParams.trainingEpisodes = remainingParams{1};
            algorithmParams.savePath = remainingParams{2};

            paramIdx = 3;
            algorithmParams.paramMode = '5subgroup';
            if length(remainingParams) >= paramIdx && ischar(remainingParams{paramIdx}) && ...
                    any(strcmp(remainingParams{paramIdx}, {'global', '5subgroup', 'per-particle'}))
                algorithmParams.paramMode = remainingParams{paramIdx};
                paramIdx = paramIdx + 1;
            end

            algorithmParams.popSize = remainingParams{paramIdx}; paramIdx = paramIdx + 1;
            algorithmParams.maxIterations = remainingParams{paramIdx}; paramIdx = paramIdx + 1;
            algorithmParams.w = remainingParams{paramIdx}; paramIdx = paramIdx + 1;
            algorithmParams.c1 = remainingParams{paramIdx}; paramIdx = paramIdx + 1;
            algorithmParams.c2 = remainingParams{paramIdx}; paramIdx = paramIdx + 1;

            commonStartIdx = paramIdx;

        case 'DQN_PSO'
            algorithmParams.popSize = remainingParams{1};
            algorithmParams.maxIterations = remainingParams{2};
            algorithmParams.w = remainingParams{3};
            algorithmParams.c1 = remainingParams{4};
            algorithmParams.c2 = remainingParams{5};

            paramIdx = 6;
            algorithmParams.pretrainedModelPath = '';
            if length(remainingParams) >= paramIdx && ischar(remainingParams{paramIdx})
                candidate = remainingParams{paramIdx};
                if ~isempty(candidate) && (contains(candidate, {'/', '\'}) || endsWith(lower(candidate), '.mat'))
                    algorithmParams.pretrainedModelPath = candidate;
                    paramIdx = paramIdx + 1;
                end
            end

            if length(remainingParams) >= paramIdx && ischar(remainingParams{paramIdx})
                paramMode = remainingParams{paramIdx};
                if any(strcmp(paramMode, {'global', '5subgroup', 'per-particle'}))
                    algorithmParams.paramMode = paramMode;
                    paramIdx = paramIdx + 1;
                end
            end
            commonStartIdx = paramIdx;

        otherwise
            error('Unknown algorithm: %s', algorithmName);
    end
    
    % Extract common parameters
    commonParams.globalPlanInterval = remainingParams{commonStartIdx};
    commonParams.pathDeviationThreshold = remainingParams{commonStartIdx + 1};
    commonParams.obstacleChangeThreshold = remainingParams{commonStartIdx + 2};
    commonParams.timeStep = remainingParams{commonStartIdx + 3};
    commonParams.totalTime = remainingParams{commonStartIdx + 4};
    commonParams.terrain_map = remainingParams{commonStartIdx + 5};
end

