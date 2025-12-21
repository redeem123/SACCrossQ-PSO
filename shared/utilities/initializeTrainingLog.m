function logFilePath = initializeTrainingLog(algorithmName, trainingConfig)
    % Initialize CSV log file for training monitoring
    % Creates trainingresult/ directory and CSV file with headers
    %
    % Inputs:
    %   algorithmName: 'RLNNPSO' or 'RLAMPSO'
    %   trainingConfig: struct with training parameters (optional, for metadata)
    %
    % Output:
    %   logFilePath: Full path to the created CSV file
    %
    % CSV Columns:
    %   Common: Episode, TotalReward, AvgReward, BestFitness, BufferSize,
    %           ExplorationNoise, EpisodeTime, CumulativeTime
    %   RLNNPSO-specific: Phase, AvgGamma
    %   RLAMPSO-specific: AvgW, AvgC1, AvgC2

    % Create trainingresult directory
    resultDir = 'trainingresult';
    if ~exist(resultDir, 'dir')
        mkdir(resultDir);
    end

    % Generate timestamped filename
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    filename = sprintf('%s_training_%s.csv', lower(algorithmName), timestamp);
    logFilePath = fullfile(resultDir, filename);

    % Prepare header based on algorithm
    if strcmpi(algorithmName, 'RLNNPSO')
        header = ['Episode,TotalReward,AvgReward,BestFitness,BufferSize,' ...
                  'ExplorationNoise,Phase,AvgGamma,EpisodeTime_sec,CumulativeTime_sec\n'];
    elseif strcmpi(algorithmName, 'RLAMPSO')
        header = ['Episode,TotalReward,AvgReward,BestFitness,BufferSize,' ...
                  'ExplorationNoise,AvgW,AvgC1,AvgC2,EpisodeTime_sec,CumulativeTime_sec\n'];
    else
        error('Unknown algorithm name: %s. Use ''RLNNPSO'' or ''RLAMPSO''', algorithmName);
    end

    % Write header to file
    fid = fopen(logFilePath, 'w');
    if fid == -1
        error('Cannot create log file: %s', logFilePath);
    end
    fprintf(fid, header);

    % Write metadata comments (optional)
    if nargin >= 2 && ~isempty(trainingConfig)
        fprintf(fid, '# Algorithm: %s\n', algorithmName);
        fprintf(fid, '# Timestamp: %s\n', datestr(now));
        if isfield(trainingConfig, 'numEpisodes')
            fprintf(fid, '# NumEpisodes: %d\n', trainingConfig.numEpisodes);
        end
        if isfield(trainingConfig, 'popSize')
            fprintf(fid, '# PopSize: %d\n', trainingConfig.popSize);
        end
        if isfield(trainingConfig, 'maxIterations')
            fprintf(fid, '# MaxIterations: %d\n', trainingConfig.maxIterations);
        end
        if isfield(trainingConfig, 'bufferSize')
            fprintf(fid, '# BufferSize: %d\n', trainingConfig.bufferSize);
        end
        if isfield(trainingConfig, 'batchSize')
            fprintf(fid, '# BatchSize: %d\n', trainingConfig.batchSize);
        end
        fprintf(fid, '\n');
    end

    fclose(fid);

    fprintf('Training log initialized: %s\n', logFilePath);
end
