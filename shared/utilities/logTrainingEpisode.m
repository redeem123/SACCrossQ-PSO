function logTrainingEpisode(logFilePath, episodeData)
    % Log training episode data to CSV file
    % Appends one row of data to the training log
    %
    % Inputs:
    %   logFilePath: Path to the CSV log file
    %   episodeData: struct containing episode metrics
    %
    % Required fields in episodeData:
    %   - episode: Episode number
    %   - totalReward: Total reward for episode
    %   - avgReward: Average reward per iteration
    %   - bestFitness: Best fitness found in episode
    %   - bufferSize: Current replay buffer size
    %   - explorationNoise: Current exploration noise level
    %   - episodeTime: Time taken for this episode (seconds)
    %   - cumulativeTime: Total elapsed time (seconds)
    %
    % Optional fields (algorithm-specific):
    %   RLNNPSO:
    %     - phase: 'PureRL' or 'Adaptive'
    %     - avgGamma: Average gamma value used
    %   RLAMPSO:
    %     - avgW: Average inertia weight
    %     - avgC1: Average cognitive coefficient
    %     - avgC2: Average social coefficient

    % Check if log file exists
    if ~exist(logFilePath, 'file')
        error('Log file does not exist: %s. Call initializeTrainingLog first.', logFilePath);
    end

    % Open file for appending
    fid = fopen(logFilePath, 'a');
    if fid == -1
        error('Cannot open log file for writing: %s', logFilePath);
    end

    % Determine algorithm type from fields
    isRLNNPSO = isfield(episodeData, 'phase');
    isRLAMPSO = isfield(episodeData, 'avgW');

    % Format and write data row
    try
        if isRLNNPSO
            % RLNNPSO format
            avgGamma = 0.0;
            if isfield(episodeData, 'avgGamma')
                avgGamma = episodeData.avgGamma;
            end

            fprintf(fid, '%d,%.6f,%.6f,%.6f,%d,%.6f,%s,%.6f,%.3f,%.3f\n', ...
                episodeData.episode, ...
                episodeData.totalReward, ...
                episodeData.avgReward, ...
                episodeData.bestFitness, ...
                episodeData.bufferSize, ...
                episodeData.explorationNoise, ...
                episodeData.phase, ...
                avgGamma, ...
                episodeData.episodeTime, ...
                episodeData.cumulativeTime);

        elseif isRLAMPSO
            % RLAMPSO format
            fprintf(fid, '%d,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f\n', ...
                episodeData.episode, ...
                episodeData.totalReward, ...
                episodeData.avgReward, ...
                episodeData.bestFitness, ...
                episodeData.bufferSize, ...
                episodeData.explorationNoise, ...
                episodeData.avgW, ...
                episodeData.avgC1, ...
                episodeData.avgC2, ...
                episodeData.episodeTime, ...
                episodeData.cumulativeTime);
        else
            error('Cannot determine algorithm type from episodeData fields');
        end
    catch ME
        fclose(fid);
        rethrow(ME);
    end

    fclose(fid);
end
