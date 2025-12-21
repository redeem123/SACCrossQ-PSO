function checkTrainingLog()
    % CHECKTRAININGLOG - Inspects APEXPSO training logs for data integrity and summary stats.
    % 
    % Usage:
    %   checkTrainingLog()
    
    clc;
    modelDir = fullfile('models', 'APEXPSO');
    
    % Find all log files
    files = dir(fullfile(modelDir, '*_log.mat'));
    
    if isempty(files)
        fprintf('No log files found in %s\n', modelDir);
        return;
    end
    
    fprintf('Found %d log files in %s:\n\n', length(files), modelDir);
    
    for i = 1:length(files)
        filename = files(i).name;
        filePath = fullfile(modelDir, filename);
        
        try
            data = load(filePath);
            if ~isfield(data, 'logData')
                fprintf('[!] %s: Missing "logData" structure.\n', filename);
                continue;
            end
            
            log = data.logData;
            
            % Extract key metrics
            episodes = length(log.episodeRewards);
            meanReward = mean(log.episodeRewards);
            meanFitness = mean(log.episodeFitness);
            bestFitness = min(log.episodeFitness);
            
            % Time calculation
            if isfield(log, 'startTime') && isfield(log, 'endTime')
                try
                    d = log.endTime - log.startTime; % duration object
                    [h,m,s] = hms(d);
                    timeStr = sprintf('%02d:%02d:%02.0f', h, m, s);
                catch
                    timeStr = 'N/A';
                end
            else
                timeStr = sprintf('%.1f s (est)', sum(log.episodeDuration));
            end
            
            % Check for missing/zero fields
            hasLoss = isfield(log, 'critic1Loss') && any(log.critic1Loss ~= 0);
            hasAlpha = isfield(log, 'alphaValue') && any(log.alphaValue ~= 0);
            
            fprintf('===== %s =====\n', filename);
            fprintf('  Model Name:   %s\n', log.modelName);
            fprintf('  Episodes:     %d\n', episodes);
            fprintf('  Duration:     %s\n', timeStr);
            fprintf('  Fitness:      Mean=%.2f, Best=%.2f\n', meanFitness, bestFitness);
            fprintf('  Reward:       Mean=%.2f\n', meanReward);
            fprintf('  Data Checks:  Loss=%s, Alpha=%s\n', ...
                bool2str(hasLoss), bool2str(hasAlpha));
            fprintf('\n');
            
        catch ME
            fprintf('[!] Error reading %s: %s\n', filename, ME.message);
        end
    end
end

function s = bool2str(b)
    if b
        s = '✓ OK';
    else
        s = '✗ MISSING/ZERO';
    end
end
