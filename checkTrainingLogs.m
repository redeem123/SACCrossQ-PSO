%% checkTrainingLogs.m
% Verify training times from log files to check for anomalies
%
% This script reads the _log.mat files and extracts actual timing data
% to verify the training efficiency table values.

function checkTrainingLogs()
    clc;
fprintf('╔══════════════════════════════════════════════════════════╗\n');
fprintf('║          Training Log Analysis                          ║\n');
fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

% Log files to check logs = {'apexpso_perparticle_log.mat', 'Full (Baseline)';
'apexpso_abl_noatt_log.mat', 'No Attention';
'apexpso_abl_simplerwd_log.mat', 'Simple Reward';
'apexpso_abl_nocrossq_log.mat', 'No CrossQ'
}
;

modelDir = fullfile('models', 'APEXPSO');

fprintf('%-20s | %8s | %6s | %10s | %10s | %10s\n', ... 'Variant', 'Time(h)',
        'Eps', 'Mean(s/ep)', 'Min(s/ep)', 'Max(s/ep)');
fprintf('%s\n', repmat('-', 1, 80));

results = struct();

    for
      i = 1 : size(logs, 1) logFile = logs{i, 1};
    variantName = logs{i, 2};
    logPath = fullfile(modelDir, logFile);

    if exist (logPath, 'file')
      d = load(logPath);

    % Extract timing data episodeDurations = d.logData.episodeDuration;
    numEpisodes = length(episodeDurations);
    totalTimeHours = sum(episodeDurations) / 3600;
    meanEpTime = mean(episodeDurations);
    minEpTime = min(episodeDurations);
    maxEpTime = max(episodeDurations);

    % Store results results(i).name = variantName;
    results(i).totalTime = totalTimeHours;
    results(i).numEpisodes = numEpisodes;
    results(i).meanEpTime = meanEpTime;
    results(i).episodeDurations = episodeDurations;
    results(i).episodeRewards = d.logData.episodeRewards;
    results(i).episodeFitness = d.logData.episodeFitness;

    fprintf('%-20s | %8.2f | %6d | %10.1f | %10.1f | %10.1f\n', ... variantName,
            totalTimeHours, numEpisodes, meanEpTime, minEpTime, maxEpTime);

            % Check for anomalies
            if maxEpTime > 3 * meanEpTime
                fprintf('  ⚠️  WARNING: Max episode time is %.1fx mean!\n', maxEpTime/meanEpTime);
            end else fprintf('%-20s | %s\n', variantName, 'FILE NOT FOUND');
            end end

                fprintf('\n');

            % %
                Detailed Analysis fprintf(
                    '\n--- Detailed Episode-by-Episode Analysis ---\n\n');

    for
      i = 1 : length(results) if isfield (results(i), 'episodeDurations') &&
              ~isempty(results(i).episodeDurations) durations =
              results(i).episodeDurations;

    % Find outlier episodes(> 2 std from mean) meanDur = mean(durations);
    stdDur = std(durations);
    outliers = find(durations > meanDur + 2 * stdDur);

    if
      ~isempty(outliers) fprintf('%s: Found %d slow episodes:\n',
                                 results(i).name, length(outliers));
                for
                  j = 1 : min(5, length(outliers)) ep = outliers(j);
                fprintf('  Episode %d: %.1fs (%.1fx mean)\n', ... ep,
                        durations(ep), durations(ep) / meanDur);
                end if length (outliers) >
                    5 fprintf('  ... and %d more\n', length(outliers) - 5);
                end else fprintf(
                    '%s: No outlier episodes detected (stable training)\n',
                    results(i).name);
                end end end

                    % %
                    Summary Statistics fprintf('\n--- Final Summary ---\n\n');
                fprintf('Total Samples per variant:\n');
    for
      i = 1 : length(results) if isfield (results(i), 'numEpisodes') &&
              results(i).numEpisodes > 0 totalSamples =
              results(i).numEpisodes * 600;
    % 600 iterations per episode fprintf('  %s: %d samples\n', results(i).name,
                                         totalSamples);
    end end

        fprintf('\n');
    fprintf('Training time ratios (relative to Full Baseline):\n');
    if
      ~isempty(results) &&
          isfield(results(1), 'totalTime') baselineTime = results(1).totalTime;
        for
          i = 1 : length(results) if isfield (results(i), 'totalTime') &&
                  results(i).totalTime > 0 ratio =
                  results(i).totalTime / baselineTime;
        fprintf('  %s: %.2fx\n', results(i).name, ratio);
        end end end

            fprintf('\n✓ Analysis complete.\n');
        end
