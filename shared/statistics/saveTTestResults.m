function saveTTestResults(pathLengths, algorithmNames, pValueMatrix, tStatMatrix, ...
                        scenarioIdx)
    % Save T-test results to files (.mat and .csv)
    % MERGED: Combines writeTTestPairwiseCSV + writeTTestSummaryCSV for cleaner code

    % Calculate means for saving
    % means = mean(pathLengths, 1);
    % stds = std(pathLengths, 0, 1);

    % resultsDir = getResultsDir();

    % % Save .mat file
    % matFilename = fullfile(resultsDir, sprintf('TTest_Results_Scenario%d.mat', scenarioIdx));
    % save(matFilename, 'pathLengths', 'algorithmNames', 'pValueMatrix', 'tStatMatrix', ...
    %      'means', 'stds', 'scenarioIdx');

    % % === PAIRWISE CSV (inline) ===
    % csvFilename = fullfile(resultsDir, sprintf('TTest_Pairwise_Scenario%d.csv', scenarioIdx));
    % fid = fopen(csvFilename, 'w');
    % if fid ~= -1
    %     numAlgorithms = length(algorithmNames);

    %     % Header
    %     fprintf(fid, 'Algorithm1,Algorithm2,Mean1,Mean2,MeanDifference,t_statistic,p_value,Significant,Effect_Size\n');

    %     % Data for each pairwise comparison
    %     for i = 1:numAlgorithms
    %         for j = i+1:numAlgorithms
    %             meanDiff = means(i) - means(j);
    %             tStat = tStatMatrix(i, j);
    %             pValue = pValueMatrix(i, j);

    %             % Determine significance
    %             if pValue < 0.001
    %                 significance = 'p<0.001';
    %             elseif pValue < 0.01
    %                 significance = 'p<0.01';
    %             elseif pValue < 0.05
    %                 significance = 'p<0.05';
    %             else
    %                 significance = 'not_significant';
    %             end

    %             % Calculate effect size (simplified Cohen's d approximation)
    %             effectSize = abs(meanDiff) / sqrt((std(means))^2);

    %             fprintf(fid, '%s,%s,%.4f,%.4f,%.4f,%.4f,%.6f,%s,%.3f\n', ...
    %                 algorithmNames{i}, algorithmNames{j}, means(i), means(j), ...
    %                 meanDiff, tStat, pValue, significance, effectSize);
    %         end
    %     end
    %     fclose(fid);
    % else
    %     warning('Could not open pairwise CSV file for writing: %s', csvFilename);
    % end

    % % === SUMMARY CSV (inline) ===
    % summaryFilename = fullfile(resultsDir, sprintf('TTest_Summary_Scenario%d.csv', scenarioIdx));
    % fid = fopen(summaryFilename, 'w');
    % if fid ~= -1
    %     % Header
    %     fprintf(fid, 'Algorithm,Mean_GlobalFitness,Std_GlobalFitness,Rank\n');

    %     % Sort algorithms by mean fitness (lower is better)
    %     [sortedMeans, rankIdx] = sort(means);

    %     % Data
    %     for i = 1:length(algorithmNames)
    %         algIdx = rankIdx(i);
    %         fprintf(fid, '%s,%.4f,%.4f,%d\n', ...
    %             algorithmNames{algIdx}, sortedMeans(i), stds(algIdx), i);
    %     end
    %     fclose(fid);
    % else
    %     warning('Could not open summary CSV file for writing: %s', summaryFilename);
    % end

    % fprintf('\nT-test results saved to:\n');
    % fprintf('  MAT file: %s\n', matFilename);
    % fprintf('  Pairwise CSV: %s\n', csvFilename);
    % fprintf('  Summary CSV: %s\n', summaryFilename);
end
