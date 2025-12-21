function generateReports(allRunResults, algorithms, scenarioIdx)
    % Generate reports with T-test appropriate filenames
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    
    resultsDir = getResultsDir();

    % CSV with all runs data - Updated filename
    csvFilename = fullfile(resultsDir, sprintf('TTest_AllRuns_Scenario%d_%s.csv', ...
                         scenarioIdx, timestamp));
    writeAllRunsCSV(allRunResults, algorithms, csvFilename);

    % Summary statistics CSV - Updated filename
    summaryCSVFilename = fullfile(resultsDir, sprintf('TTest_AlgorithmSummary_Scenario%d_%s.csv', ...
                                scenarioIdx, timestamp));
    writeSummaryStatisticsCSV(allRunResults, algorithms, summaryCSVFilename);
    
    fprintf('Reports generated:\n');
    fprintf('  All runs data: %s\n', csvFilename);
    fprintf('  Algorithm summary: %s\n', summaryCSVFilename);
end

