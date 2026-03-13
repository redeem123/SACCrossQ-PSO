function rebuild_result_csvs()
%REBUILD_RESULT_CSVS Rebuild summary/pairwise CSV files from saved MAT stats.

    targetDirs = {
        '/Users/hust-hwashin621m/Desktop/vietanhpaper-2/results/rlbased'
        '/Users/hust-hwashin621m/Desktop/vietanhpaper-2/results/others3chrismast'
    };

    for dirIdx = 1:numel(targetDirs)
        rebuildDirectory(targetDirs{dirIdx});
    end
end

function rebuildDirectory(resultDir)
    if exist(resultDir, 'dir') ~= 7
        error('Result directory not found: %s', resultDir);
    end

    matFiles = dir(fullfile(resultDir, 'TTest_Results_Scenario*.mat'));
    if isempty(matFiles)
        fprintf('No TTest MAT files found in %s\n', resultDir);
        return;
    end

    [~, sortIdx] = sort(extractScenarioIds({matFiles.name}));
    matFiles = matFiles(sortIdx);

    for fileIdx = 1:numel(matFiles)
        matPath = fullfile(resultDir, matFiles(fileIdx).name);
        data = load(matPath, 'summaryResults', 'pairwiseResults');

        if ~isfield(data, 'summaryResults') || ~isfield(data, 'pairwiseResults')
            error('Missing summaryResults/pairwiseResults in %s', matPath);
        end

        scenarioId = extractScenarioIds({matFiles(fileIdx).name});
        scenarioId = scenarioId(1);

        summaryTable = struct2table(data.summaryResults, 'AsArray', true);
        pairwiseTable = struct2table(data.pairwiseResults, 'AsArray', true);

        writetable(summaryTable, fullfile(resultDir, sprintf('TTest_Summary_Scenario%d.csv', scenarioId)));
        writetable(pairwiseTable, fullfile(resultDir, sprintf('TTest_Pairwise_Scenario%d.csv', scenarioId)));

        fprintf('Rebuilt Scenario %d CSVs in %s\n', scenarioId, resultDir);
    end
end

function ids = extractScenarioIds(fileNames)
    ids = zeros(size(fileNames));
    for idx = 1:numel(fileNames)
        tokens = regexp(fileNames{idx}, 'Scenario(\d+)', 'tokens', 'once');
        if isempty(tokens)
            error('Unable to parse scenario id from %s', fileNames{idx});
        end
        ids(idx) = str2double(tokens{1});
    end
end
