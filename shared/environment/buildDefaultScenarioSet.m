function scenarios = buildDefaultScenarioSet(dataDir)
    % Build the default 4-scenario suite across both supported terrain maps.

    if nargin < 1 || isempty(dataDir)
        currentDir = fileparts(mfilename('fullpath'));
        projectRoot = fileparts(fileparts(currentDir));
        dataDir = fullfile(projectRoot, 'data');
    end

    mapA = fullfile(dataDir, 'ChrismasTerrain2.tif');
    mapB = fullfile(dataDir, 'terrainStruct_c_100.mat');

    scenarios = [ ...
        makeScenario('ChrismasTerrain2 - Simple', mapA, 0); ...
        makeScenario('ChrismasTerrain2 - Medium', mapA, 5); ...
        makeScenario('ChrismasTerrain2 - Complex', mapA, 10); ...
        makeScenario('terrainStruct_c_100 - Simple', mapB, 0, ...
            struct('goalPoint', [200, 200, 10])) ...
    ];
end

function scenario = makeScenario(label, terrainFile, numDangerZones, overrides)
    scenario = struct( ...
        'label', label, ...
        'terrainFile', terrainFile, ...
        'numDangerZones', numDangerZones, ...
        'startPoint', [], ...
        'goalPoint', []);

    if nargin >= 4 && isstruct(overrides)
        overrideFields = fieldnames(overrides);
        for idx = 1:numel(overrideFields)
            fieldName = overrideFields{idx};
            scenario.(fieldName) = overrides.(fieldName);
        end
    end
end
