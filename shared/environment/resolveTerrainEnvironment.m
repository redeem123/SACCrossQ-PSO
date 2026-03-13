function environment = resolveTerrainEnvironment(environment)
    % Resolve map bounds and endpoints from the configured terrain source.

    if nargin < 1 || isempty(environment)
        environment = struct();
    end

    terrainFile = '';
    if isfield(environment, 'terrainFile') && ~isempty(environment.terrainFile)
        terrainFile = char(string(environment.terrainFile));
    end

    envTerrainFile = strtrim(getenv('VIETANH_TERRAIN_FILE'));
    if ~isempty(envTerrainFile)
        terrainFile = envTerrainFile;
    elseif isempty(terrainFile)
        currentDir = fileparts(mfilename('fullpath'));
        projectRoot = fileparts(fileparts(currentDir));
        defaultTerrainFile = fullfile(projectRoot, 'data', 'terrainStruct_c_100.mat');
        if isfile(defaultTerrainFile)
            terrainFile = defaultTerrainFile;
        end
    end

    if isempty(terrainFile)
        environment.terrainFile = '';
        return;
    end

    terrainDef = loadTerrainDefinition(terrainFile, environment);
    environment.terrainFile = terrainDef.terrainFile;
    environment.mapSize = terrainDef.mapSize;
    environment.startPoint = terrainDef.startPoint;
    environment.goalPoint = terrainDef.goalPoint;
end
