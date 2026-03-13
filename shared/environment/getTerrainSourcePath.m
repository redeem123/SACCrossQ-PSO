function terrainFile = getTerrainSourcePath(configuredTerrainFile)
    % Resolve an explicit terrain file path or an environment override.

    terrainFile = '';

    if nargin >= 1 && ~isempty(configuredTerrainFile)
        terrainFile = char(string(configuredTerrainFile));
    end

    envTerrainFile = strtrim(getenv('VIETANH_TERRAIN_FILE'));
    if ~isempty(envTerrainFile)
        terrainFile = envTerrainFile;
    end

    if isempty(terrainFile)
        return;
    end

    terrainFile = char(string(terrainFile));
    if ~isfile(terrainFile)
        error('Terrain file not found: %s', terrainFile);
    end
end
