function [terrainGrid, terrainX, terrainY, terrainMeta] = generateFixedTerrain(mapSize, scenarioIdx, terrainFile)
    %#ok<INUSD>
    if nargin < 3
        terrainFile = '';
    end

    terrainMeta = struct();

    resolvedTerrainFile = getTerrainSourcePath(terrainFile);
    if ~isempty(resolvedTerrainFile)
        terrainDef = loadTerrainDefinition(resolvedTerrainFile, struct('mapSize', mapSize));
        terrainGrid = terrainDef.terrainGrid;
        terrainX = terrainDef.terrainX;
        terrainY = terrainDef.terrainY;
        terrainMeta = terrainDef;
        return;
    end

    % Load terrain from ChrismasTerrain2.tif file
    currentDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(fileparts(currentDir));
    terrainFilePath = fullfile(projectRoot, 'data', 'ChrismasTerrain2.tif');
    terrainDef = loadTerrainDefinition(terrainFilePath, struct('mapSize', mapSize));

    terrainGrid = terrainDef.terrainGrid;
    terrainX = terrainDef.terrainX;
    terrainY = terrainDef.terrainY;
    terrainMeta = terrainDef;
end

