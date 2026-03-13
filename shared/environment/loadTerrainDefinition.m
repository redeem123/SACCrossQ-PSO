function terrainDef = loadTerrainDefinition(terrainFile, fallbackEnvironment)
    % Load a terrain source into the planner's common terrain representation.

    if nargin < 2 || isempty(fallbackEnvironment)
        fallbackEnvironment = struct();
    end

    terrainFile = getTerrainSourcePath(terrainFile);
    if isempty(terrainFile)
        error('A terrain file must be provided to loadTerrainDefinition.');
    end

    fallbackMapSize = getFallbackMapSize(fallbackEnvironment);
    fallbackStart = getFallbackPoint(fallbackEnvironment, 'startPoint', [10, 95, 10]);
    fallbackGoal = getFallbackPoint(fallbackEnvironment, 'goalPoint', [97, 2, 10]);

    [~, ~, ext] = fileparts(terrainFile);
    switch lower(ext)
        case {'.tif', '.tiff'}
            terrainDef = loadGeoTiffTerrainDefinition(terrainFile, fallbackMapSize, fallbackStart, fallbackGoal);
        case '.mat'
            terrainDef = loadMatTerrainDefinition(terrainFile, fallbackMapSize, fallbackStart, fallbackGoal);
        otherwise
            error('Unsupported terrain file extension: %s', ext);
    end
end

function terrainDef = loadGeoTiffTerrainDefinition(terrainFile, mapSize, startPoint, goalPoint)
    H = imread(terrainFile);
    H = double(H);
    H(H < 0) = 0;

    xTerrain = linspace(0, mapSize(1), size(H, 2));
    yTerrain = linspace(0, mapSize(2), size(H, 1));
    [XTerrain, YTerrain] = meshgrid(xTerrain, yTerrain);

    xNew = linspace(0, mapSize(1), 100);
    yNew = linspace(0, mapSize(2), 100);
    [terrainX, terrainY] = meshgrid(xNew, yNew);

    terrainGrid = normalizeTerrainHeights(H, mapSize(3));
    terrainGrid = interp2(XTerrain, YTerrain, terrainGrid, terrainX, terrainY, 'linear', 0);
    terrainGrid = max(terrainGrid, 0);

    terrainDef = struct();
    terrainDef.terrainFile = terrainFile;
    terrainDef.terrainGrid = terrainGrid;
    terrainDef.terrainX = terrainX;
    terrainDef.terrainY = terrainY;
    terrainDef.mapSize = mapSize;
    terrainDef.startPoint = sanitizeAnchorPoint(startPoint, terrainGrid, terrainX, terrainY, mapSize);
    terrainDef.goalPoint = sanitizeAnchorPoint(goalPoint, terrainGrid, terrainX, terrainY, mapSize);
end

function terrainDef = loadMatTerrainDefinition(terrainFile, fallbackMapSize, fallbackStart, fallbackGoal)
    matData = load(terrainFile);

    if isfield(matData, 'terrainStruct') && isstruct(matData.terrainStruct)
        terrainStruct = matData.terrainStruct;

        rawGrid = double(getRequiredField(terrainStruct, {'H', 'terrainGrid', 'Z', 'elevation', 'DEM'}));
        rawGrid(rawGrid < 0) = 0;

        rawX = getRequiredField(terrainStruct, {'X', 'terrainX', 'x'});
        rawY = getRequiredField(terrainStruct, {'Y', 'terrainY', 'y'});

        xInfo = buildAxisInfo(rawX, getOptionalScalar(terrainStruct, 'xmin', []), ...
            getOptionalScalar(terrainStruct, 'xmax', []), size(rawGrid, 2));
        yInfo = buildAxisInfo(rawY, getOptionalScalar(terrainStruct, 'ymin', []), ...
            getOptionalScalar(terrainStruct, 'ymax', []), size(rawGrid, 1));

        rawGrid = orientGrid(rawGrid, xInfo.axis, yInfo.axis);

        [terrainX, terrainY] = meshgrid(xInfo.normalizedAxis, yInfo.normalizedAxis);

        zMax = getOptionalScalar(terrainStruct, 'zmax', fallbackMapSize(3));
        if isempty(zMax) || ~isfinite(zMax) || zMax <= 0
            zMax = fallbackMapSize(3);
        end
        mapSize = [xInfo.targetMax, yInfo.targetMax, zMax];

        terrainGrid = normalizeTerrainHeights(rawGrid, mapSize(3));

        rawStart = getOptionalVector(terrainStruct, 'start', fallbackStart);
        rawGoal = getOptionalVector(terrainStruct, 'end', fallbackGoal);

        startPoint = normalizePlanPoint(rawStart, xInfo, yInfo, mapSize(3));
        goalPoint = normalizePlanPoint(rawGoal, xInfo, yInfo, mapSize(3));
    else
        rawGrid = double(getRequiredField(matData, {'terrainGrid', 'Z', 'elevation', 'DEM'}));
        rawGrid(rawGrid < 0) = 0;

        rawX = getRequiredField(matData, {'terrainX', 'X', 'x'});
        rawY = getRequiredField(matData, {'terrainY', 'Y', 'y'});

        xInfo = buildAxisInfo(rawX, [], [], size(rawGrid, 2));
        yInfo = buildAxisInfo(rawY, [], [], size(rawGrid, 1));
        rawGrid = orientGrid(rawGrid, xInfo.axis, yInfo.axis);

        [terrainX, terrainY] = meshgrid(xInfo.normalizedAxis, yInfo.normalizedAxis);
        mapSize = fallbackMapSize;
        terrainGrid = normalizeTerrainHeights(rawGrid, mapSize(3));
        startPoint = fallbackStart;
        goalPoint = fallbackGoal;
    end

    terrainDef = struct();
    terrainDef.terrainFile = terrainFile;
    terrainDef.terrainGrid = terrainGrid;
    terrainDef.terrainX = terrainX;
    terrainDef.terrainY = terrainY;
    terrainDef.mapSize = mapSize;
    terrainDef.startPoint = sanitizeAnchorPoint(startPoint, terrainGrid, terrainX, terrainY, mapSize);
    terrainDef.goalPoint = sanitizeAnchorPoint(goalPoint, terrainGrid, terrainX, terrainY, mapSize);
end

function mapSize = getFallbackMapSize(fallbackEnvironment)
    if isfield(fallbackEnvironment, 'mapSize') && numel(fallbackEnvironment.mapSize) == 3
        mapSize = double(reshape(fallbackEnvironment.mapSize, 1, 3));
    else
        mapSize = [100, 100, 100];
    end
end

function point = getFallbackPoint(fallbackEnvironment, fieldName, defaultPoint)
    if isfield(fallbackEnvironment, fieldName) && numel(fallbackEnvironment.(fieldName)) == 3
        point = double(reshape(fallbackEnvironment.(fieldName), 1, 3));
    else
        point = double(reshape(defaultPoint, 1, 3));
    end
end

function fieldValue = getRequiredField(sourceStruct, fieldNames)
    fieldValue = [];
    for idx = 1:numel(fieldNames)
        fieldName = fieldNames{idx};
        if isfield(sourceStruct, fieldName) && ~isempty(sourceStruct.(fieldName))
            fieldValue = sourceStruct.(fieldName);
            return;
        end
    end

    error('Missing required terrain field. Checked: %s', strjoin(fieldNames, ', '));
end

function scalarValue = getOptionalScalar(sourceStruct, fieldName, defaultValue)
    scalarValue = defaultValue;
    if isfield(sourceStruct, fieldName) && ~isempty(sourceStruct.(fieldName))
        candidate = double(sourceStruct.(fieldName));
        if isscalar(candidate) && isfinite(candidate)
            scalarValue = candidate;
        end
    end
end

function vectorValue = getOptionalVector(sourceStruct, fieldName, defaultValue)
    vectorValue = double(reshape(defaultValue, 1, []));
    if isfield(sourceStruct, fieldName) && ~isempty(sourceStruct.(fieldName))
        candidate = double(sourceStruct.(fieldName));
        if numel(candidate) >= 3
            vectorValue = reshape(candidate(1:3), 1, 3);
        end
    end
end

function axisInfo = buildAxisInfo(rawAxis, structMin, structMax, expectedLength)
    axisValues = double(rawAxis);
    if isvector(axisValues)
        axisValues = axisValues(:)';
    elseif size(axisValues, 1) == 1 || size(axisValues, 2) == 1
        axisValues = axisValues(:)';
    else
        if size(axisValues, 1) == expectedLength
            axisValues = axisValues(:, 1)';
        elseif size(axisValues, 2) == expectedLength
            axisValues = axisValues(1, :);
        else
            error('Unable to interpret terrain axis with size %s.', mat2str(size(axisValues)));
        end
    end

    if numel(axisValues) ~= expectedLength
        error('Terrain axis length mismatch: expected %d, got %d.', expectedLength, numel(axisValues));
    end

    rawMin = min(axisValues);
    rawMax = max(axisValues);

    if ~isempty(structMin) && isfinite(structMin)
        rawMin = double(structMin);
    end
    if ~isempty(structMax) && isfinite(structMax)
        rawMax = double(structMax);
    end

    if rawMax <= rawMin
        normalizedAxis = linspace(0, max(expectedLength - 1, 1), expectedLength);
        targetMax = normalizedAxis(end);
    else
        targetMax = double(rawMax);
        normalizedAxis = ((axisValues - rawMin) ./ (rawMax - rawMin)) * targetMax;
    end

    axisInfo = struct();
    axisInfo.axis = axisValues;
    axisInfo.rawMin = rawMin;
    axisInfo.rawMax = rawMax;
    axisInfo.targetMax = targetMax;
    axisInfo.normalizedAxis = normalizedAxis;
end

function terrainGrid = orientGrid(rawGrid, xAxis, yAxis)
    if size(rawGrid, 1) == numel(yAxis) && size(rawGrid, 2) == numel(xAxis)
        terrainGrid = rawGrid;
    elseif size(rawGrid, 1) == numel(xAxis) && size(rawGrid, 2) == numel(yAxis)
        terrainGrid = rawGrid';
    else
        error('Terrain grid size %s does not match axis lengths [%d, %d].', ...
            mat2str(size(rawGrid)), numel(yAxis), numel(xAxis));
    end
end

function terrainGrid = normalizeTerrainHeights(rawGrid, zMax)
    rawGrid = double(rawGrid);
    rawGrid(~isfinite(rawGrid)) = 0;
    rawGrid = rawGrid - min(rawGrid(:));
    rawGrid = max(rawGrid, 0);

    maxRawHeight = max(rawGrid(:));
    if maxRawHeight <= 0
        terrainGrid = rawGrid;
        return;
    end

    targetMaxHeight = min(0.4 * zMax, zMax - 20);
    targetMaxHeight = max(10, targetMaxHeight);

    if maxRawHeight > targetMaxHeight
        terrainGrid = rawGrid * (targetMaxHeight / maxRawHeight);
    else
        terrainGrid = rawGrid;
    end
end

function point = normalizePlanPoint(rawPoint, xInfo, yInfo, zMax)
    point = double(reshape(rawPoint, 1, []));
    if numel(point) < 3
        point(3) = 10;
    end

    point = point(1:3);
    point(1) = normalizeHorizontalCoordinate(point(1), xInfo);
    point(2) = normalizeHorizontalCoordinate(point(2), yInfo);
    point(3) = max(0, min(point(3), zMax));
end

function value = normalizeHorizontalCoordinate(rawValue, axisInfo)
    if axisInfo.rawMax <= axisInfo.rawMin
        value = rawValue;
        return;
    end

    value = ((rawValue - axisInfo.rawMin) ./ (axisInfo.rawMax - axisInfo.rawMin)) * axisInfo.targetMax;
end

function point = sanitizeAnchorPoint(point, terrainGrid, terrainX, terrainY, mapSize)
    minFlightAltitude = 10;
    minTerrainClearance = 8;

    point = double(reshape(point, 1, 3));
    point(1) = max(0, min(point(1), mapSize(1)));
    point(2) = max(0, min(point(2), mapSize(2)));

    groundHeight = getTerrainHeight(point(1), point(2), terrainGrid, terrainX, terrainY);
    if ~isfinite(groundHeight)
        groundHeight = 0;
    end

    point(3) = max(point(3), max(minFlightAltitude, groundHeight + minTerrainClearance));
    point(3) = min(point(3), mapSize(3));
end
