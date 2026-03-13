function result = generate_pso_paths_preview(outputRoot)
%GENERATE_PSO_PATHS_PREVIEW Run PSO once on the default 4 scenarios and save previews.

    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(repoRoot);
    addpath(genpath(fullfile(repoRoot, 'algorithms')));
    addpath(genpath(fullfile(repoRoot, 'scripts')));
    addpath(genpath(fullfile(repoRoot, 'shared')));
    addpath(fullfile(repoRoot, 'data'));

    if nargin < 1 || isempty(outputRoot)
        outputRoot = fullfile(repoRoot, 'outputs', 'pso_paths');
    end
    figuresDir = fullfile(outputRoot, 'figures');
    if exist(outputRoot, 'dir') ~= 7
        mkdir(outputRoot);
    end
    if exist(figuresDir, 'dir') ~= 7
        mkdir(figuresDir);
    end

    scenarios = buildDefaultScenarioSet(fullfile(repoRoot, 'data'));
    baseEnvironment = struct( ...
        'mapSize', [100, 100, 100], ...
        'startPoint', [10, 95, 10], ...
        'goalPoint', [97, 2, 10], ...
        'terrainFile', '', ...
        'timeStep', 0.1, ...
        'totalTime', 0.1, ...
        'globalPlanInterval', 1e5, ...
        'pathDeviationThreshold', 1e5, ...
        'obstacleChangeThreshold', 1e5);

    % Match the baseline PSO settings used in the main comparison runner.
    popSize = 40;
    maxIterations = 600;
    w = 0.7;
    c1 = 1.5;
    c2 = 1.5;

    result = struct();
    result.generatedAt = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    result.outputRoot = outputRoot;
    scenarioTemplate = struct( ...
        'label', '', ...
        'terrainFile', '', ...
        'numDangerZones', 0, ...
        'mapSize', zeros(1, 3), ...
        'startPoint', zeros(1, 3), ...
        'goalPoint', zeros(1, 3), ...
        'path', zeros(0, 3), ...
        'pathLength', NaN, ...
        'executionTime', NaN, ...
        'bestFitness', NaN);
    result.scenarios = repmat(scenarioTemplate, numel(scenarios), 1);

    figureHandle = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1800, 1300]);
    layout = tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    figureHandle3d = figure('Visible', 'off', 'Color', 'w', 'Position', [120, 120, 1800, 1300]);
    layout3d = tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    for scenarioIdx = 1:numel(scenarios)
        scenario = scenarios(scenarioIdx);
        environment = baseEnvironment;
        environment.terrainFile = scenario.terrainFile;
        environment = resolveTerrainEnvironment(environment);
        environment = applyScenarioPointOverrides(environment, scenario);

        [terrainGrid, terrainX, terrainY] = generateFixedTerrain(environment.mapSize, scenarioIdx, environment.terrainFile);
        dangerZones = generateDangerZones(scenario.numDangerZones, environment.mapSize, terrainGrid, terrainX, terrainY);

        rng(1000 + scenarioIdx * 100 + 1, 'twister');
        [finalPath, pathLength, executionTime, ~, metrics] = directGlobalPlanning( ...
            environment.startPoint, environment.goalPoint, dangerZones, ...
            terrainGrid, terrainX, terrainY, environment.mapSize, ...
            'PSO', popSize, maxIterations, w, c1, c2, ...
            environment.globalPlanInterval, environment.pathDeviationThreshold, ...
            environment.obstacleChangeThreshold, environment.timeStep, environment.totalTime, []);

        scenarioResult = struct();
        scenarioResult.label = scenario.label;
        scenarioResult.terrainFile = scenario.terrainFile;
        scenarioResult.numDangerZones = scenario.numDangerZones;
        scenarioResult.mapSize = environment.mapSize;
        scenarioResult.startPoint = environment.startPoint;
        scenarioResult.goalPoint = environment.goalPoint;
        scenarioResult.path = finalPath;
        scenarioResult.pathLength = pathLength;
        scenarioResult.executionTime = executionTime;
        if isfield(metrics, 'actualBestFitness')
            scenarioResult.bestFitness = metrics.actualBestFitness;
        else
            scenarioResult.bestFitness = NaN;
        end
        result.scenarios(scenarioIdx) = scenarioResult;

        pathTable = array2table(finalPath, 'VariableNames', {'x', 'y', 'z'});
        writetable(pathTable, fullfile(outputRoot, sprintf('scenario%d_path.csv', scenarioIdx)));

        ax = nexttile(layout, scenarioIdx);
        contourf(ax, terrainX, terrainY, terrainGrid, 30, 'LineColor', 'none');
        hold(ax, 'on');
        if ~isempty(dangerZones)
            plotDangerZones(ax, dangerZones);
        end
        plot(ax, finalPath(:,1), finalPath(:,2), 'Color', [0.1, 0.3, 0.9], 'LineWidth', 2.2);
        plot(ax, environment.startPoint(1), environment.startPoint(2), 'gp', 'MarkerSize', 11, 'MarkerFaceColor', 'g');
        plot(ax, environment.goalPoint(1), environment.goalPoint(2), 'rp', 'MarkerSize', 11, 'MarkerFaceColor', 'r');
        axis(ax, 'equal');
        xlim(ax, [0 environment.mapSize(1)]);
        ylim(ax, [0 environment.mapSize(2)]);
        grid(ax, 'on');
        title(ax, sprintf('%s\nPath %.2f | %.2fs', scenario.label, pathLength, executionTime), ...
            'Interpreter', 'none', 'FontSize', 11);
        xlabel(ax, 'X');
        ylabel(ax, 'Y');

        ax3 = nexttile(layout3d, scenarioIdx);
        surf(ax3, terrainX, terrainY, terrainGrid, 'EdgeColor', 'none', 'FaceAlpha', 0.96);
        hold(ax3, 'on');
        if ~isempty(dangerZones)
            plotDangerZones3D(ax3, dangerZones, terrainGrid, terrainX, terrainY, environment.mapSize);
        end
        plot3(ax3, finalPath(:,1), finalPath(:,2), finalPath(:,3), ...
            'Color', [0.1, 0.3, 0.9], 'LineWidth', 2.6);
        plot3(ax3, environment.startPoint(1), environment.startPoint(2), environment.startPoint(3), ...
            'gp', 'MarkerSize', 11, 'MarkerFaceColor', 'g');
        plot3(ax3, environment.goalPoint(1), environment.goalPoint(2), environment.goalPoint(3), ...
            'rp', 'MarkerSize', 11, 'MarkerFaceColor', 'r');
        axis(ax3, 'tight');
        xlim(ax3, [0 environment.mapSize(1)]);
        ylim(ax3, [0 environment.mapSize(2)]);
        zlim(ax3, [0 environment.mapSize(3)]);
        grid(ax3, 'on');
        view(ax3, 48, 30);
        title(ax3, sprintf('%s\nGoal z=%.0f | Path %.2f', ...
            scenario.label, environment.goalPoint(3), pathLength), ...
            'Interpreter', 'none', 'FontSize', 11);
        xlabel(ax3, 'X');
        ylabel(ax3, 'Y');
        zlabel(ax3, 'Z');
    end

    colormap(figureHandle, turbo(256));
    previewPng = fullfile(figuresDir, 'pso_paths_4scenario_preview.png');
    exportgraphics(figureHandle, previewPng, 'Resolution', 180);
    close(figureHandle);

    colormap(figureHandle3d, turbo(256));
    previewPng3d = fullfile(figuresDir, 'pso_paths_4scenario_preview_3d.png');
    exportgraphics(figureHandle3d, previewPng3d, 'Resolution', 180);
    close(figureHandle3d);

    summary = struct2table(orderfields(dropPathField(result.scenarios)));
    writetable(summary, fullfile(outputRoot, 'pso_paths_summary.csv'));
    save(fullfile(outputRoot, 'pso_paths_4scenarios.mat'), 'result');

    result.previewImage = previewPng;
    result.previewImage3d = previewPng3d;
    result.summaryCsv = fullfile(outputRoot, 'pso_paths_summary.csv');
    result.matFile = fullfile(outputRoot, 'pso_paths_4scenarios.mat');
end

function plotDangerZones(ax, dangerZones)
    theta = linspace(0, 2 * pi, 160);
    for idx = 1:size(dangerZones, 1)
        x = dangerZones(idx, 1) + dangerZones(idx, 3) * cos(theta);
        y = dangerZones(idx, 2) + dangerZones(idx, 3) * sin(theta);
        patch(ax, x, y, [0.85, 0.1, 0.1], ...
            'FaceAlpha', 0.18, 'EdgeColor', [0.65, 0.0, 0.0], 'LineWidth', 1.0);
    end
end

function summaryStruct = dropPathField(scenarios)
    summaryStruct = rmfield(scenarios, 'path');
end

function plotDangerZones3D(ax, dangerZones, terrainGrid, terrainX, terrainY, mapSize)
    nTheta = 48;
    for idx = 1:size(dangerZones, 1)
        cx = dangerZones(idx, 1);
        cy = dangerZones(idx, 2);
        radius = dangerZones(idx, 3);
        baseZ = interp2(terrainX, terrainY, terrainGrid, cx, cy, 'linear', 0);
        [xc, yc, zc] = cylinder([radius radius], nTheta);
        xc = xc + cx;
        yc = yc + cy;
        zc = zc * max(mapSize(3) - baseZ, 1) + baseZ;
        surf(ax, xc, yc, zc, ...
            'FaceColor', [0.85, 0.1, 0.1], 'FaceAlpha', 0.16, ...
            'EdgeColor', [0.65, 0.0, 0.0], 'EdgeAlpha', 0.12);
    end
end
