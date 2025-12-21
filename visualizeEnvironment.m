%% visualizeEnvironment.m
% Professional visualization of UAV path planning environment
% Generates publication-quality figures for scientific papers
%
% This script visualizes Scenario 2 (5 danger zones) with:
% - 3D terrain surface
% - Start and goal positions
% - Cylindrical no-fly zones (danger zones)
% - Coordinate axes and labels
%
% Author: Generated for APEXPSO paper
% Date: 2024

function visualizeEnvironment()
    % Clear workspace
    close all;

    % Get project root directory
    currentDir = fileparts(mfilename('fullpath'));

    % Add paths
    addpath(fullfile(currentDir, 'shared', 'environment'));
    addpath(fullfile(currentDir, 'data'));

    % Configuration - Scenario 2 (5 danger zones)
    mapSize = [100, 100, 100];
    startPoint = [10, 95, 10];
    goalPoint = [97, 2, 10];
    numDangerZones = 5;
    % Scenario 2
    scenarioIdx = 2;

    % Generate terrain and danger zones
    [terrainGrid, terrainX, terrainY] = generateFixedTerrain(mapSize, scenarioIdx);
    dangerZones = generateDangerZones(numDangerZones, mapSize, terrainGrid, terrainX, terrainY);

    % Create figure with professional styling
    fig = figure('Position', [100, 100, 1200, 900], 'Color', 'white', 'Renderer', 'opengl');

    % Plot terrain surface
    ax = axes('Parent', fig);
    hold(ax, 'on');

    % Downsample terrain for smoother visualization and performance
    step = 2; % Plot every 2nd point
    
    % Terrain colormap - professional scientific style (Geographic/Hypsometric desaturated)
    h_surf = surf(ax, terrainX(1:step:end, 1:step:end), ...
                      terrainY(1:step:end, 1:step:end), ...
                      terrainGrid(1:step:end, 1:step:end), ...
                      'EdgeColor', 'none', ...
                      'FaceAlpha', 1.0, 'FaceLighting', 'gouraud');
         
    % Remove shininess (Specular -> 0)
    h_surf.SpecularStrength = 0.0;
    h_surf.DiffuseStrength = 0.8;
    h_surf.AmbientStrength = 0.5;

    % Professional Topographic Colormap (Sand -> Grey -> White)
    % Designed to be unobtrusive and high-contrast for overlaying data
    terrainColors = [
        0.90, 0.88, 0.82;  % Low: Light Sand/Beige
        0.75, 0.73, 0.68;  % Mid-Low: Stone Grey
        0.60, 0.58, 0.54;  % Mid-High: Slate
        0.95, 0.95, 0.95   % High: Snow/White
    ];
    
    n_colors = 256;
    terrainCmap = interp1(linspace(0, 1, size(terrainColors, 1)), terrainColors, linspace(0, 1, n_colors));
    colormap(ax, terrainCmap);
    
    % Matte finish
    material(ax, 'dull');
    
    % Soft, shadow-filling light
    light('Position', [0, 0, 1], 'Style', 'infinite'); % Overhead light
    lighting(ax, 'gouraud');

    % Plot danger zones as semi-transparent cylinders
    cylinderHeight = 80;  % Height of cylinders (visual representation)
    cylinderResolution = 50;  % Number of points for cylinder surface
    
    for i = 1:size(dangerZones, 1)
        x_center = dangerZones(i, 1);
        y_center = dangerZones(i, 2);
        radius = dangerZones(i, 3);

        % Get terrain height at cylinder base
        baseHeight = interp2(terrainX, terrainY, terrainGrid, x_center, y_center, 'linear', 0);

        % Create cylinder surface
        theta = linspace(0, 2 * pi, cylinderResolution);
        z_cyl = linspace(baseHeight, cylinderHeight, 20);
        [Theta, Z_cyl] = meshgrid(theta, z_cyl);
        X_cyl = x_center + radius * cos(Theta);
        Y_cyl = y_center + radius * sin(Theta);

        % Plot cylinder with red semi-transparent surface
        surf(ax, X_cyl, Y_cyl, Z_cyl, 'FaceColor', [0.85, 0.2, 0.2], ...
             'EdgeColor', 'none', 'FaceAlpha', 0.6);

        % Plot cylinder top cap
        [X_cap, Y_cap] = meshgrid(linspace(-radius, radius, 30));
        mask = (X_cap.^2 + Y_cap.^2) <= radius^2;
        Z_cap = ones(size(X_cap)) * cylinderHeight;
        Z_cap(~mask) = NaN;
        X_cap_shifted = X_cap + x_center;
        Y_cap_shifted = Y_cap + y_center;

        surf(ax, X_cap_shifted, Y_cap_shifted, Z_cap, 'FaceColor', [0.85, 0.2, 0.2], ...
             'EdgeColor', 'none', 'FaceAlpha', 0.6);

        % Plot danger zone outline on terrain (projected circle)
        x_circle = x_center + radius * cos(theta);
        y_circle = y_center + radius * sin(theta);
        z_circle = interp2(terrainX, terrainY, terrainGrid, x_circle, y_circle, 'linear', 0) + 0.5;
        plot3(ax, x_circle, y_circle, z_circle, 'r-', 'LineWidth', 2);
    end

    % Plot start point
    startZ = interp2(terrainX, terrainY, terrainGrid, startPoint(1), startPoint(2), 'linear', 0) + startPoint(3);
    scatter3(ax, startPoint(1), startPoint(2), startZ, 200, 'o', 'filled', ...
             'MarkerFaceColor', [0.2, 0.7, 0.3], 'MarkerEdgeColor', 'k', 'LineWidth', 2);

    % Start point label
    text(ax, startPoint(1) + 3, startPoint(2) - 5, startZ + 5, 'Start', ...
         'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.1, 0.5, 0.2]);

    % Plot goal point
    goalZ = interp2(terrainX, terrainY, terrainGrid, goalPoint(1), goalPoint(2), 'linear', 0) + goalPoint(3);
    scatter3(ax, goalPoint(1), goalPoint(2), goalZ, 200, 'd', 'filled', ...
             'MarkerFaceColor', [0.9, 0.3, 0.1], 'MarkerEdgeColor', 'k', 'LineWidth', 2);

    % Goal point label
    text(ax, goalPoint(1) - 3, goalPoint(2) + 5, goalZ + 5, 'Goal', ...
         'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.7, 0.2, 0.1]);

    %% Plot vertical lines from terrain to start/goal (for visual clarity)
    % Start vertical line
    startTerrainZ = interp2(terrainX, terrainY, terrainGrid, startPoint(1), startPoint(2), 'linear', 0);
    plot3(ax, [startPoint(1), startPoint(1)], [startPoint(2), startPoint(2)], ...
          [startTerrainZ, startZ], 'g--', 'LineWidth', 1.5);

    % Goal vertical line
    goalTerrainZ = interp2(terrainX, terrainY, terrainGrid, goalPoint(1), goalPoint(2), 'linear', 0);
    plot3(ax, [goalPoint(1), goalPoint(1)], [goalPoint(2), goalPoint(2)], ...
          [goalTerrainZ, goalZ], 'r--', 'LineWidth', 1.5);

    % Axis formatting
    xlabel(ax, 'X (m)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel(ax, 'Y (m)', 'FontSize', 14, 'FontWeight', 'bold');
    zlabel(ax, 'Z (m)', 'FontSize', 14, 'FontWeight', 'bold');

    title(ax, 'UAV Path Planning Environment - Scenario 2 (5 Danger Zones)', ...
          'FontSize', 16, 'FontWeight', 'bold');

    % Set axis limits
    xlim(ax, [0, mapSize(1)]);
    ylim(ax, [0, mapSize(2)]);
    zlim(ax, [0, 80]);

    % Grid and appearance
    grid(ax, 'on');
    ax.GridAlpha = 0.3;
    ax.GridLineStyle = '-';
    ax.Box = 'on';
    ax.FontSize = 12;

    % Set view angle for optimal visualization
    view(ax, 30, 30);

    % Add lighting
    light('Position', [1, 1, 1], 'Style', 'infinite');
    light('Position', [-1, -1, 0.5], 'Style', 'infinite');
    lighting gouraud;

    % Colorbar for terrain elevation
    cb = colorbar(ax);
    cb.Label.String = 'Terrain Elevation (m)';
    cb.Label.FontSize = 12;
    cb.Label.FontWeight = 'bold';

    hold(ax, 'off');

    %% Add legend
    % Create dummy plots for legend
    hold(ax, 'on');
    h_terrain = surf(ax, nan(2), nan(2), nan(2), 'FaceColor', [0.45, 0.55, 0.30], 'EdgeColor', 'none');
    h_danger = surf(ax, nan(2), nan(2), nan(2), 'FaceColor', [0.85, 0.2, 0.2], 'FaceAlpha', 0.6, 'EdgeColor', 'none');
    h_start = scatter3(ax, NaN, NaN, NaN, 150, 'o', 'filled', 'MarkerFaceColor', [0.2, 0.7, 0.3], 'MarkerEdgeColor', 'k');
    h_goal = scatter3(ax, NaN, NaN, NaN, 150, 'd', 'filled', 'MarkerFaceColor', [0.9, 0.3, 0.1], 'MarkerEdgeColor', 'k');
    hold(ax, 'off');

    legend(ax, [h_terrain, h_danger, h_start, h_goal], ...
           {'Terrain Surface', 'Danger Zone (No-Fly)', 'Start Position', 'Goal Position'}, ...
           'Location', 'northeast', 'FontSize', 11);

    % Save figure
    outputDir = fullfile(currentDir, 'outputs', 'figures');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Save as high-resolution PNG
    outputFile = fullfile(outputDir, 'environment_scenario2.png');
    % Check if exportgraphics is available (R2020a+), else use print
    try
        exportgraphics(fig, outputFile, 'Resolution', 300);
        fprintf('Saved figure to: %s\n', outputFile);
        
        % Save as PDF for LaTeX
        outputFilePDF = fullfile(outputDir, 'environment_scenario2.pdf');
        exportgraphics(fig, outputFilePDF, 'ContentType', 'vector');
        fprintf('Saved PDF to: %s\n', outputFilePDF);
    catch
        % Fallback for older MATLAB or if exportgraphics fails
        print(fig, outputFile, '-dpng', '-r300');
        print(fig, fullfile(outputDir, 'environment_scenario2.pdf'), '-dpdf', '-r300');
        fprintf('Saved figure (using print) to: %s\n', outputFile);
    end

    fprintf('\nEnvironment visualization complete.\n');
    fprintf('Scenario: 2 (Medium complexity)\n');
    fprintf('Danger zones: %d\n', numDangerZones);
    fprintf('Map size: [%d, %d, %d]\n', mapSize(1), mapSize(2), mapSize(3));
    fprintf('Start: [%.1f, %.1f, %.1f]\n', startPoint(1), startPoint(2), startPoint(3));
    fprintf('Goal: [%.1f, %.1f, %.1f]\n', goalPoint(1), goalPoint(2), goalPoint(3));
end