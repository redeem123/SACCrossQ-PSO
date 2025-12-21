function createUAVTrajectoryAnimation(results, algorithms, dangerZones, terrainGrid, terrainX, terrainY, startPoint, goalPoint, mapSize, terrain_map, scenarioIdx)
    % Create animation comparing all algorithms dynamically
    h_fig = figure('Position', [100, 100, 1800, 1200], 'Color', 'w');  % Increased size
    
    % Setup axes
    ax = axes;
    hold on;

    % Plot terrain using NMOPSO GitHub style
    terrain_surface = surf(terrainX, terrainY, terrainGrid, 'EdgeColor', 'none');

    % Validate and apply terrain colormap
    if nargin >= 10 && ~isempty(terrain_map) && size(terrain_map, 2) == 3
        colormap(gcf, terrain_map);
    else
        % Fallback to default terrain colormap if invalid
        colormap(gcf, [
            0.2 0.4 0.1;    % Dark green for valleys/low areas
            0.3 0.6 0.2;    % Medium green for foothills
            0.6 0.5 0.3;    % Brown for mid elevations
            0.7 0.6 0.4;    % Light brown for higher elevations
            0.8 0.7 0.6;    % Tan for rocky areas
            0.9 0.9 0.9     % White/gray for peaks
        ]);
    end
    shading interp;              % Interpolate color across faces
    material dull;               % Mountains aren't shiny
    camlight left;               % Add a light over to the left
    lighting gouraud;            % Use decent lighting    
    
    % Plot Danger Zones (Cylinders)
    if ~isempty(dangerZones)
        for i = 1:size(dangerZones, 1)
            % dangerZones is [x, y, radius]
            x_center = dangerZones(i, 1);
            y_center = dangerZones(i, 2);
            radius = dangerZones(i, 3);
            
            % Generate cylinder
            [X, Y, Z] = cylinder(radius, 20);
            
            % Scale Z to extend from ground to map ceiling
            Z = Z * mapSize(3); 
            
            % Shift X and Y to center position
            X = X + x_center;
            Y = Y + y_center;
            
            % Plot as red semi-transparent cylinder
            surf(X, Y, Z, 'FaceColor', [1.0, 0.0, 0.0], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
            
            % Plot base circle for top-down visibility (at the top)
            theta = linspace(0, 2*pi, 40);
            x_circle = x_center + radius * cos(theta);
            y_circle = y_center + radius * sin(theta);
            
            % Place it at the top of the map so it's always visible in Top View
            patch(x_circle, y_circle, repmat(mapSize(3), size(x_circle)), 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'r');
        end
    end
    
    % Plot start and goal points
    start_handle = plot3(startPoint(1), startPoint(2), startPoint(3), 'ko', 'MarkerSize', 14, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    goal_handle = plot3(goalPoint(1), goalPoint(2), goalPoint(3), 'ko', 'MarkerSize', 14, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    
    % Generate unique colors for each algorithm with very distinct colors
    numAlgorithms = length(algorithms);
    
    customColors = [
        1.0, 0.0, 0.0;  % Red
        0.0, 0.0, 1.0;  % Blue  
        0.0, 0.8, 0.0;  % Green
        1.0, 0.5, 0.0;  % Orange
        0.5, 0.0, 0.5;  % Purple
        0.0, 0.8, 0.8;  % Cyan
        1.0, 0.0, 1.0;  % Magenta
        0.8, 0.8, 0.0;  % Yellow
        0.5, 0.5, 0.5;  % Gray
        0.0, 0.0, 0.0;  % Black - for SAEPSO-aligned SAEPSO
        1.0, 0.4, 0.7;  % Pink - for BS-SAPSO (very distinct from black)
        0.2, 0.6, 0.2;  % Dark Green
        0.6, 0.2, 0.8;  % Dark Purple
        0.8, 0.6, 0.2;  % Brown
    ];
    
    % Use custom colors, cycling if we have more algorithms than colors
    colors = zeros(numAlgorithms, 3);
    for i = 1:numAlgorithms
        colors(i, :) = customColors(mod(i-1, size(customColors, 1)) + 1, :);
    end
    
    % Plot global paths for each algorithm
    globalPathHandles = [];
    legendLabels = {'Start', 'Goal'};
    
    for i = 1:length(algorithms)
        alg = algorithms{i};  % Using cell array indexing
        globalPath = results.([alg.fieldName '_globalPath']);
        
        if ~isempty(globalPath)
            % Plot global path (continuous line)
            h_global = plot3(globalPath(:,1), globalPath(:,2), globalPath(:,3), ...
                           '-', 'Color', colors(i,:), 'LineWidth', 3);
            globalPathHandles = [globalPathHandles, h_global];
            
            % Add to legend
            legendLabels{end+1} = alg.displayName;
        end
    end
    
    % Set labels and title
    xlabel('X (m)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Y (m)', 'FontSize', 14, 'FontWeight', 'bold');
    zlabel('Z (m)', 'FontSize', 14, 'FontWeight', 'bold');
    % title(sprintf('Scenario %d: Algorithm Comparison (3D View)', scenarioIdx), 'FontSize', 16, 'FontWeight', 'bold');
    
    grid off;
    axis equal;
    
    % Force axis limits to map boundaries
    xlim([0, mapSize(1)]);
    ylim([0, mapSize(2)]);
    zlim([0, mapSize(3)]);

    view(3);  % 3D view
    
    % Create legend with better positioning
    allHandles = [start_handle, goal_handle, globalPathHandles];
    if length(allHandles) > 2
        legend(allHandles, legendLabels, 'Location', 'eastoutside', 'FontSize', 8, 'NumColumns', 1);
    end
    
    set(gca, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1.5, 'Box', 'on');
    
    % Set up figure for proper PDF sizing (3D view)
    set(gcf, 'Color', 'white');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [20, 12]);  % Wider paper size
    set(gcf, 'PaperPosition', [0, 0, 20, 12]);  % Full paper coverage
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'Renderer', 'opengl');
    
    resultsDir = getResultsDir();

    % Save 3D view
    finalFrameFilename = fullfile(resultsDir, sprintf('plot_Algorithm_Comparison_Scenario%d_3D', scenarioIdx));
    print(gcf, finalFrameFilename, '-dpdf', '-r600', '-opengl');
    
    % Now create top view
    view(2);  % Top-down view
    % title(sprintf('Scenario %d: Algorithm Comparison (Top View)', scenarioIdx), 'FontSize', 16, 'FontWeight', 'bold');
    
    % Adjust legend for top view
    if length(allHandles) > 2
        legend(allHandles, legendLabels, 'Location', 'eastoutside', 'FontSize', 8, 'NumColumns', 1);
    end
    
    % Make paths more visible in top view
    for i = 1:length(globalPathHandles)
        set(globalPathHandles(i), 'LineWidth', 4);
    end
    
    % Add grid for better navigation reference in top view
    grid on;
    set(gca, 'GridAlpha', 0.3);
    
    % Ensure proper aspect ratio for top view
    axis equal;
    xlim([0, mapSize(1)]);
    ylim([0, mapSize(2)]);
    
    % Set up figure for top view PDF
    set(gcf, 'PaperSize', [18, 12]);  % Different aspect ratio for top view
    set(gcf, 'PaperPosition', [0, 0, 18, 12]);
    
    % Save top view
    topViewFilename = fullfile(resultsDir, sprintf('plot_Algorithm_Comparison_Scenario%d_TopView', scenarioIdx));
    print(gcf, topViewFilename, '-dpdf', '-r600', '-opengl');
    
    % Restore 3D view for any subsequent operations
    view(3);
    grid off;
    % title(sprintf('Scenario %d: Algorithm Comparison (3D View)', scenarioIdx), 'FontSize', 16, 'FontWeight', 'bold');
    
    fprintf('\n=== Algorithm Comparison Complete for Scenario %d ===\n', scenarioIdx);
    fprintf('3D view saved as: %s.pdf\n', finalFrameFilename);
    fprintf('Top view saved as: %s.pdf\n', topViewFilename);
    
    % Display path length comparison
    fprintf('\n=== Path Length Comparison ===\n');
    for i = 1:length(algorithms)
        alg = algorithms{i};
        if isfield(results, [alg.fieldName '_finalPath'])
            finalPath = results.([alg.fieldName '_finalPath']);
            pathLength = calculatePathLength(finalPath);
            fprintf('%s: %.2f units\n', alg.displayName, pathLength);
        end
    end
end