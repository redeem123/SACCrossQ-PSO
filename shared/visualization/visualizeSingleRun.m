function visualizeSingleRun(path, startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize)
    % VISUALIZESINGLERUN Plots the result of a single path planning run in 3D.
    %
    % Inputs:
    %   path: N-by-3 matrix of path waypoints
    %   startPoint: [x, y, z]
    %   goalPoint: [x, y, z]
    %   dangerZones: M-by-3 matrix [x, y, radius] or [x, y, z, radius]
    %   terrainGrid: Height map matrix
    %   terrainX, terrainY: Grid coordinates
    %   mapSize: [x_max, y_max, z_max]

    % Create or reuse figure
    h_fig = findobj('Tag', 'SingleRunViz');
    if isempty(h_fig)
        h_fig = figure('Tag', 'SingleRunViz', 'Name', 'Current Run Visualization', 'NumberTitle', 'off');
        % Initial setup
        view(3);
        grid on;
        axis equal;
        hold on;
        
        % Plot terrain (only once for performance if reusing figure, but clf handles clean slate)
        % For simplicity, we assume we want to refresh everything or check if axes are empty.
        % But terrain plotting is heavy. Let's just clf every time for correctness.
        clf(h_fig);
        h_ax = axes(h_fig);
        hold(h_ax, 'on');
        view(h_ax, 3);
        grid(h_ax, 'on');
        axis(h_ax, 'equal');
        
        % Plot terrain
        surf(h_ax, terrainX, terrainY, terrainGrid, 'EdgeColor', 'none');
        colormap(h_ax, summer);
        shading(h_ax, 'interp');
        light('Position', [-1 -1 1], 'Style', 'infinite');
        lighting(h_ax, 'gouraud');
        material(h_ax, 'dull');
        
        % Plot Danger Zones
        if ~isempty(dangerZones)
            theta = linspace(0, 2*pi, 20);
            for i = 1:size(dangerZones, 1)
                r = dangerZones(i, 3);
                xc = dangerZones(i, 1);
                yc = dangerZones(i, 2);
                [X, Y, Z] = cylinder(r, 20);
                Z = Z * mapSize(3);
                X = X + xc;
                Y = Y + yc;
                surf(h_ax, X, Y, Z, 'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            end
        end
        
        % Plot Start/Goal
        plot3(h_ax, startPoint(1), startPoint(2), startPoint(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        plot3(h_ax, goalPoint(1), goalPoint(2), goalPoint(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        
        xlim(h_ax, [0 mapSize(1)]);
        ylim(h_ax, [0 mapSize(2)]);
        zlim(h_ax, [0 mapSize(3)]);
        xlabel(h_ax, 'X'); ylabel(h_ax, 'Y'); zlabel(h_ax, 'Z');
    else
        % If figure exists, just clear the previous path lines? 
        % No, user might want to see updates. 
        % The request is "plot 3D visualization every run". 
        % To avoid flickering terrain, we could keep terrain and just delete old paths.
        % But strictly speaking, the run might be on a DIFFERENT scenario/terrain.
        % So safer to redraw.
        figure(h_fig);
        clf(h_fig);
        h_ax = axes(h_fig);
        hold(h_ax, 'on');
        view(h_ax, 3);
        grid(h_ax, 'on');
        axis(h_ax, 'equal');
        
        % Plot terrain
        surf(h_ax, terrainX, terrainY, terrainGrid, 'EdgeColor', 'none');
        colormap(h_ax, summer);
        shading(h_ax, 'interp');
        light('Position', [-1 -1 1], 'Style', 'infinite');
        lighting(h_ax, 'gouraud');
        material(h_ax, 'dull');
        
         % Plot Danger Zones
        if ~isempty(dangerZones)
            for i = 1:size(dangerZones, 1)
                r = dangerZones(i, 3);
                xc = dangerZones(i, 1);
                yc = dangerZones(i, 2);
                [X, Y, Z] = cylinder(r, 20);
                Z = Z * mapSize(3);
                X = X + xc;
                Y = Y + yc;
                surf(h_ax, X, Y, Z, 'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            end
        end
        
        % Plot Start/Goal
        plot3(h_ax, startPoint(1), startPoint(2), startPoint(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        plot3(h_ax, goalPoint(1), goalPoint(2), goalPoint(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        
        xlim(h_ax, [0 mapSize(1)]);
        ylim(h_ax, [0 mapSize(2)]);
        zlim(h_ax, [0 mapSize(3)]);
        xlabel(h_ax, 'X'); ylabel(h_ax, 'Y'); zlabel(h_ax, 'Z');
    end

    % Plot the path
    if ~isempty(path)
        plot3(h_ax, path(:,1), path(:,2), path(:,3), 'b-', 'LineWidth', 2);
    end
    
    drawnow;
end
