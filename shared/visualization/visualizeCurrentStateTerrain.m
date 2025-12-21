function visualizeCurrentStateTerrain(finalPath, globalPath, obstacles, trees, terrainGrid, terrainX, terrainY, startPoint, goalPoint, mapSize, currentTime, isReplanning, terrain_map)
    % Visualize the current state of the simulation with terrain and trees
    clf;
    hold on;
    surf(terrainX, terrainY, terrainGrid, 'EdgeColor', 'none');

    % Validate and apply terrain colormap
    if nargin >= 13 && ~isempty(terrain_map) && size(terrain_map, 2) == 3
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
    
    % Plot trees with variable heights
    for i = 1:size(trees, 1)
        % Extract tree parameters
        x = trees(i,1);
        y = trees(i,2);
        z = trees(i,3);
        trunkRadius = trees(i,4);
        canopyRadius = trees(i,5);
        isCylinder = trees(i,6);
        treeHeight = trees(i,7);  % Use the actual height from the array
        
        % Draw trunk/cylinder
        [X, Y, Z] = cylinder(trunkRadius, 10);
        Z = Z * treeHeight + z;  % Use variable height instead of fixed 10
        
        if isCylinder
            % Cylinder - solid grey
            surf(X+x, Y+y, Z, 'FaceColor', [0.5, 0.5, 0.5], 'EdgeColor', 'none');
        else
            % Regular tree trunk - brown
            surf(X+x, Y+y, Z, 'FaceColor', [0.6, 0.3, 0], 'EdgeColor', 'none');
        end
        
        % Draw canopy only if it's not a cylinder
        if ~isCylinder
            [X, Y, Z] = sphere(20);
            X = canopyRadius * X + x;
            Y = canopyRadius * Y + y;
            Z = canopyRadius * Z + z + treeHeight * 0.8; % Canopy at 80% of tree height
            surf(X, Y, Z, 'FaceColor', [0.1, 0.7, 0.1], 'EdgeColor', 'none');
        end
    end
    
    % Plot dynamic obstacles
    for i = 1:size(obstacles, 1)
        [X, Y, Z] = sphere(20);
        X = obstacles(i,4) * X + obstacles(i,1);
        Y = obstacles(i,4) * Y + obstacles(i,2);
        Z = obstacles(i,4) * Z + obstacles(i,3);
        surf(X, Y, Z, 'FaceColor', [0.7, 0.7, 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    end
    
    % Plot global planned path
    if ~isempty(globalPath) && size(globalPath, 1) > 1
        plot3(globalPath(:,1), globalPath(:,2), globalPath(:,3), 'g--', 'LineWidth', 1);
    end
    
    % Plot actual path history
    plot3(finalPath(:,1), finalPath(:,2), finalPath(:,3), 'r-', 'LineWidth', 2);
    
    % Plot current UAV position
    currentPosition = finalPath(end,:);
    plot3(currentPosition(1), currentPosition(2), currentPosition(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    
    % Plot start and goal points
    plot3(startPoint(1), startPoint(2), startPoint(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    plot3(goalPoint(1), goalPoint(2), goalPoint(3), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    
    % Display planning status
    if isReplanning
        title(['Path Planning - Time: ', num2str(currentTime, '%.2f'), 's - Global Replanning']);
    else
        title(['Path Planning - Time: ', num2str(currentTime, '%.2f'), 's - Local Navigation']);
    end
    
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    grid off;
    axis equal;
    view(3);
    
    % Add legend
    legend('Terrain', 'Trees', 'Obstacles', 'Global Path', 'Path History', 'UAV Position', 'Start', 'Goal', 'Location', 'northeastoutside');
    
    % Add text with distance to goal
    distanceToGoal = norm(currentPosition - goalPoint);
    textInfo = sprintf('Distance to Goal: %.2f units', distanceToGoal);
    annotation('textbox', [0.15, 0.15, 0.2, 0.1], 'String', textInfo, 'EdgeColor', 'none', 'BackgroundColor', [1 1 1 0.7]);
    
    hold off;
end

