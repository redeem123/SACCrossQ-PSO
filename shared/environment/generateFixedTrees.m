function trees = generateFixedTrees(numTrees, terrainGrid, terrainX, terrainY, mapSize)
    % Generate trees at fixed positions
    trees = zeros(numTrees, 7); % [x, y, z, trunkRadius, canopyRadius, isCylinder, height]
    
    % Predefined tree positions and properties
    treeSpacingX = mapSize(1) / ceil(sqrt(numTrees));
    treeSpacingY = mapSize(2) / ceil(sqrt(numTrees));
    idx = 1;
    
    for i = 1:ceil(sqrt(numTrees))
        for j = 1:ceil(sqrt(numTrees))
            if idx <= numTrees
                % Calculate position with a small fixed offset
                x = i * treeSpacingX - treeSpacingX/2 + 2*sin(i*j);
                y = j * treeSpacingY - treeSpacingY/2 + 2*cos(i+j);
                
                % Ensure within map boundaries
                x = max(1, min(x, mapSize(1)-1));
                y = max(1, min(y, mapSize(2)-1));
                
                % Find terrain height at this position
                [~, xIndex] = min(abs(terrainX(1,:) - x));
                [~, yIndex] = min(abs(terrainY(:,1) - y));
                z = terrainGrid(yIndex, xIndex);
                
                % Make every 3rd tree a big cylinder
                if mod(idx, 10) == 0
                    trunkRadius = 7.0 + 0.5*sin(i*j);  % Big cylinder radius
                    canopyRadius = 0;  % No canopy for cylinders
                    isCylinder = 1;
                    height = 40 ;  % Cylinder height
                else
                    % Regular tree parameters
                    trunkRadius = 0.7 + 0.3*sin(i*j);
                    canopyRadius = 2.5 + 1.0*cos(i+j);
                    isCylinder = 0;
                    height = 8 + 2*sin(i+j);  % Tree trunk height
                end
                
                trees(idx,:) = [x, y, z, trunkRadius, canopyRadius, isCylinder, height];
                idx = idx + 1;
            end
        end
    end
end

