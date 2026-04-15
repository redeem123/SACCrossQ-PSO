function [bestPath, bestFitness] = waypointDecomposedNM(psoPath, seedPositions, ...
    startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, ...
    numWaypoints, config)
    % WAYPOINTDECOMPOSEDNM  Multi-start Nelder-Mead local search on waypoints.
    %
    % After PSO converges, refine the solution by optimizing each waypoint
    % independently in 3D using fminsearch. Multiple starts (top pbests +
    % random paths) escape local optima and discover low-altitude valley
    % routes that PSO misses due to premature convergence.
    %
    % Inputs:
    %   psoPath        - (numWaypoints+2) x 3 path from PSO [start; waypoints; goal]
    %   seedPositions  - K x (numWaypoints*3) top personal-best positions (flat)
    %   startPoint     - [x, y, z] start position
    %   goalPoint      - [x, y, z] goal position
    %   dangerZones    - Nx3 [x, y, radius] danger zone definitions
    %   terrainGrid    - 2D terrain height map
    %   terrainX       - terrain X coordinate grid
    %   terrainY       - terrain Y coordinate grid
    %   numWaypoints   - number of intermediate waypoints
    %   config         - AFSACPSO config struct with polish* fields
    %
    % Outputs:
    %   bestPath    - (numWaypoints+2) x 3 refined path
    %   bestFitness - scalar fitness of refined path

    % Read config with defaults
    numStarts       = getOpt(config, 'polishNumStarts', 30);
    numSeedPaths    = getOpt(config, 'polishNumSeedPaths', 3);
    maxPasses       = getOpt(config, 'polishMaxPasses', 8);
    evalsPerWP      = getOpt(config, 'polishEvalsPerWaypoint', 400);
    stepXY          = getOpt(config, 'polishStepXY', 100);
    stepZ           = getOpt(config, 'polishStepZ', 15);
    mapSize         = config.mapSize;

    dims = numWaypoints * 3;

    % --- Build starting paths ---
    % Each row: 1 x dims flat vector of waypoint coordinates (no start/goal)
    startPaths = zeros(numStarts, dims);

    % Seeds 1..numSeedPaths: top pbests from PSO (includes global best)
    nSeeds = min(numSeedPaths, size(seedPositions, 1));
    for k = 1:nSeeds
        startPaths(k, :) = seedPositions(k, :);
    end

    % Remaining starts: fully random paths (x,y uniform in map, z = terrain + 10)
    for k = (nSeeds + 1):numStarts
        for j = 1:numWaypoints
            idx = (j-1)*3 + 1;
            x = rand() * mapSize(1);
            y = rand() * mapSize(2);
            [~, xIdx] = min(abs(terrainX(1,:) - x));
            [~, yIdx] = min(abs(terrainY(:,1) - y));
            z = terrainGrid(yIdx, xIdx) + 10;
            startPaths(k, idx:idx+2) = [x, y, z];
        end
    end

    % fminsearch options (TolX in scaled coords: 0.005 * scale = [0.5, 0.5, 0.075] real)
    fmOpts = optimset('MaxFunEvals', evalsPerWP, 'MaxIter', 300, ...
        'TolFun', 0.1, 'TolX', 0.005, 'Display', 'off');

    % --- Multi-start optimization ---
    bestFitness = Inf;
    bestWaypoints = startPaths(1, :);

    for s = 1:numStarts
        wp = startPaths(s, :);  % 1 x dims flat

        for pass = 1:maxPasses
            improved = false;
            for j = 1:numWaypoints
                idx = (j-1)*3 + 1;
                wp0 = wp(idx:idx+2);  % current waypoint [x, y, z]

                % Objective: evaluate full path with this waypoint varied
                objFun = @(v) evalWithWaypoint(v, wp, j, startPoint, goalPoint, ...
                    dangerZones, terrainGrid, terrainX, terrainY, numWaypoints, mapSize);

                fitBefore = objFun(wp0);

                % Scale coordinates so fminsearch's default 5% perturbation
                % produces steps of [stepXY, stepXY, stepZ] in real space.
                % fminsearch perturbs by 5% of each starting value, so we
                % center at 20 in scaled space: 5% * 20 = 1.0 scaled unit
                % = [stepXY, stepXY, stepZ] real units.
                origin = wp0;
                scale = [stepXY, stepXY, stepZ];
                center = 20;
                scaledObj = @(u) objFun(origin + (u - center) .* scale);
                [uOpt, fitAfter] = fminsearch(scaledObj, center * ones(1,3), fmOpts);
                wpOpt = origin + (uOpt - center) .* scale;

                if fitAfter < fitBefore - 0.01
                    % Enforce constraints on optimized waypoint
                    wpOpt = clampWaypoint(wpOpt, terrainGrid, terrainX, terrainY, mapSize);
                    wp(idx:idx+2) = wpOpt;
                    improved = true;
                end
            end

            if ~improved
                break;  % No waypoint improved this pass — converged
            end
        end

        % Evaluate final path for this start
        finalFit = evaluatePathFitness(wp, startPoint, goalPoint, dangerZones, ...
            terrainGrid, terrainX, terrainY, numWaypoints);

        if finalFit < bestFitness
            bestFitness = finalFit;
            bestWaypoints = wp;
        end
    end

    % Reconstruct full path
    bestPath = constructPathFromPSO(bestWaypoints, startPoint, goalPoint, numWaypoints, mapSize);
end

% =========================================================================

function fitness = evalWithWaypoint(v, wp, waypointIdx, startPoint, goalPoint, ...
    dangerZones, terrainGrid, terrainX, terrainY, numWaypoints, mapSize)
    % Evaluate path fitness with waypoint waypointIdx replaced by v
    idx = (waypointIdx-1)*3 + 1;
    wpTrial = wp;

    % Clamp candidate to terrain floor and map bounds
    v = clampWaypoint(v, terrainGrid, terrainX, terrainY, mapSize);
    wpTrial(idx:idx+2) = v;

    fitness = evaluatePathFitness(wpTrial, startPoint, goalPoint, dangerZones, ...
        terrainGrid, terrainX, terrainY, numWaypoints);

    % Penalize infeasible solutions heavily but keep them finite for NM
    if isinf(fitness)
        fitness = 1e9;
    end
end

function wp = clampWaypoint(wp, terrainGrid, terrainX, terrainY, mapSize)
    % Clamp x, y to map bounds
    wp(1) = max(1, min(wp(1), mapSize(1)));
    wp(2) = max(1, min(wp(2), mapSize(2)));

    % Enforce terrain + 10 floor on z
    [~, xIdx] = min(abs(terrainX(1,:) - wp(1)));
    [~, yIdx] = min(abs(terrainY(:,1) - wp(2)));
    terrainHeight = terrainGrid(yIdx, xIdx);
    wp(3) = max(wp(3), terrainHeight + 10);

    % Clamp z to map ceiling
    wp(3) = min(wp(3), mapSize(3));
end

function val = getOpt(config, field, default)
    if isfield(config, field)
        val = config.(field);
    else
        val = default;
    end
end
