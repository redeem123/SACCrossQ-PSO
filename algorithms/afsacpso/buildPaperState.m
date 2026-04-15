function state = buildPaperState(particles, iter, maxIterations, config)
%BUILDPAPERSTATE Build SAC-SAPSO paper state representation.
%
%   State = [v_norm_1, ..., v_norm_ns, %stable, %infeasible, %completion]
%   Dimensionality: popSize + 3
%
%   Per-particle velocity is normalized by search bounds and tanh-squashed
%   to [-1, 1], following Equation (22) of von Eschwege & Engelbrecht 2024.

    popSize = size(particles.cartesianPositions, 1);
    dims = size(particles.cartesianPositions, 2);

    % --- Per-particle normalized velocity (one scalar each) ---
    % Paper Eq 22: v_norm = tanh(2/(u-l) * (v - (l+u)/2))
    % For UAV: bounds are [1, mapSize] per waypoint dimension
    if isfield(config, 'mapSize') && ~isempty(config.mapSize)
        mapSize = config.mapSize;
        numWaypoints = dims / 3;
        lb = ones(1, dims);
        ub = repmat(mapSize(:)', 1, numWaypoints);
    else
        lb = ones(1, dims);
        ub = 100 * ones(1, dims);
    end

    % For benchmark functions, use config bounds if available
    if isfield(config, 'benchmarkLB') && ~isempty(config.benchmarkLB)
        lb = config.benchmarkLB;
        ub = config.benchmarkUB;
    end

    range = ub - lb;
    center = (lb + ub) / 2;
    range(range < 1e-8) = 1;

    vNorm = zeros(popSize, 1);
    for i = 1:popSize
        v = particles.velocities(i, :);
        normalized = 2 ./ range .* (v - center);
        % Mean absolute normalized velocity per particle, then tanh
        vNorm(i) = tanh(mean(abs(normalized)));
    end

    % --- % stable particles (Poli's convergence criterion) ---
    % c1 + c2 < 24(1 - w^2) / (7 - 5w), w in [-1, 1]
    stablePct = 0;
    if isfield(config, 'lastParticleParams') && ~isempty(config.lastParticleParams)
        params = config.lastParticleParams;
        w_vals = params(:, 1);
        c1_vals = params(:, 2);
        c2_vals = params(:, 3);
        stable = false(popSize, 1);
        for i = 1:popSize
            w = w_vals(i);
            if abs(w) < 1
                threshold = 24 * (1 - w^2) / (7 - 5*w);
                stable(i) = (c1_vals(i) + c2_vals(i)) < threshold;
            end
        end
        stablePct = mean(double(stable));
    end

    % --- % infeasible particles ---
    infeasiblePct = 0;
    if isfield(particles, 'isFeasible') && ~isempty(particles.isFeasible)
        infeasiblePct = 1 - mean(double(particles.isFeasible(:)));
    else
        % Check if any position is out of bounds
        pos = particles.cartesianPositions;
        outOfBounds = any(pos < lb, 2) | any(pos > ub, 2);
        infeasiblePct = mean(double(outOfBounds));
    end

    % --- % completion ---
    completionPct = iter / maxIterations;

    % --- Assemble state ---
    state = [vNorm; stablePct; infeasiblePct; completionPct];
end
