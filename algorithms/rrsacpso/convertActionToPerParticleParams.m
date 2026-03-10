function particleParams = convertActionToPerParticleParams(action, config, particles, iter, maxIterations)
    % Convert action to per-particle PSO parameters
    %
    % Supports three parameter granularity modes:
    %   - global: 3D action -> same params for all particles
    %   - 5subgroup: 15D action -> 5 groups with different params
    %   - per-particle: 120D action -> each particle has unique params
    %
    % Inputs:
    %   action: Nx1 action vector from SAC actor (bounded to [-1, 1] by tanh)
    %   config: RRSACPSO configuration struct
    %
    % Outputs:
    %   particleParams: [popSize x 3] matrix with [w, c1, c2] for each particle
    %
    % Parameter Ranges:
    %   w (inertia):  [0.1, 0.9]  - Standard PSO range
    %   c1 (cognitive): [0.5, 2.5] - Pull toward personal best
    %   c2 (social):   [0.5, 2.5] - Pull toward global best

    if nargin < 3
        particles = [];
    end
    if nargin < 4
        iter = 1;
    end
    if nargin < 5
        maxIterations = 1;
    end
    action = action(:);
    numParticles = config.popSize;
    paramsPerParticle = config.paramsPerParticle;

    % Determine parameter mode (robust to missing config flags)
    if isfield(config, 'paramMode')
        paramMode = config.paramMode;
    elseif isfield(config, 'usePerParticleActions')
        if config.usePerParticleActions
            paramMode = 'per-particle';
        elseif config.actionSize == 3
            paramMode = 'global';
        elseif config.actionSize == 15
            paramMode = '5subgroup';
        elseif config.actionSize == 9
            paramMode = 'rank-residual';
        else
            paramMode = 'per-particle';
        end
    else
        if config.actionSize == 3
            paramMode = 'global';
        elseif config.actionSize == 15
            paramMode = '5subgroup';
        elseif config.actionSize == 9
            paramMode = 'rank-residual';
        else
            paramMode = 'per-particle';
        end
    end

    % Resolve parameter ranges (allow overrides in config)
    wMin = 0.1;
    wMax = 0.9;
    c1Min = 0.5;
    c1Max = 2.5;
    c2Min = 0.5;
    c2Max = 2.5;
    if isfield(config, 'paramRanges')
        ranges = config.paramRanges;
        if isfield(ranges, 'wMin'), wMin = ranges.wMin; end
        if isfield(ranges, 'wMax'), wMax = ranges.wMax; end
        if isfield(ranges, 'c1Min'), c1Min = ranges.c1Min; end
        if isfield(ranges, 'c1Max'), c1Max = ranges.c1Max; end
        if isfield(ranges, 'c2Min'), c2Min = ranges.c2Min; end
        if isfield(ranges, 'c2Max'), c2Max = ranges.c2Max; end
    end

    % Initialize output
    particleParams = zeros(numParticles, paramsPerParticle);

    % Convert based on mode
    switch paramMode
        case 'global'
            % Same parameters for all particles
            assert(length(action) == 3, 'Global mode expects 3D action');

            % Convert action to parameters
            w = wMin + (action(1) + 1.0) / 2.0 * (wMax - wMin);
            c1 = c1Min + (action(2) + 1.0) / 2.0 * (c1Max - c1Min);
            c2 = c2Min + (action(3) + 1.0) / 2.0 * (c2Max - c2Min);

            % Apply bounds
            w = max(wMin, min(wMax, w));
            c1 = max(c1Min, min(c1Max, c1));
            c2 = max(c2Min, min(c2Max, c2));

            % Assign to all particles
            particleParams(:, 1) = w;
            particleParams(:, 2) = c1;
            particleParams(:, 3) = c2;

        case '5subgroup'
            % 5 groups with different parameters
            assert(length(action) == 15, '5-subgroup mode expects 15D action');

            groupEdges = round(linspace(0, numParticles, 6));
            for g = 1:5
                % Extract action for this group
                startIdx = (g - 1) * 3 + 1;
                a_w = action(startIdx);
                a_c1 = action(startIdx + 1);
                a_c2 = action(startIdx + 2);

                % Convert to parameters
                w = wMin + (a_w + 1.0) / 2.0 * (wMax - wMin);
                c1 = c1Min + (a_c1 + 1.0) / 2.0 * (c1Max - c1Min);
                c2 = c2Min + (a_c2 + 1.0) / 2.0 * (c2Max - c2Min);

                % Apply bounds
                w = max(wMin, min(wMax, w));
                c1 = max(c1Min, min(c1Max, c1));
                c2 = max(c2Min, min(c2Max, c2));

                % Assign to particles in this group
                groupStartIdx = groupEdges(g) + 1;
                groupEndIdx = groupEdges(g + 1);
                if groupStartIdx > groupEndIdx
                    continue;
                end
                particleParams(groupStartIdx:groupEndIdx, 1) = w;
                particleParams(groupStartIdx:groupEndIdx, 2) = c1;
                particleParams(groupStartIdx:groupEndIdx, 3) = c2;
            end

        case 'per-particle'
            % Each particle has unique parameters
            expectedSize = numParticles * paramsPerParticle;
            assert(length(action) == expectedSize, ...
                'Per-particle mode: expected %d, got %d', expectedSize, length(action));

            % Process each particle individually
            for p = 1:numParticles
                % Extract 3D action for this particle
                startIdx = (p - 1) * paramsPerParticle + 1;
                a_w = action(startIdx);
                a_c1 = action(startIdx + 1);
                a_c2 = action(startIdx + 2);

                % Convert from [-1, 1] to parameter ranges
                % Inertia weight: [0.4, 0.9]
                w = wMin + (a_w + 1.0) / 2.0 * (wMax - wMin);

                % Cognitive coefficient
                c1 = c1Min + (a_c1 + 1.0) / 2.0 * (c1Max - c1Min);

                % Social coefficient
                c2 = c2Min + (a_c2 + 1.0) / 2.0 * (c2Max - c2Min);

                % Apply bounds (safety) - MUST match intended ranges above!
                w = max(wMin, min(wMax, w));
                c1 = max(c1Min, min(c1Max, c1));
                c2 = max(c2Min, min(c2Max, c2));

                % Store parameters for this particle
                particleParams(p, :) = [w, c1, c2];
            end

        case 'rank-residual'
            % Low-dimensional latent control expanded with particle context.
            assert(length(action) == 9, 'Rank-residual mode expects 9D action');

            % Baseline schedule (deterministic prior).
            progress = min(1, max(0, (iter - 1) / max(1, maxIterations)));
            if isfield(config, 'baseParamSchedule')
                sched = config.baseParamSchedule;
            else
                sched = struct('wMax', wMax, 'wMin', wMin, ...
                    'c1Max', c1Max, 'c1Min', c1Min, ...
                    'c2Max', c2Max, 'c2Min', c2Min, ...
                    'inertiaPower', 1.0);
            end

            wBase = sched.wMax - (sched.wMax - sched.wMin) * (progress ^ sched.inertiaPower);
            c1Base = sched.c1Max - (sched.c1Max - sched.c1Min) * progress;
            c2Base = sched.c2Min + (sched.c2Max - sched.c2Min) * progress;

            scale = struct('w', 0.12, 'c1', 0.35, 'c2', 0.35);
            if isfield(config, 'residualActionScale')
                scale = config.residualActionScale;
            end

            rankCentered = zeros(numParticles, 1);
            velCentered = zeros(numParticles, 1);
            stagCentered = zeros(numParticles, 1);

            if ~isempty(particles)
                fitnessForRank = particles.bestFitness(:);
                invalidMask = ~isfinite(fitnessForRank);
                if any(invalidMask)
                    if any(~invalidMask)
                        worstValid = max(fitnessForRank(~invalidMask), [], 'omitnan');
                    else
                        worstValid = 1e9;
                    end
                    fitnessForRank(invalidMask) = worstValid + 1;
                end
                [~, order] = sort(fitnessForRank, 'ascend');
                ranks = zeros(numParticles, 1);
                ranks(order) = (0:numParticles-1)';
                if numParticles > 1
                    rankCentered = ranks / (numParticles - 1) - 0.5;
                end

                if isfield(particles, 'velocities') && ~isempty(particles.velocities)
                    velMag = sqrt(sum(particles.velocities.^2, 2));
                    velMean = mean(velMag);
                    velStd = std(velMag);
                    if velStd < 1e-8
                        velCentered = zeros(size(velMag));
                    else
                        velCentered = tanh((velMag - velMean) / (velStd + 1e-8));
                    end
                end

                if isfield(particles, 'stagnationCounter')
                    stagCounter = particles.stagnationCounter(:);
                    stagCentered = min(stagCounter / 40, 1) - 0.5;
                end
            end

            deltaW = scale.w * (action(1) + action(2) * rankCentered + action(3) * velCentered);
            deltaC1 = scale.c1 * (action(4) + action(5) * stagCentered + action(6) * rankCentered);
            deltaC2 = scale.c2 * (action(7) - action(8) * rankCentered - action(9) * stagCentered);

            particleParams(:, 1) = max(wMin, min(wMax, wBase + deltaW));
            particleParams(:, 2) = max(c1Min, min(c1Max, c1Base + deltaC1));
            particleParams(:, 3) = max(c2Min, min(c2Max, c2Base + deltaC2));

        otherwise
            error('Unsupported paramMode: %s', paramMode);
    end

    if isfield(config, 'verbose') && config.verbose
        % Detailed logging for debugging parameter bounds.
        w_mean = mean(particleParams(:,1));
        w_std = std(particleParams(:,1));
        w_min = min(particleParams(:,1));
        w_max = max(particleParams(:,1));
        w_at_bounds = sum(particleParams(:,1) >= (wMax - 0.01 * (wMax - wMin)) | particleParams(:,1) <= (wMin + 0.01 * (wMax - wMin)));

        c1_mean = mean(particleParams(:,2));
        c1_std = std(particleParams(:,2));
        c1_min = min(particleParams(:,2));
        c1_max = max(particleParams(:,2));
        c1_at_bounds = sum(particleParams(:,2) >= (c1Max - 0.01 * (c1Max - c1Min)) | particleParams(:,2) <= (c1Min + 0.01 * (c1Max - c1Min)));

        c2_mean = mean(particleParams(:,3));
        c2_std = std(particleParams(:,3));
        c2_min = min(particleParams(:,3));
        c2_max = max(particleParams(:,3));
        c2_at_bounds = sum(particleParams(:,3) >= (c2Max - 0.01 * (c2Max - c2Min)) | particleParams(:,3) <= (c2Min + 0.01 * (c2Max - c2Min)));

        fprintf('      [Params] w: %.3f±%.3f [%.3f-%.3f] (%d@bounds) | c1: %.3f±%.3f [%.3f-%.3f] (%d@bounds) | c2: %.3f±%.3f [%.3f-%.3f] (%d@bounds)\n', ...
            w_mean, w_std, w_min, w_max, w_at_bounds, ...
            c1_mean, c1_std, c1_min, c1_max, c1_at_bounds, ...
            c2_mean, c2_std, c2_min, c2_max, c2_at_bounds);
    end
end
