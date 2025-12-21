function particleParams = convertActionToPerParticleParams(action, config)
    % Convert action to per-particle PSO parameters
    %
    % Supports three parameter granularity modes:
    %   - global: 3D action -> same params for all particles
    %   - 5subgroup: 15D action -> 5 groups with different params
    %   - per-particle: 120D action -> each particle has unique params
    %
    % Inputs:
    %   action: Nx1 action vector from SAC actor (bounded to [-1, 1] by tanh)
    %   config: APEX-PSO configuration struct
    %
    % Outputs:
    %   particleParams: [popSize x 3] matrix with [w, c1, c2] for each particle
    %
    % Parameter Ranges:
    %   w (inertia):  [0.4, 0.9]  - Standard PSO range
    %   c1 (cognitive): [0.5, 2.5] - Pull toward personal best
    %   c2 (social):   [0.5, 2.5] - Pull toward global best

    numParticles = config.popSize;
    paramsPerParticle = config.paramsPerParticle;

    % Determine parameter mode
    if config.usePerParticleActions
        paramMode = 'per-particle';
    elseif config.actionSize == 3
        paramMode = 'global';
    elseif config.actionSize == 15
        paramMode = '5subgroup';
    else
        error('Unknown parameter mode: actionSize=%d', config.actionSize);
    end

    % Initialize output
    particleParams = zeros(numParticles, paramsPerParticle);

    % Convert based on mode
    switch paramMode
        case 'global'
            % Same parameters for all particles
            assert(length(action) == 3, 'Global mode expects 3D action');

            % Convert action to parameters
            w = 0.4 + (action(1) + 1.0) / 2.0 * 0.5;
            c1 = 0.5 + (action(2) + 1.0) / 2.0 * 2.0;
            c2 = 0.5 + (action(3) + 1.0) / 2.0 * 2.0;

            % Apply bounds
            w = max(0.4, min(0.9, w));
            c1 = max(0.5, min(2.5, c1));
            c2 = max(0.5, min(2.5, c2));

            % Assign to all particles
            particleParams(:, 1) = w;
            particleParams(:, 2) = c1;
            particleParams(:, 3) = c2;

        case '5subgroup'
            % 5 groups with different parameters
            assert(length(action) == 15, '5-subgroup mode expects 15D action');

            particlesPerGroup = numParticles / 5;
            for g = 1:5
                % Extract action for this group
                startIdx = (g - 1) * 3 + 1;
                a_w = action(startIdx);
                a_c1 = action(startIdx + 1);
                a_c2 = action(startIdx + 2);

                % Convert to parameters
                w = 0.4 + (a_w + 1.0) / 2.0 * 0.5;
                c1 = 0.5 + (a_c1 + 1.0) / 2.0 * 2.0;
                c2 = 0.5 + (a_c2 + 1.0) / 2.0 * 2.0;

                % Apply bounds
                w = max(0.4, min(0.9, w));
                c1 = max(0.5, min(2.5, c1));
                c2 = max(0.5, min(2.5, c2));

                % Assign to particles in this group
                groupStartIdx = (g - 1) * particlesPerGroup + 1;
                groupEndIdx = g * particlesPerGroup;
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
                w = 0.4 + (a_w + 1.0) / 2.0 * 0.5;  % Map [-1,1] -> [0.4, 0.9]

                % Cognitive coefficient: [0.5, 2.5]
                c1 = 0.5 + (a_c1 + 1.0) / 2.0 * 2.0;  % Map [-1,1] -> [0.5, 2.5]

                % Social coefficient: [0.5, 2.5]
                c2 = 0.5 + (a_c2 + 1.0) / 2.0 * 2.0;  % Map [-1,1] -> [0.5, 2.5]

                % Apply bounds (safety) - MUST match intended ranges above!
                % These bounds enforce the standard PSO parameter ranges
                w = max(0.4, min(0.9, w));      % Constrain to [0.4, 0.9]
                c1 = max(0.5, min(2.5, c1));    % Constrain to [0.5, 2.5]
                c2 = max(0.5, min(2.5, c2));    % Constrain to [0.5, 2.5]

                % Store parameters for this particle
                particleParams(p, :) = [w, c1, c2];
            end
    end

    % Detailed logging for debugging parameter bounds (ALWAYS log, not just verbose)
    w_mean = mean(particleParams(:,1));
    w_std = std(particleParams(:,1));
    w_min = min(particleParams(:,1));
    w_max = max(particleParams(:,1));
    w_at_bounds = sum(particleParams(:,1) >= 0.89 | particleParams(:,1) <= 0.41);

    c1_mean = mean(particleParams(:,2));
    c1_std = std(particleParams(:,2));
    c1_min = min(particleParams(:,2));
    c1_max = max(particleParams(:,2));
    c1_at_bounds = sum(particleParams(:,2) >= 2.49 | particleParams(:,2) <= 0.51);

    c2_mean = mean(particleParams(:,3));
    c2_std = std(particleParams(:,3));
    c2_min = min(particleParams(:,3));
    c2_max = max(particleParams(:,3));
    c2_at_bounds = sum(particleParams(:,3) >= 2.49 | particleParams(:,3) <= 0.51);

    % Log parameter statistics with bounds info
    fprintf('      [Params] w: %.3f±%.3f [%.3f-%.3f] (%d@bounds) | c1: %.3f±%.3f [%.3f-%.3f] (%d@bounds) | c2: %.3f±%.3f [%.3f-%.3f] (%d@bounds)\n', ...
        w_mean, w_std, w_min, w_max, w_at_bounds, ...
        c1_mean, c1_std, c1_min, c1_max, c1_at_bounds, ...
        c2_mean, c2_std, c2_min, c2_max, c2_at_bounds);
end
