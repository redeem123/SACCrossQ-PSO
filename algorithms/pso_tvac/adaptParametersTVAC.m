function [w, c1, c2] = adaptParametersTVAC(current_w, current_c1_unused, current_c2_unused, iter, maxIterations, initialW, c1Min, c1Max, c2Min, c2Max)
    % Time-Variant Acceleration Coefficients (TVAC)
    % c1 decreases and c2 increases linearly; inertia weight w decreases linearly 0.9 -> 0.1.
    %
    % Inputs:
    %   current_w      - current inertia (unused; w is kept constant)
    %   iter           - current iteration (1-based)
    %   maxIterations  - total number of iterations
    %   initialW       - unused for w schedule (kept for signature compatibility)
    %   c1Min,c1Max    - bounds for cognitive coefficient
    %   c2Min,c2Max    - bounds for social coefficient
    %
    % Outputs:
    %   w  - inertia weight (constant = initialW)
    %   c1 - cognitive coefficient at this iteration
    %   c2 - social coefficient at this iteration

    %#ok<INUSD>  %% Silence unused placeholders

    % Normalize progress to [0,1]
    progress = min(max(iter / maxIterations, 0), 1);

    % Inertia linearly decreases from 0.9 to 0.1
    wMax = 0.9; wMin = 0.1;
    w = wMax - (wMax - wMin) * progress;

    % TVAC schedules (linear for c1,c2):
    % c1 decreases from c1Max -> c1Min; c2 increases from c2Min -> c2Max
    c1 = c1Max - (c1Max - c1Min) * progress;
    c2 = c2Min + (c2Max - c2Min) * progress;

    % Clamp to bounds (defensive)
    c1 = max(c1Min, min(c1Max, c1));
    c2 = max(c2Min, min(c2Max, c2));
end
