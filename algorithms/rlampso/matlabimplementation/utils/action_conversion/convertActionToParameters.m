function subgroupParams = convertActionToParameters(action)
    % Convert action to PSO parameters (supports variable subgroup sizes)
    % Based on RLAMPSO.pdf Equation 11 (page 11)
    %
    % Inputs:
    %   action: Nx1 action vector (bounded to [-1, 1] by tanh)
    %           Layout: [subgroup1(4D), subgroup2(4D), ..., subgroupM(4D)]
    %           Each 4D: [a_w, a_c1, a_c2, a_scale]
    %           Supports: 4D (global), 20D (5-subgroup), 160D (40 per-particle)
    %
    % Outputs:
    %   subgroupParams: Mx3 matrix with [w, c1, c2] for each subgroup
    %
    % Paper Specification (Equation 11):
    %   w = a_w * 0.4 + 0.5                          (range: [0.1, 0.9])
    %   scale = 1/(a_c1 + a_c2 + ε) * a_scale * 8
    %   c1 = scale * a_c1
    %   c2 = scale * a_c2

    % Determine number of subgroups from action size
    actionSize = length(action);
    numSubgroups = actionSize / 4;  % Each subgroup has 4 params
    assert(mod(actionSize, 4) == 0, 'Action size must be divisible by 4');

    % Initialize output matrix
    subgroupParams = zeros(numSubgroups, 3);
    epsilon = 1e-8;  % Prevent division by zero

    % Process each subgroup separately
    for subgroup = 1:numSubgroups
        % Extract 4D action for this subgroup
        startIdx = (subgroup - 1) * 4 + 1;
        a_w = action(startIdx);
        a_c1 = action(startIdx + 1);
        a_c2 = action(startIdx + 2);
        a_scale = action(startIdx + 3);

        % Paper-exact conversion (Equation 11)
        w = a_w * 0.4 + 0.5;  % Range: [0.1, 0.9]

        % Normalize a_c1, a_c2 to [0, 1] from [-1, 1]
        a_c1_norm = (a_c1 + 1.0) / 2.0;
        a_c2_norm = (a_c2 + 1.0) / 2.0;
        a_scale_norm = (a_scale + 1.0) / 2.0;

        % Calculate scale factor (prevents c1+c2 from being 0)
        scale = (1.0 / (a_c1_norm + a_c2_norm + epsilon)) * a_scale_norm * 8.0;

        % Calculate c1, c2 using scale
        c1 = scale * a_c1_norm;
        c2 = scale * a_c2_norm;

        % Apply reasonable bounds (paper doesn't specify, use standard ranges)
        w = max(0.1, min(0.9, w));
        c1 = max(0.5, min(2.5, c1));
        c2 = max(0.5, min(2.5, c2));

        % Store parameters for this subgroup
        subgroupParams(subgroup, :) = [w, c1, c2];
    end
end
