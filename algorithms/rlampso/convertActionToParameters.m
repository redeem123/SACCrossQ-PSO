function subgroupParams = convertActionToParameters(action)
    % Paper Eq. 11: Convert tanh action [-1,1] to PSO parameters
    %
    % Per subgroup, 4D action [a0, a1, a2, a3]:
    %   w     = a[0] * 0.8 + 0.1
    %   scale = 1/(a[1] + a[2] + 0.00001) * a[3] * 8
    %   c1    = scale * a[1]
    %   c2    = scale * a[2]

    actionSize = length(action);
    numSubgroups = actionSize / 4;
    assert(mod(actionSize, 4) == 0, 'Action size must be divisible by 4');

    subgroupParams = zeros(numSubgroups, 3);

    for sg = 1:numSubgroups
        base = (sg - 1) * 4;
        a0 = action(base + 1);
        a1 = action(base + 2);
        a2 = action(base + 3);
        a3 = action(base + 4);

        % Eq. 11 — paper-exact
        w     = a0 * 0.8 + 0.1;
        scale = 1.0 / (a1 + a2 + 0.00001) * a3 * 8.0;
        c1    = scale * a1;
        c2    = scale * a2;

        subgroupParams(sg, :) = [w, c1, c2];
    end
end
