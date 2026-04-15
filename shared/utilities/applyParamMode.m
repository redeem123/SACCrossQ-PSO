function config = applyParamMode(config)
    % Map paramMode string to action/control flags.
    %
    % Sets: usePerParticleActions, useRankResidualControl, actionSize, targetEntropy.

    switch config.paramMode
        case 'global'
            config.usePerParticleActions = false;
            config.useRankResidualControl = false;
            config.actionSize = 3;
        case '5subgroup'
            config.usePerParticleActions = false;
            config.useRankResidualControl = false;
            config.actionSize = 15;
        case 'per-particle'
            config.usePerParticleActions = true;
            config.useRankResidualControl = false;
            config.actionSize = config.popSize * config.paramsPerParticle;
        case {'rank-residual', 'attractor-field'}
            config.usePerParticleActions = false;
            config.useRankResidualControl = true;
            config.actionSize = 9;
        otherwise
            error('Unsupported paramMode: %s', config.paramMode);
    end

    config.targetEntropy = -config.actionSize;
end
