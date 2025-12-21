function effectSize = interpretEffectSize(absD)
    % Interpret Cohen's d effect size
    if absD < 0.2
        effectSize = 'Negligible';
    elseif absD < 0.5
        effectSize = 'Small';
    elseif absD < 0.8
        effectSize = 'Medium';
    else
        effectSize = 'Large';
    end
end

