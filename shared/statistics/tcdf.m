function p = tcdf(t, df)
    % Calculate cumulative distribution function for t-distribution
    % Using incomplete beta function relationship
    
    if df <= 0
        p = 0.5;
        return;
    end
    
    if t == 0
        p = 0.5;
        return;
    end
    
    % For large degrees of freedom, approximate with normal distribution
    if df > 100
        p = normcdf(t);
        return;
    end
    
    % Use the relationship: t-cdf = 0.5 + 0.5 * sign(t) * I_x(1/2, df/2)
    % where x = df/(df + t^2) and I_x is the regularized incomplete beta function
    
    x = df / (df + t^2);
    
    if t > 0
        p = 0.5 + 0.5 * incompleteBeta(0.5, df/2, 1-x);
    else
        p = 0.5 - 0.5 * incompleteBeta(0.5, df/2, 1-x);
    end
end

