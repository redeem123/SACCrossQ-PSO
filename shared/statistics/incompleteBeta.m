function result = incompleteBeta(a, b, x)
    % Regularized incomplete beta function I_x(a,b)
    % Simple approximation for our purposes
    
    if x <= 0
        result = 0;
        return;
    elseif x >= 1
        result = 1;
        return;
    end
    
    % Use continued fraction approximation
    % This is a simplified implementation
    if x < (a + 1) / (a + b + 2)
        result = betacf(a, b, x) * exp(a*log(x) + b*log(1-x) - logBeta(a, b)) / a;
    else
        result = 1 - betacf(b, a, 1-x) * exp(b*log(1-x) + a*log(x) - logBeta(a, b)) / b;
    end
end

