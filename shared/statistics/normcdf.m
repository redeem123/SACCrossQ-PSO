function p = normcdf(x)
    % Standard normal cumulative distribution function
    p = 0.5 * (1 + erf(x / sqrt(2)));
end

