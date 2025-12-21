function result = logBeta(a, b)
    % Logarithm of beta function
    result = gammaln(a) + gammaln(b) - gammaln(a + b);
end

