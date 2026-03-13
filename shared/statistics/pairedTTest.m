function [tStat, pValue, df] = pairedTTest(sample1, sample2)
    % Perform paired t-test without using toolbox
    
    % Calculate differences
    differences = sample1 - sample2;
    n = length(differences);
    
    % Remove any NaN differences
    differences = differences(~isnan(differences));
    n = length(differences);
    
    if n < 2
        tStat = 0;
        pValue = 1;
        df = 0;
        return;
    end
    
    % Calculate mean and standard deviation of differences
    meanDiff = mean(differences);
    stdDiff = std(differences);
    
    % Calculate t-statistic
    % Degrees of freedom (always n-1)
    df = n - 1;

    if stdDiff == 0
        if meanDiff == 0
            tStat = 0;
            pValue = 1;
        else
            tStat = sign(meanDiff) * Inf;
            pValue = 0;
        end
    else
        tStat = meanDiff / (stdDiff / sqrt(n));

        % Calculate two-tailed p-value
        pValue = 2 * (1 - tcdf(abs(tStat), df));
    end
end

