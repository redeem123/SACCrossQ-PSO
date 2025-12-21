function cohensD = calculateCohensD(sample1, sample2)
    % Calculate Cohen's d effect size
    
    % Remove NaN values
    sample1 = sample1(~isnan(sample1));
    sample2 = sample2(~isnan(sample2));
    
    if length(sample1) < 2 || length(sample2) < 2
        cohensD = 0;
        return;
    end
    
    mean1 = mean(sample1);
    mean2 = mean(sample2);
    std1 = std(sample1);
    std2 = std(sample2);
    
    n1 = length(sample1);
    n2 = length(sample2);
    
    % Pooled standard deviation
    pooledStd = sqrt(((n1-1)*std1^2 + (n2-1)*std2^2) / (n1+n2-2));
    
    if pooledStd == 0
        cohensD = 0;
    else
        cohensD = (mean1 - mean2) / pooledStd;
    end
end

