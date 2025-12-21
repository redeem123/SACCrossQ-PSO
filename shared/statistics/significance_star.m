function stars = significance_star(p)
    if p < 0.001
        stars = '*** (highly significant)';
    elseif p < 0.01
        stars = '** (very significant)';
    elseif p < 0.05
        stars = '* (significant)';
    else
        stars = '(not significant)';
    end
end

