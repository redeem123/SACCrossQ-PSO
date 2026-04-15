function reward = calculatePaperReward(yPrev, yCurr)
%CALCULATEPAPERREWARD SAC-SAPSO paper relative reward (Equation 25).
%
%   von Eschwege & Engelbrecht 2024, Mathematics 12(22):3481.
%   Handles all sign combinations; bounded in [0, 1] for improvement.
%
%   yPrev = global best fitness at previous observation
%   yCurr = global best fitness at current observation

    if yCurr == yPrev
        reward = 0;
        return;
    end

    beta = abs(yPrev) + abs(yCurr);

    if yPrev > 0 && yCurr > 0
        % Both positive: shift by beta
        reward = 2 * (yPrev + beta - yCurr - beta) / (yPrev + beta);
        % Simplifies to: 2 * (yPrev - yCurr) / (yPrev + beta)
        reward = 2 * (yPrev - yCurr) / (yPrev + beta + 1e-30);
    elseif yPrev < 0 && yCurr < 0
        % Both negative: shift by 2*beta
        reward = 2 * (yPrev + 2*beta - (yCurr + 2*beta)) / (yPrev + 2*beta + 1e-30);
        % Simplifies to: 2 * (yPrev - yCurr) / (yPrev + 2*beta)
    else
        % Sign transition (rare): fixed reward of 1
        reward = 1;
    end

    % Clamp to reasonable range
    reward = max(-1, min(1, reward));
end
