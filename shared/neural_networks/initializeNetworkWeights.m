function net = initializeNetworkWeights(net)
    % He initialization for dlnetwork Learnables.
    % Weights: He normal (sqrt(2/fan_in)), Bias: zeros.
    % Gate layers (name ending in "gate_fc2"): zero weights, bias = 2.0.

    learnables = net.Learnables;
    for i = 1:height(learnables)
        paramName = learnables.Parameter{i};
        layerName = string(learnables.Layer{i});
        if strcmp(paramName, 'Weights')
            w = learnables.Value{i};
            [outSz, inSz] = size(w);
            if endsWith(layerName, "gate_fc2")
                learnables.Value{i} = dlarray(zeros(outSz, inSz, 'single'));
            else
                learnables.Value{i} = dlarray(randn(outSz, inSz, 'single') * sqrt(2.0 / inSz));
            end
        elseif strcmp(paramName, 'Bias')
            b = learnables.Value{i};
            if endsWith(layerName, "gate_fc2")
                learnables.Value{i} = dlarray(ones(size(b), 'single') * 2.0);
            else
                learnables.Value{i} = dlarray(zeros(size(b), 'single'));
            end
        end
    end
    net.Learnables = learnables;
end
