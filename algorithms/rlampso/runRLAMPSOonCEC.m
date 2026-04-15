function [bestFitness, convergenceHistory, trainedAgent] = runRLAMPSOonCEC(functionID, dimensions, config, preloadedAgent)
    % Run RLAMPSO on a CEC 2013 benchmark function (paper experiment).
    %
    % Thin wrapper around runRLAMPSOonBenchmark using CECBenchmarks.

    if nargin < 2, dimensions = 30; end
    if nargin < 3, config = RLAMPSO_Config('baseline'); end
    if nargin < 4, preloadedAgent = []; end

    [~, optimalValue] = CECBenchmarks(zeros(dimensions, 1), functionID);
    fitFcn = @(x) CECBenchmarks(x, functionID);

    [bestFitness, convergenceHistory, trainedAgent] = ...
        runRLAMPSOonBenchmark(fitFcn, dimensions, -100, 100, optimalValue, config, preloadedAgent);
end
