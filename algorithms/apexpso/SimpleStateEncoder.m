classdef SimpleStateEncoder < handle
    % Simplified fallback state encoder for RRSACPSO.
    %
    % Keeps the original non-cross-scale temporal summary path available for
    % direct head-to-head comparisons against the retained full encoder.

    properties
        config
        temporalWindow
        basicFeatureDim
        outputDim
        featureHistory
        historyIndex
        isInitialized
    end

    methods
        function obj = SimpleStateEncoder(config)
            obj.config = config;
            obj.temporalWindow = config.temporalWindow;
            obj.basicFeatureDim = 9;
            obj.outputDim = config.stateSize;
            obj.featureHistory = zeros(obj.temporalWindow, obj.basicFeatureDim);
            obj.historyIndex = 0;
            obj.isInitialized = false;
        end

        function addIteration(obj, basicFeatures)
            obj.historyIndex = obj.historyIndex + 1;
            bufferIdx = mod(obj.historyIndex - 1, obj.temporalWindow) + 1;
            obj.featureHistory(bufferIdx, :) = basicFeatures(:)';
            if obj.historyIndex >= obj.temporalWindow
                obj.isInitialized = true;
            end
        end

        function state = encode(obj)
            if ~obj.isInitialized
                state = obj.encodeWarmupState();
                return;
            end

            sequence = obj.getOrderedSequence();
            currentFeatures = sequence(end, :)';

            weights = exp(linspace(-1, 0, size(sequence, 1)))';
            weights = weights / sum(weights);
            weightedAvg = (weights' * sequence)';
            temporalMean = mean(sequence, 1)';
            temporalStd = std(sequence, 0, 1)';

            if size(sequence, 1) >= 2
                trend = sequence(end, :)' - sequence(end-1, :)';
            else
                trend = zeros(obj.basicFeatureDim, 1);
            end

            if obj.outputDim == 15
                state = [currentFeatures; temporalMean(1:3); trend(1:3)];
            else
                fullState = [currentFeatures; weightedAvg; temporalMean; temporalStd; trend];
                state = obj.fitOutputSize(fullState);
            end
        end

        function reset(obj)
            obj.featureHistory = zeros(obj.temporalWindow, obj.basicFeatureDim);
            obj.historyIndex = 0;
            obj.isInitialized = false;
        end
    end

    methods (Access = private)
        function state = encodeWarmupState(obj)
            if obj.historyIndex == 0
                state = zeros(obj.outputDim, 1);
                return;
            end

            currentIdx = mod(obj.historyIndex - 1, obj.temporalWindow) + 1;
            currentFeatures = obj.featureHistory(currentIdx, :)';
            if obj.outputDim == 15
                state = [currentFeatures; zeros(6, 1)];
            else
                state = obj.fitOutputSize(repmat(currentFeatures, 5, 1));
            end
        end

        function sequence = getOrderedSequence(obj)
            if obj.historyIndex < obj.temporalWindow
                sequence = obj.featureHistory(1:obj.historyIndex, :);
            else
                currentIdx = mod(obj.historyIndex - 1, obj.temporalWindow) + 1;
                if currentIdx == obj.temporalWindow
                    sequence = obj.featureHistory;
                else
                    sequence = [obj.featureHistory(currentIdx+1:end, :);
                                obj.featureHistory(1:currentIdx, :)];
                end
            end
        end

        function state = fitOutputSize(obj, rawState)
            if obj.outputDim == numel(rawState)
                state = rawState;
            elseif obj.outputDim > numel(rawState)
                state = [rawState; zeros(obj.outputDim - numel(rawState), 1)];
            else
                state = rawState(1:obj.outputDim);
            end
        end
    end
end
