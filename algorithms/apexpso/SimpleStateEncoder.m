classdef SimpleStateEncoder < handle
    % Simplified fallback state encoder for CQSAC-PSO.
    %
    % Uses temporal aggregation with learned weights when the structured
    % cross-scale encoder is disabled.
    %
    % Architecture:
    %   - Maintains circular buffer of last N iterations' features
    %   - Aggregates using weighted average (recent iterations weighted higher)
    %   - Expands features with statistical aggregates (mean, std, trend)
    %   - Outputs fixed 45D state vector

    properties
        config              % CQSAC-PSO configuration
        temporalWindow      % Number of past iterations to track
        basicFeatureDim     % Dimension of basic features per iteration (9D)
        outputDim           % Final state dimension (45D)

        % State
        featureHistory      % Circular buffer [temporalWindow × basicFeatureDim]
        historyIndex        % Current position in circular buffer
        isInitialized       % Whether encoder has enough history
    end

    methods
        function obj = SimpleStateEncoder(config)
            % Constructor
            obj.config = config;
            obj.temporalWindow = config.temporalWindow;  % 5
            obj.basicFeatureDim = 9;  % 9 basic features per iteration
            obj.outputDim = config.stateSize;  % 45D

            % Initialize circular buffer
            obj.featureHistory = zeros(obj.temporalWindow, obj.basicFeatureDim);
            obj.historyIndex = 0;
            obj.isInitialized = false;
        end

        function addIteration(obj, basicFeatures)
            % Add new iteration's basic features to circular buffer
            %
            % Inputs:
            %   basicFeatures: [9×1] vector of basic features

            obj.historyIndex = obj.historyIndex + 1;
            bufferIdx = mod(obj.historyIndex - 1, obj.temporalWindow) + 1;
            obj.featureHistory(bufferIdx, :) = basicFeatures';

            if obj.historyIndex >= obj.temporalWindow
                obj.isInitialized = true;
            end
        end

        function state = encode(obj)
            % Encode current temporal context into fixed-size state
            %
            % Returns:
            %   state: [45×1] context-aware state vector
            %
            % State composition (45D total):
            %   - Current features: 9D
            %   - Weighted average (recent bias): 9D
            %   - Temporal mean: 9D
            %   - Temporal std: 9D
            %   - Temporal trend (delta): 9D

            if ~obj.isInitialized
                % Not enough history: return zero-padded state of correct dimension
                if obj.historyIndex > 0
                    currentIdx = mod(obj.historyIndex - 1, obj.temporalWindow) + 1;
                    currentFeatures = obj.featureHistory(currentIdx, :)';
                    if obj.outputDim == 15
                        state = [currentFeatures; zeros(6, 1)];
                    elseif obj.outputDim == 45
                        state = repmat(currentFeatures, 5, 1);
                    else
                        state = [currentFeatures; zeros(obj.outputDim - 9, 1)];
                    end
                else
                    state = zeros(obj.outputDim, 1);
                end
                return;
            end

            % Get ordered sequence from circular buffer
            sequence = obj.getOrderedSequence();  % [temporalWindow × basicFeatureDim]

            % 1. Current features (most recent)
            currentFeatures = sequence(end, :)';  % [9×1]

            % 2. Weighted average with exponential decay (recent iterations weighted higher)
            weights = exp(linspace(-1, 0, size(sequence, 1)))';
            weights = weights / sum(weights);
            weightedAvg = (weights' * sequence)';  % [9×1]

            % 3. Temporal mean
            temporalMean = mean(sequence, 1)';  % [9×1]

            % 4. Temporal std (captures variability)
            temporalStd = std(sequence, 0, 1)';  % [9×1]

            % 5. Temporal trend (first-order difference)
            if size(sequence, 1) >= 2
                trend = sequence(end, :)' - sequence(end-1, :)';  % [9×1]
            else
                trend = zeros(obj.basicFeatureDim, 1);
            end

            % Concatenate components based on configured output dimension
            if obj.outputDim == 15
                % Simplified state for ablation: current + mean + trend
                state = [currentFeatures; temporalMean(1:3); trend(1:3)];
            elseif obj.outputDim == 45
                % Full state: all temporal features
                state = [currentFeatures; weightedAvg; temporalMean; temporalStd; trend];
            else
                % Fallback: pad or truncate to match outputDim
                fullState = [currentFeatures; weightedAvg; temporalMean; temporalStd; trend];
                if obj.outputDim > 45
                    state = [fullState; zeros(obj.outputDim - 45, 1)];
                else
                    state = fullState(1:obj.outputDim);
                end
            end

            % Verify size
            assert(length(state) == obj.outputDim, 'State dimension mismatch');
        end

        function sequence = getOrderedSequence(obj)
            % Get features in chronological order from circular buffer
            if obj.historyIndex < obj.temporalWindow
                % Not full yet
                sequence = obj.featureHistory(1:obj.historyIndex, :);
            else
                % Reorder circular buffer to chronological order
                currentIdx = mod(obj.historyIndex - 1, obj.temporalWindow) + 1;
                if currentIdx == obj.temporalWindow
                    sequence = obj.featureHistory;
                else
                    sequence = [obj.featureHistory(currentIdx+1:end, :);
                               obj.featureHistory(1:currentIdx, :)];
                end
            end
        end

        function reset(obj)
            % Reset encoder state (for new episode)
            obj.featureHistory = zeros(obj.temporalWindow, obj.basicFeatureDim);
            obj.historyIndex = 0;
            obj.isInitialized = false;
        end
    end
end
