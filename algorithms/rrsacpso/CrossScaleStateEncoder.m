classdef CrossScaleStateEncoder < handle
    % Configurable state encoder for online RL-PSO.
    %
    % This encoder supports several low-variance temporal abstractions so
    % candidate state components can be searched automatically:
    %   - delta18: current + slow-fast regime delta
    %   - delta27: current + regime delta + recent delta
    %   - delta36: current + regime delta + recent delta + volatility
    %   - delta45: current + regime delta + recent delta + volatility + acceleration
    %   - temporalstats45: current + short EMA + long EMA + volatility + trend

    properties
        config
        temporalWindow
        basicFeatureDim
        outputDim
        featureHistory
        historyIndex
        isInitialized
        shortAlpha
        longAlpha
        encoderMode
    end

    methods
        function obj = CrossScaleStateEncoder(config)
            obj.config = config;
            obj.temporalWindow = config.temporalWindow;
            obj.basicFeatureDim = 9;
            obj.outputDim = config.stateSize;
            obj.featureHistory = zeros(obj.temporalWindow, obj.basicFeatureDim);
            obj.historyIndex = 0;
            obj.isInitialized = false;
            obj.shortAlpha = 0.65;
            obj.longAlpha = 0.18;
            obj.encoderMode = obj.getConfigString('crossScaleEncoderMode', 'delta27');
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
            if obj.historyIndex == 0
                state = zeros(obj.outputDim, 1);
                return;
            end

            sequence = obj.getOrderedSequence();
            current = sequence(end, :)';
            previous = sequence(max(1, end - 1), :)';
            shortEMA = obj.computeEMA(sequence, obj.shortAlpha);
            longEMA = obj.computeEMA(sequence, obj.longAlpha);
            volatility = obj.computeVolatility(sequence, shortEMA, longEMA);
            recentDelta = current - previous;
            previousDelta = zeros(size(current));
            if size(sequence, 1) >= 3
                previousDelta = previous - sequence(end - 2, :)';
            end
            acceleration = recentDelta - previousDelta;
            regimeDelta = shortEMA - longEMA;
            trend = obj.computeTrend(sequence, shortEMA, longEMA, volatility, recentDelta);

            rawState = obj.encodeByMode(current, shortEMA, longEMA, volatility, ...
                recentDelta, regimeDelta, acceleration, trend);
            state = obj.fitOutputSize(rawState);
        end

        function reset(obj)
            obj.featureHistory = zeros(obj.temporalWindow, obj.basicFeatureDim);
            obj.historyIndex = 0;
            obj.isInitialized = false;
        end
    end

    methods (Access = private)
        function sequence = getOrderedSequence(obj)
            if obj.historyIndex < obj.temporalWindow
                sequence = obj.featureHistory(1:obj.historyIndex, :);
            else
                currentIdx = mod(obj.historyIndex - 1, obj.temporalWindow) + 1;
                if currentIdx == obj.temporalWindow
                    sequence = obj.featureHistory;
                else
                    sequence = [obj.featureHistory(currentIdx + 1:end, :);
                                obj.featureHistory(1:currentIdx, :)];
                end
            end
        end

        function ema = computeEMA(~, sequence, alpha)
            emaVec = sequence(1, :);
            for t = 2:size(sequence, 1)
                emaVec = alpha * sequence(t, :) + (1 - alpha) * emaVec;
            end
            ema = emaVec';
        end

        function volatility = computeVolatility(~, sequence, shortEMA, longEMA)
            centeredStd = std(sequence, 0, 1)';
            scaleBias = 0.15 * abs(shortEMA - longEMA);
            volatility = centeredStd + scaleBias;
            volatility = min(1.5, max(0.0, volatility));
        end

        function trend = computeTrend(~, sequence, shortEMA, longEMA, volatility, recentDelta)
            splitIdx = max(1, floor(size(sequence, 1) / 2));
            headMean = mean(sequence(1:splitIdx, :), 1)';
            tailMean = mean(sequence(splitIdx:end, :), 1)';
            coarseTrend = tailMean - headMean;
            multiScaleTrend = 0.45 * (shortEMA - longEMA) + 0.35 * coarseTrend + 0.20 * recentDelta;
            trend = tanh(multiScaleTrend ./ (0.10 + volatility));
        end

        function rawState = encodeByMode(obj, current, shortEMA, longEMA, volatility, ...
                recentDelta, regimeDelta, acceleration, trend)
            scale = 0.10 + volatility;
            regimeSignal = tanh(regimeDelta ./ scale);
            deltaSignal = tanh(recentDelta ./ scale);
            accelSignal = tanh(acceleration ./ scale);

            switch lower(obj.encoderMode)
                case 'delta18'
                    rawState = [current; regimeSignal];
                case 'delta27'
                    rawState = [current; regimeSignal; deltaSignal];
                case 'delta36'
                    rawState = [current; regimeSignal; deltaSignal; volatility];
                case 'delta45'
                    rawState = [current; regimeSignal; deltaSignal; volatility; accelSignal];
                case 'temporalstats45'
                    rawState = [current; shortEMA; longEMA; volatility; trend];
                otherwise
                    error('Unsupported crossScaleEncoderMode: %s', obj.encoderMode);
            end
        end

        function state = fitOutputSize(obj, rawState)
            if obj.outputDim == numel(rawState)
                state = rawState;
                return;
            end
            if obj.outputDim > numel(rawState)
                state = [rawState; zeros(obj.outputDim - numel(rawState), 1)];
            else
                state = rawState(1:obj.outputDim);
            end
        end

        function value = getConfigString(obj, fieldName, defaultValue)
            if isfield(obj.config, fieldName) && ~isempty(obj.config.(fieldName))
                value = char(string(obj.config.(fieldName)));
            else
                value = defaultValue;
            end
        end
    end
end
