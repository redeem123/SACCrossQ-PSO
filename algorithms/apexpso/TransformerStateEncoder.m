classdef TransformerStateEncoder < handle
    % Transformer-based State Encoder with Multi-Head Attention
    %
    % Uses attention mechanism to encode temporal context from last N PSO iterations
    % into a fixed-size state representation for the SAC agent.
    %
    % Based on research:
    %   - Transformer architectures in RL (2024)
    %   - Multi-head attention for temporal dependencies
    %   - Positional encoding for sequence order
    %
    % Architecture:
    %   Input: [T × D] sequence of basic features (T=temporalWindow, D=basic features)
    %   → Multi-head self-attention (4 heads)
    %   → Feed-forward network
    %   → Output: [1 × outputDim] context-aware state

    properties
        config              % APEX-PSO configuration
        temporalWindow      % Number of past iterations to track
        numHeads            % Number of attention heads
        basicFeatureDim     % Dimension of basic features per iteration
        outputDim           % Final state dimension (45D)

        % Network parameters (initialized during setup)
        W_Q                 % Query projection weights
        W_K                 % Key projection weights
        W_V                 % Value projection weights
        W_O                 % Output projection weights
        W_FF1               % Feed-forward layer 1
        W_FF2               % Feed-forward layer 2
        b_FF1               % Bias for FF1
        b_FF2               % Bias for FF2

        % State
        featureHistory      % Circular buffer of basic features [temporalWindow × basicFeatureDim]
        historyIndex        % Current position in circular buffer
        isInitialized       % Whether encoder is initialized
    end

    methods
        function obj = TransformerStateEncoder(config)
            % Constructor
            obj.config = config;
            obj.temporalWindow = config.temporalWindow;
            obj.numHeads = config.attentionHeads;
            obj.basicFeatureDim = 9;  % 9 basic features per iteration
            obj.outputDim = config.stateSize;  % 45D final state

            % Initialize circular buffer
            obj.featureHistory = zeros(obj.temporalWindow, obj.basicFeatureDim);
            obj.historyIndex = 0;
            obj.isInitialized = false;

            % Initialize network weights
            obj.initializeWeights();
        end

        function initializeWeights(obj)
            % Initialize attention and feed-forward network weights
            % Using Xavier initialization

            headDim = obj.basicFeatureDim;  % Dimension per attention head

            % Multi-head attention weights (for each head)
            obj.W_Q = cell(obj.numHeads, 1);
            obj.W_K = cell(obj.numHeads, 1);
            obj.W_V = cell(obj.numHeads, 1);

            for h = 1:obj.numHeads
                scale = sqrt(2.0 / (obj.basicFeatureDim + headDim));
                obj.W_Q{h} = randn(obj.basicFeatureDim, headDim) * scale;
                obj.W_K{h} = randn(obj.basicFeatureDim, headDim) * scale;
                obj.W_V{h} = randn(obj.basicFeatureDim, headDim) * scale;
            end

            % Output projection
            totalHeadDim = headDim * obj.numHeads;
            scale = sqrt(2.0 / (totalHeadDim + obj.basicFeatureDim));
            obj.W_O = randn(totalHeadDim, obj.basicFeatureDim) * scale;

            % Feed-forward network: basicFeatureDim → 64 → outputDim
            ffHiddenDim = 64;
            scale1 = sqrt(2.0 / (obj.basicFeatureDim + ffHiddenDim));
            obj.W_FF1 = randn(obj.basicFeatureDim, ffHiddenDim) * scale1;
            obj.b_FF1 = zeros(1, ffHiddenDim);

            scale2 = sqrt(2.0 / (ffHiddenDim + obj.outputDim));
            obj.W_FF2 = randn(ffHiddenDim, obj.outputDim) * scale2;
            obj.b_FF2 = zeros(1, obj.outputDim);
        end

        function addIteration(obj, basicFeatures)
            % Add new iteration's basic features to circular buffer
            %
            % Inputs:
            %   basicFeatures: [9×1] vector of basic features for this iteration

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

            if ~obj.isInitialized
                % Not enough history: return zero state
                state = zeros(obj.outputDim, 1);
                return;
            end

            % Get ordered sequence from circular buffer
            sequence = obj.getOrderedSequence();  % [temporalWindow × basicFeatureDim]

            % Add positional encoding
            sequence = obj.addPositionalEncoding(sequence);

            % Multi-head self-attention
            attentionOutput = obj.multiHeadAttention(sequence);

            % Take last timestep output (most recent)
            contextVector = attentionOutput(end, :);  % [1 × basicFeatureDim]

            % Feed-forward network
            hidden = relu(contextVector * obj.W_FF1 + obj.b_FF1);
            state = (hidden * obj.W_FF2 + obj.b_FF2)';  % [outputDim × 1]
        end

        function sequence = getOrderedSequence(obj)
            % Get features in chronological order from circular buffer
            if obj.historyIndex < obj.temporalWindow
                % Not full yet
                sequence = obj.featureHistory(1:obj.historyIndex, :);
            else
                % Reorder circular buffer
                currentIdx = mod(obj.historyIndex - 1, obj.temporalWindow) + 1;
                if currentIdx == obj.temporalWindow
                    sequence = obj.featureHistory;
                else
                    sequence = [obj.featureHistory(currentIdx+1:end, :);
                               obj.featureHistory(1:currentIdx, :)];
                end
            end
        end

        function sequence = addPositionalEncoding(obj, sequence)
            % Add sinusoidal positional encoding to sequence
            [T, D] = size(sequence);
            posEncoding = zeros(T, D);

            for pos = 1:T
                for i = 1:2:D
                    posEncoding(pos, i) = sin(pos / (10000^(i/D)));
                    if i+1 <= D
                        posEncoding(pos, i+1) = cos(pos / (10000^(i/D)));
                    end
                end
            end

            sequence = sequence + posEncoding;
        end

        function output = multiHeadAttention(obj, sequence)
            % Multi-head self-attention mechanism
            %
            % Inputs:
            %   sequence: [T × D] temporal sequence
            % Returns:
            %   output: [T × D] attended sequence

            [T, D] = size(sequence);
            headOutputs = cell(obj.numHeads, 1);

            % Process each attention head
            for h = 1:obj.numHeads
                % Compute Q, K, V for this head
                Q = sequence * obj.W_Q{h};  % [T × headDim]
                K = sequence * obj.W_K{h};  % [T × headDim]
                V = sequence * obj.W_V{h};  % [T × headDim]

                % Scaled dot-product attention
                scores = Q * K' / sqrt(size(K, 2));  % [T × T]
                attentionWeights = softmax(scores, 2);  % Softmax over keys
                headOutputs{h} = attentionWeights * V;  % [T × headDim]
            end

            % Concatenate all heads
            concatenated = cat(2, headOutputs{:});  % [T × (numHeads*headDim)]

            % Output projection
            output = concatenated * obj.W_O;  % [T × D]
        end

        function reset(obj)
            % Reset encoder state (for new episode)
            obj.featureHistory = zeros(obj.temporalWindow, obj.basicFeatureDim);
            obj.historyIndex = 0;
            obj.isInitialized = false;
        end
    end
end

% Helper functions
function y = relu(x)
    y = max(0, x);
end

function y = softmax(x, dim)
    % Numerically stable softmax
    x_shifted = x - max(x, [], dim);
    exp_x = exp(x_shifted);
    y = exp_x ./ sum(exp_x, dim);
end
