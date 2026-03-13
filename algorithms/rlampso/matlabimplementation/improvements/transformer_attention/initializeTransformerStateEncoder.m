function transformerEncoder = initializeTransformerStateEncoder(inputFeatureDim, hiddenDim, numHeads)
    % Initialize Transformer State Encoder for particle representation
    %
    % Inputs:
    %   inputFeatureDim: Dimension of raw particle features (default: auto-computed)
    %   hiddenDim: Hidden dimension for attention (default: 64)
    %   numHeads: Number of attention heads (default: 4)
    %
    % Outputs:
    %   transformerEncoder: Structure containing all transformer parameters
    %
    % Architecture:
    %   - Input embedding: inputFeatureDim → hiddenDim
    %   - Multi-head self-attention: 4 heads, 64 hidden dims
    %   - Feed-forward network: 64 → 128 → 64
    %   - Temporal attention: Self-attention across time
    %   - Layer normalization at each stage

    % Default parameters
    if nargin < 1 || isempty(inputFeatureDim)
        % Auto-compute based on particle features
        % Features: position (15) + velocity (15) + fitness (1) + metadata (3) = 34
        inputFeatureDim = 34;
    end
    if nargin < 2 || isempty(hiddenDim)
        hiddenDim = 64;
    end
    if nargin < 3 || isempty(numHeads)
        numHeads = 4;
    end

    % Validate parameters
    if mod(hiddenDim, numHeads) ~= 0
        error('hiddenDim must be divisible by numHeads');
    end

    transformerEncoder = struct();
    transformerEncoder.hiddenDim = hiddenDim;
    transformerEncoder.numHeads = numHeads;
    transformerEncoder.headDim = hiddenDim / numHeads;
    transformerEncoder.inputFeatureDim = inputFeatureDim;

    % === EMBEDDING LAYER ===
    % Project input features to hidden dimension
    scale = sqrt(2 / (inputFeatureDim + hiddenDim));
    transformerEncoder.W_embed = randn(inputFeatureDim, hiddenDim) * scale;
    transformerEncoder.b_embed = zeros(hiddenDim, 1);

    % === MULTI-HEAD ATTENTION PARAMETERS ===
    headDim = hiddenDim / numHeads;
    scale_qkv = sqrt(2 / (hiddenDim + headDim));

    % Query, Key, Value weights for each head
    transformerEncoder.W_query = cell(numHeads, 1);
    transformerEncoder.W_key = cell(numHeads, 1);
    transformerEncoder.W_value = cell(numHeads, 1);

    for h = 1:numHeads
        transformerEncoder.W_query{h} = randn(headDim, hiddenDim) * scale_qkv;
        transformerEncoder.W_key{h} = randn(headDim, hiddenDim) * scale_qkv;
        transformerEncoder.W_value{h} = randn(headDim, hiddenDim) * scale_qkv;
    end

    % Output projection after concatenating heads
    scale_out = sqrt(2 / hiddenDim);
    transformerEncoder.W_output = randn(hiddenDim, hiddenDim) * scale_out;
    transformerEncoder.b_output = zeros(hiddenDim, 1);

    % Layer normalization parameters for attention
    transformerEncoder.gamma_attn = ones(hiddenDim, 1);
    transformerEncoder.beta_attn = zeros(hiddenDim, 1);

    % === FEED-FORWARD NETWORK ===
    % Two-layer FFN: hiddenDim → ffnHiddenDim → hiddenDim
    ffnHiddenDim = hiddenDim * 2;  % Typically 2-4x hidden dimension

    scale_ffn1 = sqrt(2 / (hiddenDim + ffnHiddenDim));
    transformerEncoder.W_ffn1 = randn(ffnHiddenDim, hiddenDim) * scale_ffn1;
    transformerEncoder.b_ffn1 = zeros(ffnHiddenDim, 1);

    scale_ffn2 = sqrt(2 / (ffnHiddenDim + hiddenDim));
    transformerEncoder.W_ffn2 = randn(hiddenDim, ffnHiddenDim) * scale_ffn2;
    transformerEncoder.b_ffn2 = zeros(hiddenDim, 1);

    % === TEMPORAL ATTENTION PARAMETERS ===
    % For attention across time steps
    scale_temporal = sqrt(2 / hiddenDim);
    transformerEncoder.W_temporal_query = randn(hiddenDim, hiddenDim) * scale_temporal;
    transformerEncoder.W_temporal_key = randn(hiddenDim, hiddenDim) * scale_temporal;
    transformerEncoder.W_temporal_value = randn(hiddenDim, hiddenDim) * scale_temporal;

    % Layer normalization for temporal attention
    transformerEncoder.gamma_temporal = ones(hiddenDim, 1);
    transformerEncoder.beta_temporal = zeros(hiddenDim, 1);

    % === OUTPUT LAYER NORMALIZATION ===
    transformerEncoder.gamma_out = ones(hiddenDim, 1);
    transformerEncoder.beta_out = zeros(hiddenDim, 1);

    % === TRAINING METADATA ===
    transformerEncoder.initialized = true;
    transformerEncoder.version = '1.0';

    fprintf('Transformer State Encoder Initialized:\n');
    fprintf('  Input Feature Dim: %d\n', inputFeatureDim);
    fprintf('  Hidden Dim: %d\n', hiddenDim);
    fprintf('  Number of Heads: %d\n', numHeads);
    fprintf('  Head Dim: %d\n', headDim);
    fprintf('  Total Parameters: %d\n', countTransformerParameters(transformerEncoder));
end

function totalParams = countTransformerParameters(encoder)
    % Count total trainable parameters in transformer

    totalParams = 0;

    % Embedding layer
    totalParams = totalParams + numel(encoder.W_embed) + numel(encoder.b_embed);

    % Multi-head attention
    for h = 1:encoder.numHeads
        totalParams = totalParams + numel(encoder.W_query{h});
        totalParams = totalParams + numel(encoder.W_key{h});
        totalParams = totalParams + numel(encoder.W_value{h});
    end
    totalParams = totalParams + numel(encoder.W_output) + numel(encoder.b_output);
    totalParams = totalParams + numel(encoder.gamma_attn) + numel(encoder.beta_attn);

    % Feed-forward network
    totalParams = totalParams + numel(encoder.W_ffn1) + numel(encoder.b_ffn1);
    totalParams = totalParams + numel(encoder.W_ffn2) + numel(encoder.b_ffn2);

    % Temporal attention
    totalParams = totalParams + numel(encoder.W_temporal_query);
    totalParams = totalParams + numel(encoder.W_temporal_key);
    totalParams = totalParams + numel(encoder.W_temporal_value);
    totalParams = totalParams + numel(encoder.gamma_temporal) + numel(encoder.beta_temporal);

    % Output layer norm
    totalParams = totalParams + numel(encoder.gamma_out) + numel(encoder.beta_out);
end
