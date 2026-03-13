function [transformerState, attentionWeights] = calculateTransformerState(particles, iter, maxIterations, lastImprovementIteration, transformerEncoder, temporalBuffer)
    % IMPROVEMENT #2: Transformer-based attention state representation
    % Replaces the sin-encoding with multi-head self-attention over particle data
    %
    % Inputs:
    %   particles: PSO particle structure with positions, velocities, fitness
    %   iter: Current iteration
    %   maxIterations: Maximum iterations
    %   lastImprovementIteration: Last iteration with fitness improvement
    %   transformerEncoder: Transformer encoder network structure
    %   temporalBuffer: (optional) Buffer of last 5 states for temporal attention
    %
    % Outputs:
    %   transformerState: 64D attention-weighted state representation
    %   attentionWeights: Attention weights for visualization
    %
    % Architecture:
    %   - Input: Raw particle data (positions, velocities, fitness)
    %   - Positional encoding: Sinusoidal encoding for particle ordering
    %   - Multi-head self-attention: 4 heads, 64 hidden dimensions
    %   - Temporal attention: Self-attention across last 5 states
    %   - Output: 64D state vector

    % Get particle data
    positions = particles.cartesianPositions;  % (numParticles × dims)
    velocities = particles.velocities;         % (numParticles × dims)
    fitness = particles.fitness;               % (numParticles × 1)

    [numParticles, posDims] = size(positions);

    % === STEP 1: Create particle embeddings ===
    % Each particle: [position (normalized), velocity (normalized), fitness (normalized), metadata]

    % Normalize positions to [0, 1]
    positionsNorm = positions / max(abs(positions(:)) + 1e-8);

    % Normalize velocities
    velocitiesNorm = velocities / max(abs(velocities(:)) + 1e-8);

    % Normalize fitness
    fitnessNorm = (fitness - min(fitness)) / (max(fitness) - min(fitness) + 1e-8);

    % Add global metadata features
    iterProgress = iter / maxIterations;
    stagnation = (iter - lastImprovementIteration) / maxIterations;

    % Compute diversity metric (average distance from centroid)
    meanPosition = mean(positions, 1);
    distances = sqrt(sum((positions - meanPosition).^2, 2));
    diversity = mean(distances) / (max(distances) + 1e-8);

    % Create particle feature matrix: (numParticles × featureDim)
    % Features: [pos (posDims), vel (posDims), fitness (1), metadata (3)]
    particleFeatures = [positionsNorm, velocitiesNorm, fitnessNorm, ...
                        repmat([iterProgress, stagnation, diversity], numParticles, 1)];

    featureDim = size(particleFeatures, 2);

    % === STEP 2: Project to embedding dimension ===
    % Linear projection to hiddenDim (64)
    hiddenDim = transformerEncoder.hiddenDim;

    % particleFeatures: (numParticles × featureDim)
    % W_embed: (featureDim × hiddenDim)
    % Result: (numParticles × hiddenDim)
    embeddings = particleFeatures * transformerEncoder.W_embed + transformerEncoder.b_embed';

    % === STEP 3: Add positional encoding ===
    % Sinusoidal positional encoding for particle ordering
    posEncoding = computePositionalEncoding(numParticles, hiddenDim);
    embeddingsWithPos = embeddings + posEncoding;  % (numParticles × hiddenDim)

    % === STEP 4: Multi-head self-attention ===
    % Transpose to (hiddenDim × numParticles) for matrix operations
    X = embeddingsWithPos';  % (hiddenDim × numParticles)

    [attentionOutput, attentionWeights] = multiHeadSelfAttention(X, transformerEncoder);

    % === STEP 5: Feed-forward network ===
    % Two-layer FFN with ReLU
    ffnHidden = max(0, transformerEncoder.W_ffn1 * attentionOutput + transformerEncoder.b_ffn1);
    ffnOutput = transformerEncoder.W_ffn2 * ffnHidden + transformerEncoder.b_ffn2;

    % === STEP 6: Global pooling to get fixed-size representation ===
    % Use mean pooling across particles
    stateVector = mean(ffnOutput, 2);  % (hiddenDim × 1)

    % === STEP 7: Temporal attention (if buffer provided) ===
    if nargin >= 6 && ~isempty(temporalBuffer) && size(temporalBuffer, 2) > 0
        % temporalBuffer: (hiddenDim × T) where T is number of past states
        % Add current state to create sequence
        temporalSequence = [temporalBuffer, stateVector];  % (hiddenDim × T+1)

        % Self-attention across time dimension
        [temporalOutput, ~] = temporalSelfAttention(temporalSequence, transformerEncoder);

        % Use the last time step output
        stateVector = temporalOutput(:, end);
    end

    % === STEP 8: Layer normalization and output ===
    transformerState = layerNorm(stateVector, transformerEncoder.gamma_out, transformerEncoder.beta_out);

    % Ensure column vector
    transformerState = transformerState(:);
end

%% ============ HELPER FUNCTIONS ============

function posEnc = computePositionalEncoding(seqLen, hiddenDim)
    % Sinusoidal positional encoding (Vaswani et al., 2017)
    % PE(pos, 2i) = sin(pos / 10000^(2i/hiddenDim))
    % PE(pos, 2i+1) = cos(pos / 10000^(2i/hiddenDim))

    posEnc = zeros(seqLen, hiddenDim);
    positions = (0:seqLen-1)';  % (seqLen × 1)

    for i = 0:floor(hiddenDim/2)-1
        div_term = 10000^(2*i / hiddenDim);

        % Sin for even indices
        if 2*i+1 <= hiddenDim
            posEnc(:, 2*i+1) = sin(positions / div_term);
        end

        % Cos for odd indices
        if 2*i+2 <= hiddenDim
            posEnc(:, 2*i+2) = cos(positions / div_term);
        end
    end
end

function [output, attentionWeights] = multiHeadSelfAttention(X, encoder)
    % Multi-head self-attention mechanism
    % X: (hiddenDim × seqLen)
    %
    % Architecture:
    %   - numHeads = 4
    %   - headDim = hiddenDim / numHeads = 16
    %   - Each head computes attention independently
    %   - Concatenate and project

    numHeads = encoder.numHeads;
    hiddenDim = encoder.hiddenDim;
    headDim = hiddenDim / numHeads;
    seqLen = size(X, 2);

    % Store attention weights from all heads
    allAttentionWeights = cell(numHeads, 1);
    headOutputs = cell(numHeads, 1);

    for h = 1:numHeads
        % Get Q, K, V projection matrices for this head
        W_q = encoder.W_query{h};  % (headDim × hiddenDim)
        W_k = encoder.W_key{h};    % (headDim × hiddenDim)
        W_v = encoder.W_value{h};  % (headDim × hiddenDim)

        % Project to Q, K, V
        Q = W_q * X;  % (headDim × seqLen)
        K = W_k * X;  % (headDim × seqLen)
        V = W_v * X;  % (headDim × seqLen)

        % Scaled dot-product attention
        % Attention(Q, K, V) = softmax(QK^T / sqrt(headDim)) V
        scores = (Q' * K) / sqrt(headDim);  % (seqLen × seqLen)

        % Softmax over keys (dim 2)
        attentionWeights_h = softmax(scores, 2);  % (seqLen × seqLen)
        allAttentionWeights{h} = attentionWeights_h;

        % Apply attention to values
        headOutput = V * attentionWeights_h';  % (headDim × seqLen)
        headOutputs{h} = headOutput;
    end

    % Concatenate all head outputs
    multiHeadOutput = cat(1, headOutputs{:});  % (hiddenDim × seqLen)

    % Project back to hiddenDim
    output = encoder.W_output * multiHeadOutput + encoder.b_output;

    % Residual connection and layer norm
    output = layerNorm(output + X, encoder.gamma_attn, encoder.beta_attn);

    % Return average attention weights for visualization
    attentionWeights = zeros(size(allAttentionWeights{1}));
    for h = 1:numHeads
        attentionWeights = attentionWeights + allAttentionWeights{h};
    end
    attentionWeights = attentionWeights / numHeads;
end

function [output, attentionWeights] = temporalSelfAttention(sequence, encoder)
    % Self-attention across temporal dimension
    % sequence: (hiddenDim × T) where T is sequence length

    hiddenDim = size(sequence, 1);
    seqLen = size(sequence, 2);

    % Use temporal attention weights
    W_q = encoder.W_temporal_query;
    W_k = encoder.W_temporal_key;
    W_v = encoder.W_temporal_value;

    Q = W_q * sequence;
    K = W_k * sequence;
    V = W_v * sequence;

    % Scaled dot-product attention
    scores = (Q' * K) / sqrt(hiddenDim);
    attentionWeights = softmax(scores, 2);

    % Apply attention
    output = V * attentionWeights';

    % Residual and norm
    output = layerNorm(output + sequence, encoder.gamma_temporal, encoder.beta_temporal);
end

function y = softmax(x, dim)
    % Numerically stable softmax
    % x: input matrix
    % dim: dimension to apply softmax (1 for columns, 2 for rows)

    if nargin < 2
        dim = 1;
    end

    % Subtract max for numerical stability
    x_max = max(x, [], dim);
    x_shifted = x - x_max;

    % Compute exp and normalize
    exp_x = exp(x_shifted);
    y = exp_x ./ sum(exp_x, dim);
end

function y = layerNorm(x, gamma, beta)
    % Layer normalization
    % x: (D × N) input
    % gamma, beta: (D × 1) learnable parameters

    % Normalize across feature dimension (dim 1)
    epsilon = 1e-6;
    mu = mean(x, 1);
    sigma = std(x, 0, 1) + epsilon;

    x_normalized = (x - mu) ./ sigma;

    % Scale and shift
    y = gamma .* x_normalized + beta;
end
