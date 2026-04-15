function actorNet = createActorNetwork(config)
    % Create SAC actor network for either the retained cross-scale state or
    % the fallback flat temporal summary.
    useCrossScale = isfield(config, 'useCrossScaleState') && config.useCrossScaleState;
    useCrossScaleBranching = useCrossScale && ...
        isfield(config, 'useCrossScaleBranching') && config.useCrossScaleBranching;
    useD2RLBackbone = isfield(config, 'useD2RLBackbone') && config.useD2RLBackbone;

    if ~useCrossScale || ~useCrossScaleBranching
        if useD2RLBackbone
            actorNet = createD2RLActorNetwork(config);
        else
            actorNet = createFlatActorNetwork(config);
        end
        return;
    end

    hiddenLayers = config.actorHiddenLayers;
    actionSize = config.actionSize;
    if isfield(config, 'useActorBatchNorm')
        useBatchNorm = config.useActorBatchNorm;
    else
        useBatchNorm = isfield(config, 'useBatchNorm') && config.useBatchNorm;
    end
    [blockNames, blockStarts, branchWidths] = getCrossScaleLayout(config, 'actor');
    fusedWidth = sum(branchWidths);

    lgraph = layerGraph();
    lgraph = addLayers(lgraph, featureInputLayer(config.stateSize, 'Name', 'state_input', 'Normalization', 'none'));
    lgraph = addLayers(lgraph, concatenationLayer(1, numel(blockNames), 'Name', 'state_concat_raw'));
    for i = 1:numel(blockNames)
        branch = buildStateBranch(blockNames{i}, blockStarts(i), branchWidths(i));
        lgraph = addLayers(lgraph, branch);
        lgraph = connectLayers(lgraph, 'state_input', sprintf('%s_slice', blockNames{i}));
        lgraph = connectLayers(lgraph, sprintf('%s_relu', blockNames{i}), sprintf('state_concat_raw/in%d', i));
    end

    fusedInput = 'state_concat_raw';
    if isfield(config, 'useCrossScaleGatedFusion') && config.useCrossScaleGatedFusion
        [lgraph, fusedInput] = addCrossScaleGate(lgraph, config, fusedInput, fusedWidth, ...
            config.crossScaleActorGateHiddenWidth, 'actor_gate_');
    end

    if isfield(config, 'useSimBaBackbone') && config.useSimBaBackbone
        [lgraph, trunkOutput] = addSimBaActorTrunk(lgraph, config, fusedInput, actionSize);
    else
        trunk = [
            buildHiddenStack(hiddenLayers, useBatchNorm, config.batchNormEpsilon, 'trunk_')
            fullyConnectedLayer(actionSize * 2, 'Name', 'output_fc')
        ];
        lgraph = addLayers(lgraph, trunk);
        lgraph = connectLayers(lgraph, fusedInput, 'trunk_fc1');
        trunkOutput = 'output_fc';
    end

    actorNet = dlnetwork(lgraph);
    actorNet = initializeNetworkWeights(actorNet);
end

function actorNet = createFlatActorNetwork(config)
    hiddenLayers = config.actorHiddenLayers;
    actionSize = config.actionSize;
    if isfield(config, 'useActorBatchNorm')
        useBatchNorm = config.useActorBatchNorm;
    else
        useBatchNorm = isfield(config, 'useBatchNorm') && config.useBatchNorm;
    end

    layers = [
        featureInputLayer(config.stateSize, 'Name', 'state_input', 'Normalization', 'none')
        buildHiddenStack(hiddenLayers, useBatchNorm, config.batchNormEpsilon)
        fullyConnectedLayer(actionSize * 2, 'Name', 'output_fc')
    ];

    actorNet = dlnetwork(layerGraph(layers));
    actorNet = initializeNetworkWeights(actorNet);
end

function actorNet = createD2RLActorNetwork(config)
    hiddenLayers = config.actorHiddenLayers;
    actionSize = config.actionSize;

    lgraph = layerGraph();
    lgraph = addLayers(lgraph, featureInputLayer(config.stateSize, 'Name', 'state_input', 'Normalization', 'none'));

    previous = 'state_input';
    for i = 1:numel(hiddenLayers)
        if i > 1
            concatName = sprintf('d2rl_concat%d', i);
            lgraph = addLayers(lgraph, concatenationLayer(1, 2, 'Name', concatName));
            lgraph = connectLayers(lgraph, previous, sprintf('%s/in1', concatName));
            lgraph = connectLayers(lgraph, 'state_input', sprintf('%s/in2', concatName));
            previous = concatName;
        end

        block = [
            fullyConnectedLayer(hiddenLayers(i), 'Name', sprintf('d2rl_fc%d', i))
            reluLayer('Name', sprintf('d2rl_relu%d', i))
        ];
        lgraph = addLayers(lgraph, block);
        lgraph = connectLayers(lgraph, previous, sprintf('d2rl_fc%d', i));
        previous = sprintf('d2rl_relu%d', i);
    end

    lgraph = addLayers(lgraph, fullyConnectedLayer(actionSize * 2, 'Name', 'output_fc'));
    lgraph = connectLayers(lgraph, previous, 'output_fc');

    actorNet = dlnetwork(lgraph);
    actorNet = initializeNetworkWeights(actorNet);
end

function layers = buildStateBranch(prefix, startIdx, width)
    layers = [
        functionLayer(@(X) sliceFeatureBlock(X, startIdx, startIdx + 8), ...
            'Name', sprintf('%s_slice', prefix), 'Formattable', true)
        fullyConnectedLayer(width, 'Name', sprintf('%s_fc', prefix))
        reluLayer('Name', sprintf('%s_relu', prefix))
    ];
end

function layers = buildHiddenStack(hiddenLayers, useBatchNorm, batchNormEpsilon, prefix)
    if nargin < 4
        prefix = '';
    end

    layers = [];
    for i = 1:numel(hiddenLayers)
        hiddenSize = hiddenLayers(i);
        layers = [layers
            fullyConnectedLayer(hiddenSize, 'Name', sprintf('%sfc%d', prefix, i))];
        if useBatchNorm
            layers = [layers
                batchNormalizationLayer('Name', sprintf('%sbn%d', prefix, i), ...
                    'Epsilon', batchNormEpsilon)];
        end
        layers = [layers
            reluLayer('Name', sprintf('%srelu%d', prefix, i))];
    end
end

function block = sliceFeatureBlock(X, startIdx, endIdx)
    block = X(startIdx:endIdx, :);
end

function [blockNames, blockStarts, branchWidths] = getCrossScaleLayout(config, branchType)
    blockSize = 9;
    numBlocks = config.stateSize / blockSize;
    assert(abs(numBlocks - round(numBlocks)) < 1e-8, ...
        'Cross-scale state size must be divisible by %d.', blockSize);
    numBlocks = round(numBlocks);

    if isfield(config, 'crossScaleBlockNames') && numel(config.crossScaleBlockNames) == numBlocks
        blockNames = config.crossScaleBlockNames;
    else
        blockNames = arrayfun(@(i) sprintf('block%d', i), 1:numBlocks, 'UniformOutput', false);
    end

    widthsField = sprintf('%sCrossScaleBranchWidths', branchType);
    if isfield(config, widthsField) && numel(config.(widthsField)) == numBlocks
        branchWidths = config.(widthsField);
    elseif numBlocks == 5
        branchWidths = [32, 28, 28, 16, 16];
    else
        branchWidths = repmat(24, 1, numBlocks);
    end

    blockStarts = 1 + (0:numBlocks-1) * blockSize;
end

function [lgraph, outputName] = addSimBaActorTrunk(lgraph, config, inputName, actionSize)
    width = config.simbaActorWidth;
    numBlocks = config.simbaActorBlocks;

    stem = [
        fullyConnectedLayer(width, 'Name', 'trunk_proj')
        reluLayer('Name', 'trunk_proj_relu')
    ];
    lgraph = addLayers(lgraph, stem);
    lgraph = connectLayers(lgraph, inputName, 'trunk_proj');
    previous = 'trunk_proj_relu';

    for i = 1:numBlocks
        blockPrefix = sprintf('trunk_block%d_', i);
        blockLayers = [
            fullyConnectedLayer(width, 'Name', [blockPrefix 'fc1'])
            reluLayer('Name', [blockPrefix 'relu1'])
            layerNormalizationLayer('Name', [blockPrefix 'ln'])
            fullyConnectedLayer(width, 'Name', [blockPrefix 'fc2'])
            additionLayer(2, 'Name', [blockPrefix 'add'])
            reluLayer('Name', [blockPrefix 'out'])
        ];
        lgraph = addLayers(lgraph, blockLayers);
        lgraph = connectLayers(lgraph, previous, [blockPrefix 'fc1']);
        lgraph = connectLayers(lgraph, previous, [blockPrefix 'add/in2']);
        previous = [blockPrefix 'out'];
    end

    outputLayer = fullyConnectedLayer(actionSize * 2, 'Name', 'output_fc');
    lgraph = addLayers(lgraph, outputLayer);
    lgraph = connectLayers(lgraph, previous, 'output_fc');
    outputName = 'output_fc';
end

function [lgraph, outputName] = addCrossScaleGate(lgraph, config, inputName, inputWidth, hiddenWidth, prefix)
    gateLayers = [
        fullyConnectedLayer(hiddenWidth, 'Name', [prefix 'fc1'])
        reluLayer('Name', [prefix 'relu1'])
        fullyConnectedLayer(inputWidth, 'Name', [prefix 'fc2'])
        tanhLayer('Name', [prefix 'tanh'])
        multiplicationLayer(2, 'Name', [prefix 'mul'])
        additionLayer(2, 'Name', [prefix 'add'])
    ];
    lgraph = addLayers(lgraph, gateLayers);
    lgraph = connectLayers(lgraph, inputName, [prefix 'fc1']);
    lgraph = connectLayers(lgraph, inputName, [prefix 'mul/in2']);
    lgraph = connectLayers(lgraph, inputName, [prefix 'add/in2']);
    outputName = [prefix 'add'];
end

