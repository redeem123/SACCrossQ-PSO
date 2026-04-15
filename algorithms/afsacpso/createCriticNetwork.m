function criticNet = createCriticNetwork(config)
    % Create SAC critic network for either the retained cross-scale state or
    % the fallback flat temporal summary.
    useCrossScale = isfield(config, 'useCrossScaleState') && config.useCrossScaleState;
    useCrossScaleBranching = useCrossScale && ...
        isfield(config, 'useCrossScaleBranching') && config.useCrossScaleBranching;
    useAQECritic = isfield(config, 'useAQECritic') && config.useAQECritic;
    useDroQCritic = isfield(config, 'useDroQCritic') && config.useDroQCritic;
    useD2RLBackbone = isfield(config, 'useD2RLBackbone') && config.useD2RLBackbone;
    criticOutputSize = getCriticOutputSize(config);

    if ~useCrossScale || ~useCrossScaleBranching
        if useAQECritic
            criticNet = createAQECriticNetwork(config);
        elseif useDroQCritic
            if useD2RLBackbone
                criticNet = createD2RLDroQCriticNetwork(config);
            else
                criticNet = createDroQCriticNetwork(config);
            end
        elseif isfield(config, 'useResidualCriticDecomposition') && config.useResidualCriticDecomposition
            criticNet = createResidualCriticNetwork(config);
        elseif useD2RLBackbone
            criticNet = createD2RLFlatCriticNetwork(config);
        else
            criticNet = createFlatCriticNetwork(config);
        end
        return;
    end

    hiddenLayers = config.criticHiddenLayers;
    if isfield(config, 'useCriticBatchNorm')
        useBatchNorm = config.useCriticBatchNorm;
    else
        useBatchNorm = isfield(config, 'useBatchNorm') && config.useBatchNorm;
    end
    stateActionSize = config.stateSize + config.actionSize;
    [blockNames, blockStarts, branchWidths] = getCrossScaleLayout(config, 'critic');
    stateFusionWidth = sum(branchWidths);
    actionWidth = 20;
    actionStartIdx = config.stateSize + 1;
    actionEndIdx = config.stateSize + config.actionSize;

    lgraph = layerGraph();
    lgraph = addLayers(lgraph, featureInputLayer(stateActionSize, 'Name', 'state_action_input', 'Normalization', 'none'));
    lgraph = addLayers(lgraph, concatenationLayer(1, numel(blockNames), 'Name', 'state_concat_raw'));
    for i = 1:numel(blockNames)
        branch = buildFeatureBranch(blockNames{i}, blockStarts(i), branchWidths(i));
        lgraph = addLayers(lgraph, branch);
        lgraph = connectLayers(lgraph, 'state_action_input', sprintf('%s_slice', blockNames{i}));
        lgraph = connectLayers(lgraph, sprintf('%s_relu', blockNames{i}), sprintf('state_concat_raw/in%d', i));
    end

    stateFusionName = 'state_concat_raw';
    if isfield(config, 'useCrossScaleGatedFusion') && config.useCrossScaleGatedFusion
        [lgraph, stateFusionName] = addCrossScaleGate(lgraph, config, stateFusionName, stateFusionWidth, ...
            config.crossScaleCriticGateHiddenWidth, 'critic_gate_');
    end

    lgraph = addLayers(lgraph, concatenationLayer(1, 2, 'Name', 'fusion_concat'));

    actionBranch = [
        functionLayer(@(X) sliceFeatureBlock(X, actionStartIdx, actionEndIdx), ...
            'Name', 'action_slice', 'Formattable', true)
        fullyConnectedLayer(actionWidth, 'Name', 'action_fc')
        reluLayer('Name', 'action_relu')
    ];
    lgraph = addLayers(lgraph, actionBranch);
    lgraph = connectLayers(lgraph, 'state_action_input', 'action_slice');
    lgraph = connectLayers(lgraph, stateFusionName, 'fusion_concat/in1');
    lgraph = connectLayers(lgraph, 'action_relu', 'fusion_concat/in2');

    if useDroQCritic
        lgraph = addDroQCriticTrunk(lgraph, config, 'fusion_concat');
    elseif isfield(config, 'useSimBaBackbone') && config.useSimBaBackbone
        lgraph = addSimBaCriticTrunk(lgraph, config, 'fusion_concat');
    else
        trunk = [
            buildHiddenStack(hiddenLayers, useBatchNorm, config.batchNormEpsilon, 'trunk_')
            fullyConnectedLayer(criticOutputSize, 'Name', 'q_value')
        ];
        lgraph = addLayers(lgraph, trunk);
        lgraph = connectLayers(lgraph, 'fusion_concat', 'trunk_fc1');
    end

    criticNet = dlnetwork(lgraph);
    criticNet = initializeNetworkWeights(criticNet);
end

function criticNet = createAQECriticNetwork(config)
    hiddenLayers = config.criticHiddenLayers;
    if isfield(config, 'aqeCriticHiddenLayers') && ~isempty(config.aqeCriticHiddenLayers)
        hiddenLayers = config.aqeCriticHiddenLayers;
    end

    stateActionSize = config.stateSize + config.actionSize;
    criticOutputSize = getCriticOutputSize(config);

    layers = [
        featureInputLayer(stateActionSize, 'Name', 'state_action_input', 'Normalization', 'none')
        buildHiddenStack(hiddenLayers, false, config.batchNormEpsilon, 'trunk_')
        fullyConnectedLayer(criticOutputSize, 'Name', 'q_value')
    ];

    criticNet = dlnetwork(layerGraph(layers));
    criticNet = initializeNetworkWeights(criticNet);
end

function criticNet = createDroQCriticNetwork(config)
    stateActionSize = config.stateSize + config.actionSize;
    criticOutputSize = getCriticOutputSize(config);

    layers = [
        featureInputLayer(stateActionSize, 'Name', 'state_action_input', 'Normalization', 'none')
        buildDroQHiddenStack(config, 'trunk_')
        fullyConnectedLayer(criticOutputSize, 'Name', 'q_value')
    ];

    criticNet = dlnetwork(layerGraph(layers));
    criticNet = initializeNetworkWeights(criticNet);
end

function criticNet = createD2RLDroQCriticNetwork(config)
    stateActionSize = config.stateSize + config.actionSize;
    criticOutputSize = getCriticOutputSize(config);
    width = config.droqCriticWidth;
    depth = config.droqCriticDepth;
    dropoutProb = config.droqDropoutProbability;

    lgraph = layerGraph();
    lgraph = addLayers(lgraph, featureInputLayer(stateActionSize, 'Name', 'state_action_input', 'Normalization', 'none'));

    previous = 'state_action_input';
    for i = 1:depth
        if i > 1
            concatName = sprintf('d2rl_concat%d', i);
            lgraph = addLayers(lgraph, concatenationLayer(1, 2, 'Name', concatName));
            lgraph = connectLayers(lgraph, previous, sprintf('%s/in1', concatName));
            lgraph = connectLayers(lgraph, 'state_action_input', sprintf('%s/in2', concatName));
            previous = concatName;
        end

        block = [
            fullyConnectedLayer(width, 'Name', sprintf('d2rl_fc%d', i))
            dropoutLayer(dropoutProb, 'Name', sprintf('d2rl_drop%d', i))
            layerNormalizationLayer('Name', sprintf('d2rl_ln%d', i))
            reluLayer('Name', sprintf('d2rl_relu%d', i))
        ];
        lgraph = addLayers(lgraph, block);
        lgraph = connectLayers(lgraph, previous, sprintf('d2rl_fc%d', i));
        previous = sprintf('d2rl_relu%d', i);
    end

    lgraph = addLayers(lgraph, fullyConnectedLayer(criticOutputSize, 'Name', 'q_value'));
    lgraph = connectLayers(lgraph, previous, 'q_value');

    criticNet = dlnetwork(lgraph);
    criticNet = initializeNetworkWeights(criticNet);
end

function criticNet = createFlatCriticNetwork(config)
    hiddenLayers = config.criticHiddenLayers;
    if isfield(config, 'useCriticBatchNorm')
        useBatchNorm = config.useCriticBatchNorm;
    else
        useBatchNorm = isfield(config, 'useBatchNorm') && config.useBatchNorm;
    end
    stateActionSize = config.stateSize + config.actionSize;
    criticOutputSize = getCriticOutputSize(config);

    layers = [
        featureInputLayer(stateActionSize, 'Name', 'state_action_input', 'Normalization', 'none')
        buildHiddenStack(hiddenLayers, useBatchNorm, config.batchNormEpsilon)
        fullyConnectedLayer(criticOutputSize, 'Name', 'q_value')
    ];

    criticNet = dlnetwork(layerGraph(layers));
    criticNet = initializeNetworkWeights(criticNet);
end

function criticNet = createD2RLFlatCriticNetwork(config)
    hiddenLayers = config.criticHiddenLayers;
    stateActionSize = config.stateSize + config.actionSize;
    criticOutputSize = getCriticOutputSize(config);

    lgraph = layerGraph();
    lgraph = addLayers(lgraph, featureInputLayer(stateActionSize, 'Name', 'state_action_input', 'Normalization', 'none'));

    previous = 'state_action_input';
    for i = 1:numel(hiddenLayers)
        if i > 1
            concatName = sprintf('d2rl_concat%d', i);
            lgraph = addLayers(lgraph, concatenationLayer(1, 2, 'Name', concatName));
            lgraph = connectLayers(lgraph, previous, sprintf('%s/in1', concatName));
            lgraph = connectLayers(lgraph, 'state_action_input', sprintf('%s/in2', concatName));
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

    lgraph = addLayers(lgraph, fullyConnectedLayer(criticOutputSize, 'Name', 'q_value'));
    lgraph = connectLayers(lgraph, previous, 'q_value');

    criticNet = dlnetwork(lgraph);
    criticNet = initializeNetworkWeights(criticNet);
end

function criticNet = createResidualCriticNetwork(config)
    hiddenLayers = config.criticHiddenLayers;
    if isfield(config, 'useCriticBatchNorm')
        useBatchNorm = config.useCriticBatchNorm;
    else
        useBatchNorm = isfield(config, 'useBatchNorm') && config.useBatchNorm;
    end

    stateActionSize = config.stateSize + config.actionSize;
    stateStartIdx = 1;
    stateEndIdx = config.stateSize;
    actionStartIdx = config.stateSize + 1;
    actionEndIdx = config.stateSize + config.actionSize;
    actionWidth = config.criticActionBranchWidth;
    advantageWidth = config.criticAdvantageHiddenWidth;
    criticOutputSize = getCriticOutputSize(config);

    lgraph = layerGraph();
    lgraph = addLayers(lgraph, featureInputLayer(stateActionSize, 'Name', 'state_action_input', 'Normalization', 'none'));

    stateSlice = functionLayer(@(X) sliceFeatureBlock(X, stateStartIdx, stateEndIdx), ...
        'Name', 'state_slice', 'Formattable', true);
    actionSlice = functionLayer(@(X) sliceFeatureBlock(X, actionStartIdx, actionEndIdx), ...
        'Name', 'action_slice', 'Formattable', true);
    lgraph = addLayers(lgraph, stateSlice);
    lgraph = addLayers(lgraph, actionSlice);
    lgraph = connectLayers(lgraph, 'state_action_input', 'state_slice');
    lgraph = connectLayers(lgraph, 'state_action_input', 'action_slice');

    if isfield(config, 'useSimBaBackbone') && config.useSimBaBackbone
        [lgraph, stateFeatureName] = addSimBaProjectionTrunk(lgraph, config, 'state_slice', 'state_');
    else
        stateTrunk = buildHiddenStack(hiddenLayers, useBatchNorm, config.batchNormEpsilon, 'state_');
        lgraph = addLayers(lgraph, stateTrunk);
        lgraph = connectLayers(lgraph, 'state_slice', 'state_fc1');
        stateFeatureName = sprintf('state_relu%d', numel(hiddenLayers));
    end

    valueHead = [
        fullyConnectedLayer(max(64, floor(advantageWidth / 2)), 'Name', 'value_fc1')
        reluLayer('Name', 'value_relu1')
        fullyConnectedLayer(criticOutputSize, 'Name', 'value_fc')
    ];
    lgraph = addLayers(lgraph, valueHead);
    lgraph = connectLayers(lgraph, stateFeatureName, 'value_fc1');

    actionBranch = [
        fullyConnectedLayer(actionWidth, 'Name', 'action_fc1')
        reluLayer('Name', 'action_relu1')
    ];
    lgraph = addLayers(lgraph, actionBranch);
    lgraph = connectLayers(lgraph, 'action_slice', 'action_fc1');

    advantageFusion = concatenationLayer(1, 2, 'Name', 'adv_concat');
    advantageHead = [
        fullyConnectedLayer(advantageWidth, 'Name', 'adv_fc1')
        reluLayer('Name', 'adv_relu1')
        fullyConnectedLayer(criticOutputSize, 'Name', 'adv_fc')
    ];
    qAdd = additionLayer(2, 'Name', 'q_value');
    lgraph = addLayers(lgraph, advantageFusion);
    lgraph = addLayers(lgraph, advantageHead);
    lgraph = addLayers(lgraph, qAdd);
    lgraph = connectLayers(lgraph, stateFeatureName, 'adv_concat/in1');
    lgraph = connectLayers(lgraph, 'action_relu1', 'adv_concat/in2');
    lgraph = connectLayers(lgraph, 'adv_concat', 'adv_fc1');
    lgraph = connectLayers(lgraph, 'value_fc', 'q_value/in1');
    lgraph = connectLayers(lgraph, 'adv_fc', 'q_value/in2');

    criticNet = dlnetwork(lgraph);
    criticNet = initializeNetworkWeights(criticNet);
end

function layers = buildDroQHiddenStack(config, prefix)
    if nargin < 2
        prefix = '';
    end

    width = config.droqCriticWidth;
    depth = config.droqCriticDepth;
    dropoutProb = config.droqDropoutProbability;

    layers = [];
    for i = 1:depth
        layers = [layers
            fullyConnectedLayer(width, 'Name', sprintf('%sfc%d', prefix, i))
            dropoutLayer(dropoutProb, 'Name', sprintf('%sdrop%d', prefix, i))
            layerNormalizationLayer('Name', sprintf('%sln%d', prefix, i))
            reluLayer('Name', sprintf('%srelu%d', prefix, i))];
    end
end

function layers = buildFeatureBranch(prefix, startIdx, width)
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

function outputSize = getCriticOutputSize(config)
    if isfield(config, 'useTQCCritic') && config.useTQCCritic
        outputSize = config.tqcNumQuantiles;
    elseif isfield(config, 'useAQECritic') && config.useAQECritic
        outputSize = config.aqeHeadsPerCritic;
    else
        outputSize = 1;
    end
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
        branchWidths = [28, 28, 28, 16, 16];
    else
        branchWidths = repmat(24, 1, numBlocks);
    end

    blockStarts = 1 + (0:numBlocks-1) * blockSize;
end

function lgraph = addSimBaCriticTrunk(lgraph, config, inputName)
    width = config.simbaCriticWidth;
    numBlocks = config.simbaCriticBlocks;
    criticOutputSize = getCriticOutputSize(config);

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

    outputLayer = fullyConnectedLayer(criticOutputSize, 'Name', 'q_value');
    lgraph = addLayers(lgraph, outputLayer);
    lgraph = connectLayers(lgraph, previous, 'q_value');
end

function lgraph = addDroQCriticTrunk(lgraph, config, inputName)
    previous = inputName;
    width = config.droqCriticWidth;
    depth = config.droqCriticDepth;
    dropoutProb = config.droqDropoutProbability;
    criticOutputSize = getCriticOutputSize(config);

    for i = 1:depth
        blockPrefix = sprintf('trunk_block%d_', i);
        blockLayers = [
            fullyConnectedLayer(width, 'Name', [blockPrefix 'fc'])
            dropoutLayer(dropoutProb, 'Name', [blockPrefix 'drop'])
            layerNormalizationLayer('Name', [blockPrefix 'ln'])
            reluLayer('Name', [blockPrefix 'relu'])
        ];
        lgraph = addLayers(lgraph, blockLayers);
        lgraph = connectLayers(lgraph, previous, [blockPrefix 'fc']);
        previous = [blockPrefix 'relu'];
    end

    outputLayer = fullyConnectedLayer(criticOutputSize, 'Name', 'q_value');
    lgraph = addLayers(lgraph, outputLayer);
    lgraph = connectLayers(lgraph, previous, 'q_value');
end

function [lgraph, outputName] = addSimBaProjectionTrunk(lgraph, config, inputName, prefix)
    width = config.simbaCriticWidth;
    numBlocks = config.simbaCriticBlocks;

    stem = [
        fullyConnectedLayer(width, 'Name', [prefix 'proj'])
        reluLayer('Name', [prefix 'proj_relu'])
    ];
    lgraph = addLayers(lgraph, stem);
    lgraph = connectLayers(lgraph, inputName, [prefix 'proj']);
    previous = [prefix 'proj_relu'];

    for i = 1:numBlocks
        blockPrefix = sprintf('%sblock%d_', prefix, i);
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

    outputName = previous;
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

