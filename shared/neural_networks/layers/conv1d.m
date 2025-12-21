function output = conv1d(input, weights, bias)
    % Simple 1D convolution implementation
    % input: [inputChannels, sequenceLength]
    % weights: [kernelSize, inputChannels, outputChannels]
    % bias: [outputChannels, 1]
    
    [inputChannels, sequenceLength] = size(input);
    [kernelSize, ~, outputChannels] = size(weights);
    outputLength = sequenceLength - kernelSize + 1;
    
    output = zeros(outputChannels, outputLength);
    
    for outChan = 1:outputChannels
        for pos = 1:outputLength
            conv_sum = 0;
            for inChan = 1:inputChannels
                for k = 1:kernelSize
                    conv_sum = conv_sum + input(inChan, pos + k - 1) * weights(k, inChan, outChan);
                end
            end
            output(outChan, pos) = conv_sum + bias(outChan);
        end
    end
end

