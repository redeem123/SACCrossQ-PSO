function output = forwardPassAblatedNetwork(network, input)
    % Forward pass through ablated network architecture
    
    % Ensure input is column vector
    input = input(:);
    
    if network.numLayers == 1
        % Linear network
        output = network.W1 * input + network.b1;
        
    elseif network.numLayers == 2
        % Shallow network
        z1 = network.W1 * input + network.b1;
        z1 = max(-50, min(50, z1));  % Prevent extreme values
        h1 = max(0, z1);  % ReLU
        output = network.W2 * h1 + network.b2;
        
    else
        % Deep network (3 layers)
        z1 = network.W1 * input + network.b1;
        z1 = max(-50, min(50, z1));  % Prevent extreme values
        h1 = max(0, z1);  % ReLU
        
        z2 = network.W2 * h1 + network.b2;
        z2 = max(-50, min(50, z2));  % Prevent extreme values
        h2 = max(0, z2);  % ReLU
        
        output = network.W3 * h2 + network.b3;
    end
    
    output(~isfinite(output)) = 0;
    output = max(-20, min(20, output));
end

