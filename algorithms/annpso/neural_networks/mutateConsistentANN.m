function mutatedANN = mutateConsistentANN(originalANN)
    % Mutation exactly as specified in paper (Page 4, Section III.B):
    % "Mutation involves randomly changing the value of ONE of the ANN's weights"
    % PAPER: anni â†?mutate(annbest)

    mutatedANN = originalANN; % Copy structure

    % CRITICAL FIX: Mutate exactly ONE weight as specified in paper
    mutationStrength = 0.1;

    % Count total number of weights and biases
    numW1 = numel(originalANN.W1);
    numB1 = numel(originalANN.b1);
    numW2 = numel(originalANN.W2);
    numB2 = numel(originalANN.b2);
    totalWeights = numW1 + numB1 + numW2 + numB2;

    % Select ONE random weight to mutate
    selectedWeight = randi(totalWeights);

    % Determine which matrix/vector contains this weight and mutate it
    if selectedWeight <= numW1
        % Mutate one weight in W1
        mutatedANN.W1(selectedWeight) = originalANN.W1(selectedWeight) + mutationStrength * randn();
    elseif selectedWeight <= numW1 + numB1
        % Mutate one bias in b1
        idx = selectedWeight - numW1;
        mutatedANN.b1(idx) = originalANN.b1(idx) + mutationStrength * randn();
    elseif selectedWeight <= numW1 + numB1 + numW2
        % Mutate one weight in W2
        idx = selectedWeight - numW1 - numB1;
        mutatedANN.W2(idx) = originalANN.W2(idx) + mutationStrength * randn();
    else
        % Mutate one bias in b2
        idx = selectedWeight - numW1 - numB1 - numW2;
        mutatedANN.b2(idx) = originalANN.b2(idx) + mutationStrength * randn();
    end

    % PAPER (Page 4): "whilst keeping its accumulated success value"
    % When adopting annbest and mutating, preserve the success value
    mutatedANN.success = originalANN.success;
end

%% 11. IGPSO (Importance-Guided PSO)

