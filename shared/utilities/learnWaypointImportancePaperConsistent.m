function importanceVector = learnWaypointImportancePaperConsistent(startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY, candidateWaypoints, hiddenNeurons, positiveSampleRatio)
    % Learn waypoint importance using two-stage focal neural network
    % Implements Algorithm 1 from the IGPSO paper EXACTLY:
    %
    % Paper Algorithm 1: two-stage training
    % Input: Dataset X_Pos and Dataset X_Neg
    % Output: Importance vector ip
    %
    % 1: Take X_Pos as input for FM_pos to generate A*_Pos
    % 2: Save parameters of FM_pos as Para
    % 3: FM_neg loads Para (MLP weights transferred)
    % 4: Every element of A in FM_neg is initialized to 1 (attention reset)
    % 5: Take X_Neg as input for FM_neg to generate A*_neg
    % 6: Subtract A*_Neg from A*_Pos to get A*_f
    % 7: Use A*_f to calculate ip as Eq. (6)
    % 8: Return ip
    
    numCandidates = size(candidateWaypoints, 1);
    numSamples = 200;
    
    % Generate training samples by creating random waypoint selections
    allSamples = zeros(numSamples, numCandidates);
    allLabels = zeros(numSamples, 1);
    
    fprintf('    Generating %d training samples for waypoint importance...\n', numSamples);
    
    for i = 1:numSamples
        % Random binary selection of waypoints
        sample = rand(1, numCandidates) > 0.7; % ~30% waypoints selected on average
        allSamples(i, :) = double(sample);
        
        % Evaluate path quality
        fitness = evaluateWaypointSelectionFitness(sample, candidateWaypoints, startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY);
        
        % Binary label: 1 for good paths (low fitness), 0 for bad paths
        allLabels(i) = fitness < 800; % Threshold for good vs bad paths
    end
    
    % Split into positive and negative samples as per paper methodology
    positiveIndices = find(allLabels == 1);
    negativeIndices = find(allLabels == 0);
    
    if length(positiveIndices) < 5
        % If too few positive samples, create some manually
        fprintf('    Creating additional positive samples...\n');
        additionalPositive = createGoodWaypointSelections(candidateWaypoints, startPoint, goalPoint, 10);
        allSamples = [allSamples; additionalPositive];
        allLabels = [allLabels; ones(size(additionalPositive, 1), 1)];
        positiveIndices = find(allLabels == 1);
    end
    
    positiveSamples = allSamples(positiveIndices, :);
    negativeSamples = allSamples(negativeIndices, :);
    
    fprintf('    Positive samples: %d, Negative samples: %d\n', size(positiveSamples, 1), size(negativeSamples, 1));
    
    % Two-stage training EXACTLY as in paper Algorithm 1

    % Stage 1: Train FM_pos on positive samples to learn "good waypoint patterns"
    fprintf('    Stage 1: Training FM_pos on positive samples...\n');
    if size(positiveSamples, 1) > 3
        FM_pos = createFocalNetwork(numCandidates, hiddenNeurons);
        FM_pos = trainFocalNetwork(FM_pos, positiveSamples, ones(size(positiveSamples, 1), 1), 50);

        % Extract A*_pos (attention vector after positive training)
        A_star_pos = 1 ./ (1 + exp(-FM_pos.attention)); % Apply sigmoid (Equation 4)
    else
        A_star_pos = ones(numCandidates, 1); % Default if insufficient samples
    end

    % Stage 2: Paper Algorithm 1 - Transfer MLP parameters, re-initialize attention
    fprintf('    Stage 2: Transferring MLP parameters to FM_neg and training on negative samples...\n');
    if size(negativeSamples, 1) > 3
        % Create FM_neg and inherit MLP parameters from FM_pos (Algorithm 1, line 2-3)
        FM_neg = struct();
        FM_neg.W1 = FM_pos.W1; % Transfer MLP weights
        FM_neg.b1 = FM_pos.b1;
        FM_neg.W2 = FM_pos.W2;
        FM_neg.b2 = FM_pos.b2;
        FM_neg.W3 = FM_pos.W3;
        FM_neg.b3 = FM_pos.b3;

        % Re-initialize attention vector (Algorithm 1, line 4)
        FM_neg.attention = ones(numCandidates, 1);

        % Train on negative samples (Algorithm 1, line 5)
        FM_neg = trainFocalNetwork(FM_neg, negativeSamples, zeros(size(negativeSamples, 1), 1), 50);

        % Extract A*_neg (attention vector after negative training)
        A_star_neg = 1 ./ (1 + exp(-FM_neg.attention)); % Apply sigmoid (Equation 4)
    else
        A_star_neg = zeros(numCandidates, 1); % Default if insufficient samples
    end

    % Combine attention vectors by SUBTRACTION as per paper (Algorithm 1, line 6)
    % A*_f = A*_pos - A*_neg
    A_star_f = A_star_pos - A_star_neg;

    % Scale to [0.1, 0.9] range as per paper Equation 6
    a_star_min = min(A_star_f);
    a_star_max = max(A_star_f);

    if a_star_max - a_star_min > 1e-6 % Avoid division by zero
        importanceVector = (A_star_f - a_star_min) / (a_star_max - a_star_min);
    else
        importanceVector = ones(numCandidates, 1) * 0.5; % Uniform if no variation
    end

    importanceVector = importanceVector * 0.8 + 0.1; % Scale to [0.1, 0.9] (Equation 6)
    
    fprintf('    Waypoint importance learned. Range: [%.3f, %.3f]\n', min(importanceVector), max(importanceVector));
end

