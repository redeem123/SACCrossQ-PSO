function algorithmSpecificStats = addFitnessComponents(algorithmSpecificStats, globalBestFitness, actualComponents)
    % Add individual fitness components to algorithm statistics
    % Updated to support new fitness function from paper s41598-025-85912-4
    %
    % Supports both old and new component structures for backward compatibility

    algorithmSpecificStats.fitnessComponents = struct();

    % Check if using new fitness function (paper-based) or old
    if isfield(actualComponents, 'F1_pathCost')
        % ===== NEW FITNESS FUNCTION (Paper s41598-025-85912-4) =====
        % F_fit = ω1*F1 + ω2*F2 + ω3*F3 + ω4*F4

        % Store raw components (unweighted)
        algorithmSpecificStats.fitnessComponents.F1_pathCost = actualComponents.F1_pathCost;
        algorithmSpecificStats.fitnessComponents.F2_safetyCost = actualComponents.F2_safetyCost;
        algorithmSpecificStats.fitnessComponents.F3_heightVariation = actualComponents.F3_heightVariation;
        algorithmSpecificStats.fitnessComponents.F4_angleVariation = actualComponents.F4_angleVariation;

        % Store weighted components (actual contribution to fitness)
        algorithmSpecificStats.fitnessComponents.F1_weighted = actualComponents.F1_weighted;
        algorithmSpecificStats.fitnessComponents.F2_weighted = actualComponents.F2_weighted;
        algorithmSpecificStats.fitnessComponents.F3_weighted = actualComponents.F3_weighted;
        algorithmSpecificStats.fitnessComponents.F4_weighted = actualComponents.F4_weighted;

        % Store weights for reference
        if isfield(actualComponents, 'weights')
            algorithmSpecificStats.fitnessComponents.weights = actualComponents.weights;
        end

        % Duplicate penalty (not in paper, but practical)
        if isfield(actualComponents, 'duplicatePenalty')
            algorithmSpecificStats.fitnessComponents.duplicatePenalty = actualComponents.duplicatePenalty;
        else
            algorithmSpecificStats.fitnessComponents.duplicatePenalty = 0;
        end

    else
        % ===== OLD FITNESS FUNCTION (Legacy) =====
        % For backward compatibility with old code

        algorithmSpecificStats.fitnessComponents.pathLength = getFieldOrDefault(actualComponents, 'pathLength', 0);
        algorithmSpecificStats.fitnessComponents.turningPenalty = getFieldOrDefault(actualComponents, 'turningPenalty', 0);
        algorithmSpecificStats.fitnessComponents.climbingPenalty = getFieldOrDefault(actualComponents, 'climbingPenalty', 0);
        algorithmSpecificStats.fitnessComponents.heightPenalty = getFieldOrDefault(actualComponents, 'heightPenalty', 0);
        algorithmSpecificStats.fitnessComponents.collisionPenalty = getFieldOrDefault(actualComponents, 'collisionPenalty', 0);
        algorithmSpecificStats.fitnessComponents.terrainPenalty = getFieldOrDefault(actualComponents, 'terrainPenalty', 0);
        algorithmSpecificStats.fitnessComponents.dangerZonePenalty = getFieldOrDefault(actualComponents, 'dangerZonePenalty', 0);
        algorithmSpecificStats.fitnessComponents.duplicatePenalty = getFieldOrDefault(actualComponents, 'duplicatePenalty', 0);
    end

    % Total fitness (common to both)
    algorithmSpecificStats.fitnessComponents.totalFitness = globalBestFitness;
end

function value = getFieldOrDefault(structure, fieldName, defaultValue)
    % Helper function to safely get field or return default
    if isfield(structure, fieldName)
        value = structure.(fieldName);
    else
        value = defaultValue;
    end
end

