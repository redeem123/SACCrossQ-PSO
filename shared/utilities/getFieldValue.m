function value = getFieldValue(structure, fieldName, defaultValue)
    if isfield(structure, fieldName) && ~isempty(structure.(fieldName))
        value = structure.(fieldName);
    else
        value = defaultValue;
    end
end

%% Part 2. ABLATION STUDY

%% ABLATION STUDY FUNCTIONS

