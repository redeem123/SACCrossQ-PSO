function value = getFieldValue(structure, fieldName, defaultValue)
    if isfield(structure, fieldName) && ~isempty(structure.(fieldName))
        value = structure.(fieldName);
    else
        value = defaultValue;
    end
end
