function selectedInformantIdx = selectGbestWithANN(ann, personalBest, informants)
    % Select gbest using ANN exactly as in PAPER ALGORITHM 2, lines 19-24
    % PAPER: gbest i â†? pbest of first informant
    % PAPER: for each subsequent informant j do
    % PAPER:     if anni(pbest j) > anni(gbest i) then gbest i â†? pbestj
    
    if isempty(informants)
        selectedInformantIdx = 1; % Fallback
        return;
    end
    
    % Start with first informant (PAPER: gbest i â†? pbest of first informant)
    selectedInformantIdx = informants(1);
    bestResponse = forwardPassConsistentANN(ann, personalBest(selectedInformantIdx, :)');
    
    % Compare with subsequent informants (PAPER lines 20-24)
    for i = 2:length(informants)
        informantIdx = informants(i);
        informantResponse = forwardPassConsistentANN(ann, personalBest(informantIdx, :)');
        
        % PAPER: if anni(pbest j) > anni(gbest i) then gbest i â†? pbestj
        if informantResponse > bestResponse
            bestResponse = informantResponse;
            selectedInformantIdx = informantIdx;
        end
    end
end

