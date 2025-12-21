function performTTest(allRunResults, algorithms, scenarioIdx)
    % Perform pairwise T-tests using global path planning fitness
    
    fprintf('\n=== PAIRWISE T-TEST ANALYSIS FOR SCENARIO %d ===\n', scenarioIdx);
    
    numRuns = length(allRunResults);
    numAlgorithms = length(algorithms);
    
    % Extract global path fitness matrix: [runs Ã— algorithms]
    globalPathFitness = zeros(numRuns, numAlgorithms);
    algorithmNames = cell(numAlgorithms, 1);
    
    for run = 1:numRuns
        for alg = 1:numAlgorithms
            algName = algorithms{alg}.fieldName;
            algorithmNames{alg} = algorithms{alg}.displayName;
            
            % Use the stored fitness from the algorithm results
            if isfield(allRunResults{run}, algName) && isfield(allRunResults{run}.(algName), 'pathLength')
                % Get the actual fitness value that was optimized during global planning
                globalPathFitness(run, alg) = allRunResults{run}.(algName).pathLength;
            else
                globalPathFitness(run, alg) = NaN;
            end
        end
    end
    
    % Remove any runs with NaN values
    validRuns = ~any(isnan(globalPathFitness), 2);
    globalPathFitness = globalPathFitness(validRuns, :);
    validRunCount = sum(validRuns);
    
    fprintf('Valid runs for analysis: %d/%d\n', validRunCount, numRuns);
    
    % ===== DESCRIPTIVE STATISTICS =====
    fprintf('\n=== DESCRIPTIVE STATISTICS (Global Path Fitness) ===\n');
    fprintf('%-25s | %-10s | %-10s | %-10s\n', 'Algorithm', 'Mean', 'Std', 'Median');
    fprintf('---------------------------------------------------------------\n');
    
    means = zeros(numAlgorithms, 1);
    stds = zeros(numAlgorithms, 1);
    medians = zeros(numAlgorithms, 1);
    
    for i = 1:numAlgorithms
        values = globalPathFitness(:, i);
        means(i) = mean(values);
        stds(i) = std(values);
        medians(i) = median(values);
        
        fprintf('%-25s | %10.3f | %10.3f | %10.3f\n', ...
            algorithmNames{i}, means(i), stds(i), medians(i));
    end
    
    % ===== PAIRWISE T-TESTS =====
    fprintf('\n=== PAIRWISE T-TEST RESULTS (Global Path Planning Performance) ===\n');
    fprintf('Using Paired T-Tests (same scenarios across algorithms)\n\n');
    
    % Create results matrix for p-values and t-statistics
    pValueMatrix = ones(numAlgorithms, numAlgorithms);
    tStatMatrix = zeros(numAlgorithms, numAlgorithms);
    significantPairs = {};
    
    fprintf('%-25s vs %-25s | t-stat | df | p-value | Significance\n', 'Algorithm 1', 'Algorithm 2');
    fprintf('--------------------------------------------------------------------------------\n');
    
    % Perform pairwise comparisons
    for i = 1:numAlgorithms
        for j = i+1:numAlgorithms
            sample1 = globalPathFitness(:, i);
            sample2 = globalPathFitness(:, j);
            
            % Perform paired t-test (since same runs/scenarios)
            [tStat, pValue, df] = pairedTTest(sample1, sample2);
            
            pValueMatrix(i, j) = pValue;
            pValueMatrix(j, i) = pValue;  % Symmetric matrix
            tStatMatrix(i, j) = tStat;
            tStatMatrix(j, i) = -tStat;   % Antisymmetric for t-stats
            
            % Determine significance
            significance = '';
            isSignificant = false;
            if pValue < 0.001
                significance = '*** (p < 0.001)';
                isSignificant = true;
            elseif pValue < 0.01
                significance = '** (p < 0.01)';
                isSignificant = true;
            elseif pValue < 0.05
                significance = '* (p < 0.05)';
                isSignificant = true;
            else
                significance = '(not significant)';
            end
            
            % Store significant pairs
            if isSignificant
                if means(i) < means(j)
                    winner = algorithmNames{i};
                    loser = algorithmNames{j};
                else
                    winner = algorithmNames{j};
                    loser = algorithmNames{i};
                end
                significantPairs{end+1} = struct('winner', winner, 'loser', loser, 'pValue', pValue, 'tStat', abs(tStat));
            end
            
            fprintf('%-25s vs %-25s | %6.3f | %2d | %7.5f | %s\n', ...
                algorithmNames{i}, algorithmNames{j}, tStat, df, pValue, significance);
        end
    end
    
    % ===== SUMMARY OF SIGNIFICANT DIFFERENCES =====
    fprintf('\n=== SIGNIFICANT DIFFERENCES SUMMARY ===\n');
    if ~isempty(significantPairs)
        fprintf('Found %d significant pairwise differences:\n\n', length(significantPairs));
        
        % Sort by p-value (most significant first)
        pValues = cellfun(@(x) x.pValue, significantPairs);
        [~, sortIdx] = sort(pValues);
        
        fprintf('%-25s | %-25s | t-stat | p-value\n', 'Better Algorithm', 'Worse Algorithm');
        fprintf('------------------------------------------------------------------------\n');
        
        for i = 1:length(sortIdx)
            pair = significantPairs{sortIdx(i)};
            fprintf('%-25s | %-25s | %6.3f | %7.5f %s\n', ...
                pair.winner, pair.loser, pair.tStat, pair.pValue, significance_star(pair.pValue));
        end
        
        % Find best overall algorithm (lowest mean with most significant wins)
        fprintf('\n=== OVERALL RANKING ===\n');
        [sortedMeans, rankIdx] = sort(means);
        
        fprintf('Rank | %-25s | Mean Fitness | Significant Wins\n', 'Algorithm');
        fprintf('--------------------------------------------------------------\n');
        
        for i = 1:numAlgorithms
            algIdx = rankIdx(i);
            
            % Count significant wins for this algorithm
            wins = 0;
            for pair = significantPairs
                if strcmp(pair{1}.winner, algorithmNames{algIdx})
                    wins = wins + 1;
                end
            end
            
            fprintf('%-4d | %-25s | %10.3f | %5d\n', ...
                i, algorithmNames{algIdx}, sortedMeans(i), wins);
        end
        
        % Best and worst algorithms
        bestAlg = algorithmNames{rankIdx(1)};
        worstAlg = algorithmNames{rankIdx(end)};
        
        fprintf('\nï¿½?ï¿½ BEST ALGORITHM:  %s (Mean: %.3f)\n', bestAlg, sortedMeans(1));
        fprintf('ðŸ”» WORST ALGORITHM: %s (Mean: %.3f)\n', worstAlg, sortedMeans(end));
        
    else
        fprintf('No significant differences found between algorithms (all p-values >= 0.05)\n');
        fprintf('All algorithms show similar performance on global path planning.\n');
    end
    
    % ===== BONFERRONI CORRECTION =====
    fprintf('\n=== BONFERRONI CORRECTION FOR MULTIPLE COMPARISONS ===\n');
    numComparisons = numAlgorithms * (numAlgorithms - 1) / 2;
    bonferroniAlpha = 0.05 / numComparisons;
    fprintf('Number of pairwise comparisons: %d\n', numComparisons);
    fprintf('Bonferroni-corrected alpha level: %.6f\n', bonferroniAlpha);
    
    % Count significant differences after Bonferroni correction
    bonferroniSignificant = 0;
    for i = 1:length(significantPairs)
        if significantPairs{i}.pValue < bonferroniAlpha
            bonferroniSignificant = bonferroniSignificant + 1;
        end
    end
    
    fprintf('Significant differences after Bonferroni correction: %d/%d\n', ...
        bonferroniSignificant, length(significantPairs));
    
    % ===== EFFECT SIZE ANALYSIS =====
    fprintf('\n=== EFFECT SIZE ANALYSIS (Cohen''s d) ===\n');
    fprintf('%-25s vs %-25s | Cohen''s d | Effect Size\n', 'Algorithm 1', 'Algorithm 2');
    fprintf('----------------------------------------------------------------\n');
    
    for i = 1:numAlgorithms
        for j = i+1:numAlgorithms
            if pValueMatrix(i, j) < 0.05  % Only show for significant differences
                sample1 = globalPathFitness(:, i);
                sample2 = globalPathFitness(:, j);
                
                cohensD = calculateCohensD(sample1, sample2);
                effectSize = interpretEffectSize(abs(cohensD));
                
                fprintf('%-25s vs %-25s | %8.3f | %s\n', ...
                    algorithmNames{i}, algorithmNames{j}, cohensD, effectSize);
            end
        end
    end
    
    % ===== SAVE RESULTS =====
    saveTTestResults(globalPathFitness, algorithmNames, pValueMatrix, tStatMatrix, ...
                    scenarioIdx);
end

