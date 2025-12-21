function visualize_training_comparison(baselinePath, noCrossQPath, outputDir)
    % VISUALIZE_TRAINING_COMPARISON - Compare training curves: CrossQ-SAC vs Standard SAC
    %
    % Generates comprehensive comparison plots for paper:
    %   1. Episode rewards (learning progress)
    %   2. Best fitness convergence
    %   3. Sample efficiency
    %   4. Critic loss comparison
    %   5. Actor loss comparison
    %   6. Alpha (entropy coefficient) evolution
    %   7. Q-value stability
    %   8. Policy entropy evolution
    %   9. Training time comparison
    %
    % Usage:
    %   visualize_training_comparison(...
    %       'models/training_logs/apexpso_perparticle_log.mat', ...
    %       'models/training_logs/apexpso_abl_nocrossq_log.mat', ...
    %       'outputs/results/training_comparison');
    %
    % Inputs:
    %   baselinePath: Path to CrossQ-SAC training log
    %   noCrossQPath: Path to Standard SAC training log
    %   outputDir: Directory to save output figures

    if nargin < 3
        outputDir = 'outputs/results/training_comparison';
    end

    % Create output directory
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║          Training Comparison: CrossQ vs Standard SAC    ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

    % Load training logs
    fprintf('📂 Loading training logs...\n');
    baseline = load(baselinePath);
    noCrossQ = load(noCrossQPath);

    baselineData = baseline.logData;
    noCrossQData = noCrossQ.logData;

    fprintf('   ✓ CrossQ-SAC (Baseline): %d episodes\n', length(baselineData.episodeRewards));
    fprintf('   ✓ Standard SAC (No CrossQ): %d episodes\n', length(noCrossQData.episodeRewards));

    % Smooth curves for visualization (moving average)
    windowSize = 5;

    %% Figure 1: Episode Rewards (Learning Progress)
    fprintf('\n📊 Generating Figure 1: Episode Rewards...\n');
    fig1 = figure('Position', [100, 100, 1200, 400]);

    subplot(1, 3, 1);
    plot(smooth(baselineData.episodeRewards, windowSize), 'b-', 'LineWidth', 2);
    hold on;
    plot(smooth(noCrossQData.episodeRewards, windowSize), 'r--', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Cumulative Reward');
    title('Episode Reward Comparison');
    legend({'CrossQ-SAC (Baseline)', 'Standard SAC'}, 'Location', 'best');
    grid on;

    % Reward improvement rate
    subplot(1, 3, 2);
    baselineImprovement = diff(smooth(baselineData.episodeRewards, windowSize));
    noCrossQImprovement = diff(smooth(noCrossQData.episodeRewards, windowSize));
    plot(baselineImprovement, 'b-', 'LineWidth', 2);
    hold on;
    plot(noCrossQImprovement, 'r--', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Reward Improvement');
    title('Learning Rate (Reward Delta)');
    legend({'CrossQ-SAC', 'Standard SAC'}, 'Location', 'best');
    grid on;

    % Cumulative reward advantage
    subplot(1, 3, 3);
    cumulativeBaseline = cumsum(baselineData.episodeRewards);
    cumulativeNoCrossQ = cumsum(noCrossQData.episodeRewards);
    plot(cumulativeBaseline, 'b-', 'LineWidth', 2);
    hold on;
    plot(cumulativeNoCrossQ, 'r--', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Cumulative Sum');
    title('Total Reward Accumulated');
    legend({'CrossQ-SAC', 'Standard SAC'}, 'Location', 'best');
    grid on;

    saveas(fig1, fullfile(outputDir, 'Training_EpisodeRewards_Comparison.pdf'));
    saveas(fig1, fullfile(outputDir, 'Training_EpisodeRewards_Comparison.png'));
    fprintf('   ✓ Saved: Training_EpisodeRewards_Comparison.pdf\n');

    %% Figure 2: Best Fitness Convergence
    fprintf('📊 Generating Figure 2: Fitness Convergence...\n');
    fig2 = figure('Position', [100, 100, 800, 600]);

    % Best fitness over episodes
    baselineBestSoFar = cummin(baselineData.episodeFitness);
    noCrossQBestSoFar = cummin(noCrossQData.episodeFitness);

    plot(baselineBestSoFar, 'b-', 'LineWidth', 2.5);
    hold on;
    plot(noCrossQBestSoFar, 'r--', 'LineWidth', 2.5);
    xlabel('Episode', 'FontSize', 12);
    ylabel('Best Fitness (Lower is Better)', 'FontSize', 12);
    title('Training Convergence: Best Path Quality', 'FontSize', 14);
    legend({'CrossQ-SAC (Baseline)', 'Standard SAC (No CrossQ)'}, ...
        'Location', 'northeast', 'FontSize', 11);
    grid on;

    % Add final fitness annotations
    text(length(baselineBestSoFar), baselineBestSoFar(end), ...
        sprintf('  Final: %.1f', baselineBestSoFar(end)), ...
        'Color', 'b', 'FontSize', 10, 'VerticalAlignment', 'bottom');
    text(length(noCrossQBestSoFar), noCrossQBestSoFar(end), ...
        sprintf('  Final: %.1f', noCrossQBestSoFar(end)), ...
        'Color', 'r', 'FontSize', 10, 'VerticalAlignment', 'top');

    saveas(fig2, fullfile(outputDir, 'Training_FitnessConvergence_Comparison.pdf'));
    saveas(fig2, fullfile(outputDir, 'Training_FitnessConvergence_Comparison.png'));
    fprintf('   ✓ Saved: Training_FitnessConvergence_Comparison.pdf\n');

    %% Figure 3: Sample Efficiency
    fprintf('📊 Generating Figure 3: Sample Efficiency...\n');
    fig3 = figure('Position', [100, 100, 1200, 400]);

    subplot(1, 2, 1);
    plot(baselineData.totalSamples, baselineData.episodeRewards, 'b-', 'LineWidth', 2);
    hold on;
    plot(noCrossQData.totalSamples, noCrossQData.episodeRewards, 'r--', 'LineWidth', 2);
    xlabel('Total Environment Samples');
    ylabel('Cumulative Reward');
    title('Sample Efficiency: Reward vs Samples');
    legend({'CrossQ-SAC', 'Standard SAC'}, 'Location', 'best');
    grid on;

    subplot(1, 2, 2);
    plot(smooth(baselineData.rewardPerSample, windowSize), 'b-', 'LineWidth', 2);
    hold on;
    plot(smooth(noCrossQData.rewardPerSample, windowSize), 'r--', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Reward / Sample');
    title('Instantaneous Sample Efficiency');
    legend({'CrossQ-SAC', 'Standard SAC'}, 'Location', 'best');
    grid on;

    saveas(fig3, fullfile(outputDir, 'Training_SampleEfficiency_Comparison.pdf'));
    saveas(fig3, fullfile(outputDir, 'Training_SampleEfficiency_Comparison.png'));
    fprintf('   ✓ Saved: Training_SampleEfficiency_Comparison.pdf\n');

    %% Figure 4: Loss Comparison (Actor + Critics)
    fprintf('📊 Generating Figure 4: Loss Comparison...\n');
    fig4 = figure('Position', [100, 100, 1200, 800]);

    % Actor loss
    subplot(2, 2, 1);
    plot(smooth(baselineData.actorLoss, windowSize), 'b-', 'LineWidth', 2);
    hold on;
    plot(smooth(noCrossQData.actorLoss, windowSize), 'r--', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Loss');
    title('Actor Loss');
    legend({'CrossQ-SAC', 'Standard SAC'}, 'Location', 'best');
    grid on;

    % Critic 1 loss
    subplot(2, 2, 2);
    plot(smooth(baselineData.critic1Loss, windowSize), 'b-', 'LineWidth', 2);
    hold on;
    plot(smooth(noCrossQData.critic1Loss, windowSize), 'r--', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Loss');
    title('Critic 1 Loss');
    legend({'CrossQ-SAC', 'Standard SAC'}, 'Location', 'best');
    grid on;

    % Critic 2 loss
    subplot(2, 2, 3);
    plot(smooth(baselineData.critic2Loss, windowSize), 'b-', 'LineWidth', 2);
    hold on;
    plot(smooth(noCrossQData.critic2Loss, windowSize), 'r--', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Loss');
    title('Critic 2 Loss');
    legend({'CrossQ-SAC', 'Standard SAC'}, 'Location', 'best');
    grid on;

    % Combined critic loss
    subplot(2, 2, 4);
    baselineMeanCritic = (baselineData.critic1Loss + baselineData.critic2Loss) / 2;
    noCrossQMeanCritic = (noCrossQData.critic1Loss + noCrossQData.critic2Loss) / 2;
    plot(smooth(baselineMeanCritic, windowSize), 'b-', 'LineWidth', 2);
    hold on;
    plot(smooth(noCrossQMeanCritic, windowSize), 'r--', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Loss');
    title('Mean Critic Loss (Both Critics)');
    legend({'CrossQ-SAC', 'Standard SAC'}, 'Location', 'best');
    grid on;

    saveas(fig4, fullfile(outputDir, 'Training_Losses_Comparison.pdf'));
    saveas(fig4, fullfile(outputDir, 'Training_Losses_Comparison.png'));
    fprintf('   ✓ Saved: Training_Losses_Comparison.pdf\n');

    %% Figure 5: Alpha and Entropy Evolution
    fprintf('📊 Generating Figure 5: Alpha and Entropy...\n');
    fig5 = figure('Position', [100, 100, 1200, 400]);

    % Alpha (entropy coefficient)
    subplot(1, 2, 1);
    plot(smooth(baselineData.alphaValue, windowSize), 'b-', 'LineWidth', 2);
    hold on;
    plot(smooth(noCrossQData.alphaValue, windowSize), 'r--', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Alpha Value');
    title('Entropy Coefficient (Alpha) Evolution');
    legend({'CrossQ-SAC', 'Standard SAC'}, 'Location', 'best');
    grid on;

    % Policy entropy
    subplot(1, 2, 2);
    plot(smooth(baselineData.policyEntropy, windowSize), 'b-', 'LineWidth', 2);
    hold on;
    plot(smooth(noCrossQData.policyEntropy, windowSize), 'r--', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Policy Entropy');
    title('Policy Exploration (Entropy)');
    legend({'CrossQ-SAC', 'Standard SAC'}, 'Location', 'best');
    grid on;

    saveas(fig5, fullfile(outputDir, 'Training_AlphaEntropy_Comparison.pdf'));
    saveas(fig5, fullfile(outputDir, 'Training_AlphaEntropy_Comparison.png'));
    fprintf('   ✓ Saved: Training_AlphaEntropy_Comparison.pdf\n');

    %% Figure 6: Q-Value Stability
    fprintf('📊 Generating Figure 6: Q-Value Stability...\n');
    fig6 = figure('Position', [100, 100, 1200, 400]);

    % Mean Q-values
    subplot(1, 2, 1);
    plot(smooth(baselineData.meanQValue, windowSize), 'b-', 'LineWidth', 2);
    hold on;
    plot(smooth(noCrossQData.meanQValue, windowSize), 'r--', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Mean Q-Value');
    title('Q-Value Estimates (Mean)');
    legend({'CrossQ-SAC', 'Standard SAC'}, 'Location', 'best');
    grid on;

    % Max Q-values
    subplot(1, 2, 2);
    plot(smooth(baselineData.maxQValue, windowSize), 'b-', 'LineWidth', 2);
    hold on;
    plot(smooth(noCrossQData.maxQValue, windowSize), 'r--', 'LineWidth', 2);
    xlabel('Episode');
    ylabel('Max Q-Value');
    title('Q-Value Estimates (Maximum)');
    legend({'CrossQ-SAC', 'Standard SAC'}, 'Location', 'best');
    grid on;

    saveas(fig6, fullfile(outputDir, 'Training_QValues_Comparison.pdf'));
    saveas(fig6, fullfile(outputDir, 'Training_QValues_Comparison.png'));
    fprintf('   ✓ Saved: Training_QValues_Comparison.pdf\n');

    %% Figure 7: Training Time Efficiency
    fprintf('📊 Generating Figure 7: Training Time...\n');
    fig7 = figure('Position', [100, 100, 800, 600]);

    bar([mean(baselineData.episodeDuration), mean(noCrossQData.episodeDuration)]);
    xticklabels({'CrossQ-SAC', 'Standard SAC'});
    ylabel('Mean Episode Duration (seconds)');
    title('Training Time per Episode');
    grid on;

    % Add value labels on bars
    text(1, mean(baselineData.episodeDuration), ...
        sprintf('%.2f s', mean(baselineData.episodeDuration)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    text(2, mean(noCrossQData.episodeDuration), ...
        sprintf('%.2f s', mean(noCrossQData.episodeDuration)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

    saveas(fig7, fullfile(outputDir, 'Training_Time_Comparison.pdf'));
    saveas(fig7, fullfile(outputDir, 'Training_Time_Comparison.png'));
    fprintf('   ✓ Saved: Training_Time_Comparison.pdf\n');

    %% Summary Statistics Table
    fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║                  Training Comparison Summary             ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

    fprintf('                      CrossQ-SAC  |  Standard SAC  |  Improvement\n');
    fprintf('─────────────────────────────────────────────────────────────────\n');

    finalRewardBaseline = baselineData.episodeRewards(end);
    finalRewardNoCrossQ = noCrossQData.episodeRewards(end);
    rewardImprovement = ((finalRewardBaseline - finalRewardNoCrossQ) / abs(finalRewardNoCrossQ)) * 100;

    fprintf('Final Reward:        %9.2f  |  %9.2f  |  %+.1f%%\n', ...
        finalRewardBaseline, finalRewardNoCrossQ, rewardImprovement);

    finalFitnessBaseline = baselineBestSoFar(end);
    finalFitnessNoCrossQ = noCrossQBestSoFar(end);
    fitnessImprovement = ((finalFitnessNoCrossQ - finalFitnessBaseline) / finalFitnessNoCrossQ) * 100;

    fprintf('Best Fitness:        %9.2f  |  %9.2f  |  %+.1f%%\n', ...
        finalFitnessBaseline, finalFitnessNoCrossQ, fitnessImprovement);

    meanTimeBaseline = mean(baselineData.episodeDuration);
    meanTimeNoCrossQ = mean(noCrossQData.episodeDuration);
    timeOverhead = ((meanTimeBaseline - meanTimeNoCrossQ) / meanTimeNoCrossQ) * 100;

    fprintf('Episode Time (s):    %9.2f  |  %9.2f  |  %+.1f%%\n', ...
        meanTimeBaseline, meanTimeNoCrossQ, timeOverhead);

    totalSamplesBaseline = baselineData.totalSamples(end);
    totalSamplesNoCrossQ = noCrossQData.totalSamples(end);

    fprintf('Total Samples:       %9d  |  %9d  |  %+.1f%%\n', ...
        totalSamplesBaseline, totalSamplesNoCrossQ, ...
        ((totalSamplesBaseline - totalSamplesNoCrossQ) / totalSamplesNoCrossQ) * 100);

    fprintf('\n══════════════════════════════════════════════════════════════\n');
    fprintf('\n✓ All training comparison figures saved to: %s\n', outputDir);

    close all;
end

function smoothed = smooth(data, windowSize)
    % Simple moving average smoothing
    if length(data) < windowSize
        smoothed = data;
        return;
    end
    smoothed = movmean(data, windowSize);
end
