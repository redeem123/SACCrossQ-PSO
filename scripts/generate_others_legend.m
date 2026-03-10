function generate_others_legend()
    customColors = [
        1.0, 0.0, 0.0;
        0.0, 0.0, 1.0;
        0.0, 0.8, 0.0;
        1.0, 0.5, 0.0;
        0.5, 0.0, 0.5;
        0.0, 0.8, 0.8;
        1.0, 0.0, 1.0;
        0.8, 0.8, 0.0
    ];
    legendLabels = {'RRSACPSO', 'PSO-Standard', 'PSO-LDIW', 'FIPS', 'CLPSO', 'SAEPSO [11]', 'SAEPSO* [11]', 'UAPSO [25]'};
    fig = figure('Visible', 'off');
    hold on;
    h = gobjects(1, length(legendLabels));
    for i = 1:length(legendLabels)
        h(i) = plot([nan nan], [nan nan], '-', 'Color', customColors(mod(i-1, size(customColors, 1)) + 1, :), 'LineWidth', 3);
    end
    start_h = plot(nan, nan, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    goal_h = plot(nan, nan, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    all_h = [start_h, goal_h, h];
    all_labels = [{'Start', 'Goal'}, legendLabels];
    lgd = legend(all_h, all_labels, 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'Visible', 'off');
    set(lgd, 'Location', 'best');
    print(fig, 'legend_others', '-dpdf', '-r600');
    close(fig);
end