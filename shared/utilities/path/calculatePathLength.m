function pathLength = calculatePathLength(path)
    % Calculate total path length
    pathLength = 0;
    for i = 1:size(path, 1)-1
        pathLength = pathLength + norm(path(i+1,:) - path(i,:));
    end
end

