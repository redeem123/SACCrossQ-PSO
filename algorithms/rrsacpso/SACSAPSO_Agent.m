classdef SACSAPSO_Agent < RRSACPSO_Agent
    % Thin wrapper that fixes the baseline identity for the UAV-adapted SAC-SAPSO path.

    methods
        function obj = SACSAPSO_Agent(config)
            obj@RRSACPSO_Agent(config);
        end
    end
end
