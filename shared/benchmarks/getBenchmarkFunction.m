function info = getBenchmarkFunction(name, D)
%GETBENCHMARKFUNCTION Return function handle, bounds, and metadata for a benchmark.
%
%   info = getBenchmarkFunction('Ackley1', 30)
%
%   Returns struct with:
%     f     — function handle @(x) where x is 1×D row vector
%     lb    — 1×D lower bounds
%     ub    — 1×D upper bounds
%     fmin  — known global minimum value (NaN if unknown)
%     xmin  — known global minimizer (empty if unknown)
%     name  — canonical function name
%     notes — any implementation notes or ambiguities
%
%   SAC-SAPSO paper benchmark split (von Eschwege & Engelbrecht 2024).
%   All functions implemented for arbitrary dimension D (default 30).

    if nargin < 2, D = 30; end

    info = struct('f', [], 'lb', [], 'ub', [], 'fmin', NaN, ...
                  'xmin', [], 'name', name, 'notes', '');

    switch name
        % ================================================================
        %  TRAINING SET (45 functions)
        % ================================================================

        case 'Ackley1'
            info.f = @(x) -20*exp(-0.2*sqrt(sum(x.^2)/D)) ...
                          - exp(sum(cos(2*pi*x))/D) + 20 + exp(1);
            info.lb = -32*ones(1,D); info.ub = 32*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'Alpine1'
            info.f = @(x) sum(abs(x.*sin(x) + 0.1*x));
            info.lb = zeros(1,D); info.ub = 10*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'Bohachevsky1'
            % Pairwise generalization to n-D
            info.f = @(x) bohachevsky1(x);
            info.lb = -100*ones(1,D); info.ub = 100*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'BonyadiMichalewicz'
            % From Bonyadi & Michalewicz PSO survey. Implemented as
            % Schwefel 2.26 variant: f(x) = -sum(x.*sin(sqrt(|x|))).
            % DISTINCT from standard Michalewicz in test set.
            info.f = @(x) -sum(x.*sin(sqrt(abs(x))));
            info.lb = -500*ones(1,D); info.ub = 500*ones(1,D);
            info.fmin = -D*418.9829; info.xmin = 420.9687*ones(1,D);
            info.notes = 'Schwefel 2.26 variant attributed to Bonyadi-Michalewicz PSO survey.';

        case 'Brown'
            info.f = @(x) brownFunc(x);
            info.lb = -1*ones(1,D); info.ub = 4*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'CosineMixture'
            info.f = @(x) -0.1*sum(cos(5*pi*x)) + sum(x.^2);
            info.lb = -1*ones(1,D); info.ub = 1*ones(1,D);
            info.fmin = -0.1*D; info.xmin = zeros(1,D);
            info.notes = 'Minimization form. Global min at x=0.';

        case 'DeflectedCorrugatedSpring'
            alpha = 5; K = 5;
            info.f = @(x) 0.1*sum((x-alpha).^2) - cos(K*sqrt(sum((x-alpha).^2)));
            info.lb = zeros(1,D); info.ub = 2*alpha*ones(1,D);
            info.fmin = -1; info.xmin = alpha*ones(1,D);

        case 'Discus'
            info.f = @(x) 1e6*x(1)^2 + sum(x(2:end).^2);
            info.lb = -100*ones(1,D); info.ub = 100*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'DropWave'
            % Natural n-D generalization via radial distance
            info.f = @(x) -(1+cos(12*sqrt(sum(x.^2))))/(2+0.5*sum(x.^2));
            info.lb = -5.12*ones(1,D); info.ub = 5.12*ones(1,D);
            info.fmin = -1; info.xmin = zeros(1,D);

        case 'EggCrate'
            info.f = @(x) sum(x.^2) + 25*sum(sin(x).^2);
            info.lb = -5*ones(1,D); info.ub = 5*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'EggHolder'
            % Pairwise generalization to n-D
            info.f = @(x) eggholderFunc(x);
            info.lb = -512*ones(1,D); info.ub = 512*ones(1,D);
            info.fmin = NaN; info.notes = 'Global min depends on D; no closed form for n-D.';

        case 'Elliptic'
            info.f = @(x) ellipticFunc(x, D);
            info.lb = -100*ones(1,D); info.ub = 100*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'Exponential'
            info.f = @(x) -exp(-0.5*sum(x.^2));
            info.lb = -1*ones(1,D); info.ub = 1*ones(1,D);
            info.fmin = -1; info.xmin = zeros(1,D);

        case 'Giunta'
            % n-D generalization (sum over dimensions)
            info.f = @(x) giuntaFunc(x);
            info.lb = -1*ones(1,D); info.ub = 1*ones(1,D);
            info.fmin = NaN; info.notes = 'n-D sum form. Approx fmin=0.06447*D.';

        case 'HolderTable1'
            % Pairwise generalization to n-D
            info.f = @(x) holderTable1Func(x);
            info.lb = -10*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = NaN; info.notes = 'Pairwise n-D generalization.';

        case 'Levy3'
            info.f = @(x) levyFunc(x);
            info.lb = -10*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = 0; info.xmin = ones(1,D);

        case 'LevyMontalvo2'
            info.f = @(x) levyMontalvo2Func(x);
            info.lb = -5*ones(1,D); info.ub = 5*ones(1,D);
            info.fmin = 0; info.xmin = ones(1,D);

        case 'Mishra1'
            info.f = @(x) mishra1Func(x, D);
            info.lb = zeros(1,D); info.ub = ones(1,D);
            info.fmin = NaN; info.notes = 'Global min near 2 for D=30.';

        case 'Mishra4'
            % Pairwise generalization
            info.f = @(x) mishra4Func(x);
            info.lb = -10*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = NaN;

        case 'NeedleEye'
            eye = 0.0001;
            info.f = @(x) needleEyeFunc(x, eye);
            info.lb = -10*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = 1; info.xmin = zeros(1,D);

        case 'Norwegian'
            % n-D sum form (product collapses to 0 in high dimensions)
            info.f = @(x) norwegianFunc(x);
            info.lb = -1.1*ones(1,D); info.ub = 1.1*ones(1,D);
            info.fmin = -D; info.xmin = ones(1,D);
            info.notes = 'Sum-form n-D generalization; product form impractical at D=30.';

        case 'Pathological'
            info.f = @(x) pathologicalFunc(x);
            info.lb = -100*ones(1,D); info.ub = 100*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);
            info.notes = 'AMPGO formula: sum of pairwise terms.';

        case 'Penalty1'
            info.f = @(x) penalty1Func(x);
            info.lb = -50*ones(1,D); info.ub = 50*ones(1,D);
            info.fmin = 0; info.xmin = -ones(1,D);

        case 'Penalty2'
            info.f = @(x) penalty2Func(x);
            info.lb = -50*ones(1,D); info.ub = 50*ones(1,D);
            info.fmin = 0; info.xmin = ones(1,D);

        case 'Periodic'
            % n-D generalization
            info.f = @(x) 1 + sum(sin(x).^2) - 0.1*exp(-sum(x.^2));
            info.lb = -10*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = 0.9; info.xmin = zeros(1,D);

        case 'Pinter2'
            info.f = @(x) pinterFunc(x, D);
            info.lb = -10*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'Price2'
            % AMPGO: f(x) = 1 + sin^2(x1) + sin^2(x2) - 0.1*exp(-x1^2-x2^2)
            % n-D generalization via sum
            info.f = @(x) 1 + sum(sin(x).^2) - 0.1*exp(-sum(x.^2));
            info.lb = -10*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = 0.9; info.xmin = zeros(1,D);
            info.notes = 'Same formula as Periodic; both from AMPGO. Kept as separate entry per paper.';

        case 'Qing'
            info.f = @(x) qingFunc(x, D);
            info.lb = -500*ones(1,D); info.ub = 500*ones(1,D);
            info.fmin = 0;

        case 'Quadric'
            info.f = @(x) quadricFunc(x, D);
            info.lb = -100*ones(1,D); info.ub = 100*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'Quintic'
            info.f = @(x) sum(abs(x.^5 - 3*x.^4 + 4*x.^3 + 2*x.^2 - 10*x - 4));
            info.lb = -10*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = 0;

        case 'Rana'
            info.f = @(x) ranaFunc(x);
            info.lb = -500*ones(1,D); info.ub = 500*ones(1,D);
            info.fmin = NaN; info.notes = 'Global min depends on D.';

        case 'Rastrigin'
            info.f = @(x) 10*D + sum(x.^2 - 10*cos(2*pi*x));
            info.lb = -5.12*ones(1,D); info.ub = 5.12*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'Ripple25'
            info.f = @(x) ripple25Func(x);
            info.lb = zeros(1,D); info.ub = ones(1,D);
            info.fmin = -D; info.xmin = 0.1*ones(1,D);

        case 'Rosenbrock'
            info.f = @(x) rosenbrockFunc(x);
            info.lb = -5*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = 0; info.xmin = ones(1,D);

        case 'Salomon'
            info.f = @(x) 1 - cos(2*pi*sqrt(sum(x.^2))) + 0.1*sqrt(sum(x.^2));
            info.lb = -100*ones(1,D); info.ub = 100*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'Schubert4'
            info.f = @(x) schubert4Func(x);
            info.lb = -10*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = NaN; info.notes = 'Many global minima; value depends on D.';

        case 'Schwefel1'
            info.f = @(x) sum(x.^2)^sqrt(pi);
            info.lb = -100*ones(1,D); info.ub = 100*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'Sinusoidal'
            A = 2.5; B = 5; Z = 30;
            info.f = @(x) sinusoidalFunc(x, A, B, Z);
            info.lb = zeros(1,D); info.ub = 180*ones(1,D);
            info.fmin = -(A+1); info.xmin = (90+Z)*ones(1,D);

        case 'StepFunction3'
            info.f = @(x) sum(floor(x.^2));
            info.lb = -100*ones(1,D); info.ub = 100*ones(1,D);
            info.fmin = 0;

        case 'Trid'
            info.f = @(x) tridFunc(x);
            info.lb = -D^2*ones(1,D); info.ub = D^2*ones(1,D);
            info.fmin = -D*(D+4)*(D-1)/6;

        case 'Trigonometric'
            info.f = @(x) trigonometricFunc(x, D);
            info.lb = zeros(1,D); info.ub = pi*ones(1,D);
            info.fmin = 0;

        case 'Vincent'
            info.f = @(x) -sum(sin(10*log(x)));
            info.lb = 0.25*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = -D; info.xmin = 7.70628098*ones(1,D);

        case 'Weierstrass'
            info.f = @(x) weierstrassFunc(x, D);
            info.lb = -0.5*ones(1,D); info.ub = 0.5*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'XinSheYang1'
            % Standard form: sum(epsilon_i * |x_i|^i) with fixed epsilon
            rng_state = rng(123, 'twister');
            epsilon = rand(1, D);
            rng(rng_state);
            info.f = @(x) xinSheYang1Func(x, D, epsilon);
            info.lb = -5*ones(1,D); info.ub = 5*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'XinSheYang2'
            info.f = @(x) sum(abs(x)) * exp(-sum(sin(x.^2)));
            info.lb = -2*pi*ones(1,D); info.ub = 2*pi*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        % ================================================================
        %  TEST SET (7 functions)
        % ================================================================

        case 'CrossLegTable'
            % Pairwise n-D generalization of the 2D function
            info.f = @(x) crossLegTableFunc(x);
            info.lb = -10*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = NaN;
            info.notes = 'Pairwise n-D generalization of 2D Cross Leg Table.';

        case 'Lanczos3'
            % Sum-of-exponentials curve fitting; replicated for 30D.
            % Each block of 6 variables fits y = b1*exp(-b2*t) + b3*exp(-b4*t) + b5*exp(-b6*t).
            info.f = @(x) lanczos3Func(x, D);
            info.lb = -10*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = 0;
            info.notes = ['NIST Lanczos3 is 6-parameter. Extended to 30D by replicating ' ...
                          '5 independent sub-problems and summing residuals.'];

        case 'Michalewicz'
            m = 10;
            info.f = @(x) michalewiczFunc(x, m, D);
            info.lb = zeros(1,D); info.ub = pi*ones(1,D);
            info.fmin = NaN; info.notes = 'Global min depends on D; approx -29.63 for D=30.';

        case 'Schaffer4'
            % Pairwise n-D generalization
            info.f = @(x) schaffer4Func(x);
            info.lb = -100*ones(1,D); info.ub = 100*ones(1,D);
            info.fmin = NaN;

        case 'SineEnvelope'
            info.f = @(x) sineEnvelopeFunc(x);
            info.lb = -100*ones(1,D); info.ub = 100*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'StretchedVSineWave'
            info.f = @(x) stretchedVFunc(x);
            info.lb = -10*ones(1,D); info.ub = 10*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        case 'Wavy'
            k = 10;
            info.f = @(x) 1 - (1/D)*sum(cos(k*x).*exp(-x.^2/2));
            info.lb = -pi*ones(1,D); info.ub = pi*ones(1,D);
            info.fmin = 0; info.xmin = zeros(1,D);

        otherwise
            error('getBenchmarkFunction:UnknownFunction', ...
                'Unknown benchmark function: "%s"', name);
    end
end

% ====================================================================
%  LOCAL HELPER FUNCTIONS
% ====================================================================

function y = bohachevsky1(x)
    n = numel(x);
    y = 0;
    for i = 1:n-1
        y = y + x(i)^2 + 2*x(i+1)^2 ...
            - 0.3*cos(3*pi*x(i)) - 0.4*cos(4*pi*x(i+1)) + 0.7;
    end
end

function y = brownFunc(x)
    n = numel(x);
    y = 0;
    for i = 1:n-1
        y = y + (x(i)^2)^(x(i+1)^2+1) + (x(i+1)^2)^(x(i)^2+1);
    end
end

function y = eggholderFunc(x)
    n = numel(x);
    y = 0;
    for i = 1:n-1
        y = y - (x(i+1)+47)*sin(sqrt(abs(x(i+1)+x(i)/2+47))) ...
            - x(i)*sin(sqrt(abs(x(i)-x(i+1)-47)));
    end
end

function y = ellipticFunc(x, D)
    idx = 0:D-1;
    coeff = 1e6 .^ (idx / max(1, D-1));
    y = sum(coeff .* x.^2);
end

function y = giuntaFunc(x)
    t = 16/15*x - 1;
    y = 0.6 + sum(sin(t) + sin(t).^2 + (1/50)*sin(4*t));
end

function y = holderTable1Func(x)
    n = numel(x);
    y = 0;
    for i = 1:n-1
        r = sqrt(x(i)^2 + x(i+1)^2);
        y = y - abs(sin(x(i))*cos(x(i+1))*exp(abs(1-r/pi)));
    end
end

function y = levyFunc(x)
    w = 1 + (x - 1)/4;
    n = numel(w);
    y = sin(pi*w(1))^2;
    for i = 1:n-1
        y = y + (w(i)-1)^2*(1+10*sin(pi*w(i)+1)^2);
    end
    y = y + (w(n)-1)^2*(1+sin(2*pi*w(n))^2);
end

function y = levyMontalvo2Func(x)
    n = numel(x);
    y = 0.1*(sin(3*pi*x(1))^2);
    for i = 1:n-1
        y = y + (x(i)-1)^2*(1+sin(3*pi*x(i+1))^2);
    end
    y = y + (x(n)-1)^2*(1+sin(2*pi*x(n))^2);
end

function y = mishra1Func(x, D)
    gn = D - sum(x(1:end-1));
    y = (1 + gn)^gn;
end

function y = mishra4Func(x)
    n = numel(x);
    y = 0;
    for i = 1:n-1
        y = y + sqrt(abs(sin(sqrt(abs(x(i)^2+x(i+1)^2)))));
    end
end

function y = needleEyeFunc(x, eye)
    if all(abs(x) < eye)
        y = 1;
    else
        y = sum(100 + abs(x(abs(x) >= eye)));
    end
end

function y = norwegianFunc(x)
    % Sum-form n-D generalization (product collapses at D=30)
    y = -sum(cos(pi*x.^3) .* ((99+x) / 100));
end

function y = pathologicalFunc(x)
    n = numel(x);
    y = 0;
    for i = 1:n-1
        num = sin(sqrt(100*x(i+1)^2 + x(i)^2))^2 - 0.5;
        den = 1 + 0.001*(x(i)-x(i+1))^4;
        y = y + 0.5 + num/den;
    end
end

function y = penalty1Func(x)
    n = numel(x);
    w = 1 + (x+1)/4;
    y = (pi/n)*(10*sin(pi*w(1))^2);
    for i = 1:n-1
        y = y + (pi/n)*(w(i)-1)^2*(1+10*sin(pi*w(i+1))^2);
    end
    y = y + (pi/n)*(w(n)-1)^2;
    y = y + sum(penaltyU(x, 10, 100, 4));
end

function y = penalty2Func(x)
    n = numel(x);
    y = 0.1*(sin(3*pi*x(1))^2);
    for i = 1:n-1
        y = y + 0.1*(x(i)-1)^2*(1+sin(3*pi*x(i+1))^2);
    end
    y = y + 0.1*(x(n)-1)^2*(1+sin(2*pi*x(n))^2);
    y = y + sum(penaltyU(x, 5, 100, 4));
end

function u = penaltyU(x, a, k, m)
    u = zeros(size(x));
    mask_hi = x > a;
    mask_lo = x < -a;
    u(mask_hi) = k*(x(mask_hi)-a).^m;
    u(mask_lo) = k*(-x(mask_lo)-a).^m;
end

function y = pinterFunc(x, D)
    n = D;
    y = 0;
    for i = 1:n
        im1 = mod(i-2, n)+1;
        ip1 = mod(i, n)+1;
        A = x(im1)*sin(x(i)) + sin(x(ip1));
        B = x(im1)^2 - 2*x(i) + 3*x(ip1) - cos(x(i)) + 1;
        y = y + i*x(i)^2 + 20*i*sin(A)^2 + i*log10(1+i*B^2);
    end
end

function y = qingFunc(x, D)
    idx = 1:D;
    y = sum((x.^2 - idx).^2);
end

function y = quadricFunc(x, D)
    y = 0;
    for i = 1:D
        y = y + sum(x(1:i))^2;
    end
end

function y = ranaFunc(x)
    n = numel(x);
    y = 0;
    for i = 1:n-1
        t1 = sqrt(abs(x(i+1)+x(i)+1));
        t2 = sqrt(abs(x(i+1)-x(i)+1));
        y = y + (x(i+1)+1)*cos(t2)*sin(t1) + x(i)*cos(t1)*sin(t2);
    end
end

function y = ripple25Func(x)
    y = sum(-exp(-2*log(2)*((x-0.1)/0.8).^2) .* sin(5*pi*x).^6);
end

function y = rosenbrockFunc(x)
    n = numel(x);
    y = 0;
    for i = 1:n-1
        y = y + 100*(x(i)^2 - x(i+1))^2 + (x(i)-1)^2;
    end
end

function y = schubert4Func(x)
    n = numel(x);
    y = 0;
    for i = 1:n
        for j = 1:5
            y = y - j*cos((j+1)*x(i) + j);
        end
    end
end

function y = sinusoidalFunc(x, A, B, Z)
    % Use log-space to prevent underflow in high dimensions
    s1 = sin(x - Z);
    s2 = sin(B*(x - Z));
    % Handle sign via separate tracking
    sign1 = prod(sign(s1));
    sign2 = prod(sign(s2));
    logprod1 = sum(log(abs(s1) + 1e-300));
    logprod2 = sum(log(abs(s2) + 1e-300));
    y = -(A * sign1 * exp(logprod1) + sign2 * exp(logprod2));
end

function y = tridFunc(x)
    n = numel(x);
    y = sum((x-1).^2);
    for i = 2:n
        y = y - x(i)*x(i-1);
    end
end

function y = trigonometricFunc(x, D)
    y = 0;
    for i = 1:D
        ti = D - sum(cos(x)) + i*(1 - cos(x(i))) - sin(x(i));
        y = y + ti^2;
    end
end

function y = weierstrassFunc(x, D)
    kmax = 20; a = 0.5; b = 3;
    k = 0:kmax;
    ak = a.^k;
    bk = b.^k;
    y = 0;
    for i = 1:D
        y = y + sum(ak .* cos(2*pi*bk*(x(i)+0.5)));
    end
    y = y - D*sum(ak .* cos(pi*bk));
end

function y = xinSheYang1Func(x, D, epsilon)
    idx = 1:D;
    y = sum(epsilon .* abs(x).^idx);
end

function y = crossLegTableFunc(x)
    n = numel(x);
    y = 0;
    for i = 1:n-1
        r = sqrt(x(i)^2 + x(i+1)^2);
        inner = abs(sin(x(i))*sin(x(i+1))*exp(abs(100-r/pi)));
        y = y - (inner + 1)^(-0.1);
    end
end

function y = lanczos3Func(x, D)
    % NIST Lanczos3: y = b1*exp(-b2*t) + b3*exp(-b4*t) + b5*exp(-b6*t)
    % True params: [0.0951, 1, 0.8607, 3, 1.5576, 5]
    % Extended to 30D: 5 independent 6-parameter sub-problems
    tData = linspace(0, 1.15, 24);
    yData = 0.0951*exp(-tData) + 0.8607*exp(-3*tData) + 1.5576*exp(-5*tData);

    numBlocks = floor(D/6);
    y = 0;
    for block = 1:numBlocks
        idx = (block-1)*6 + (1:6);
        b = x(idx);
        % Constrain b2,b4,b6 to positive via abs()
        yPred = b(1)*exp(-abs(b(2))*tData) + b(3)*exp(-abs(b(4))*tData) + b(5)*exp(-abs(b(6))*tData);
        y = y + sum((yPred - yData).^2);
    end
    % Handle remaining dimensions if D is not multiple of 6
    remaining = mod(D, 6);
    if remaining > 0
        y = y + sum(x(numBlocks*6+1:end).^2);
    end
end

function y = michalewiczFunc(x, m, ~)
    n = numel(x);
    idx = 1:n;
    y = -sum(sin(x) .* sin(idx.*x.^2/pi).^(2*m));
end

function y = schaffer4Func(x)
    n = numel(x);
    y = 0;
    for i = 1:n-1
        num = cos(sin(abs(x(i)^2 - x(i+1)^2)))^2 - 0.5;
        den = (1 + 0.001*(x(i)^2 + x(i+1)^2))^2;
        y = y + 0.5 + num/den;
    end
end

function y = sineEnvelopeFunc(x)
    n = numel(x);
    y = 0;
    for i = 1:n-1
        r2 = x(i)^2 + x(i+1)^2;
        num = sin(sqrt(r2) - 0.5)^2;
        den = (1 + 0.001*r2)^2;
        y = y - (num/den + 0.5);
    end
end

function y = stretchedVFunc(x)
    n = numel(x);
    y = 0;
    for i = 1:n-1
        t = x(i)^2 + x(i+1)^2;
        y = y + t^0.25 * (sin(50*t^0.1)+1)^2;
    end
end
