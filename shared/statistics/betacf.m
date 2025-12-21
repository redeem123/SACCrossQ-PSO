function cf = betacf(a, b, x)
    % Continued fraction for incomplete beta function
    % Simplified implementation
    
    maxIter = 100;
    eps = 1e-10;
    
    am = 1;
    bm = 1;
    az = 1;
    
    qab = a + b;
    qap = a + 1;
    qam = a - 1;
    bz = 1 - qab * x / qap;
    
    for m = 1:maxIter
        em = m;
        tem = em + em;
        d = em * (b - m) * x / ((qam + tem) * (a + tem));
        ap = az + d * am;
        bp = bz + d * bm;
        d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem));
        app = ap + d * az;
        bpp = bp + d * bz;
        
        aold = az;
        am = ap / bpp;
        bm = bp / bpp;
        az = app / bpp;
        bz = 1;
        
        if abs(az - aold) < eps * abs(az)
            break;
        end
    end
    
    cf = az;
end

