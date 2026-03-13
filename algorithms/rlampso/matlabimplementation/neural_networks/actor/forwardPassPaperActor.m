function action = forwardPassPaperActor(actor, state)
    % PAPER: 4-layer network with LeakyReLU + tanh output
    
    % Layer 1
    z1 = actor.W1 * state + actor.b1;
    h1 = leakyReLU(z1);
    
    % Layer 2
    z2 = actor.W2 * h1 + actor.b2;
    h2 = leakyReLU(z2);
    
    % Layer 3
    z3 = actor.W3 * h2 + actor.b3;
    h3 = leakyReLU(z3);
    
    % Output layer with tanh (PAPER: map to [-1, 1])
    z4 = actor.W4 * h3 + actor.b4;
    action = tanh(z4);
end

