function rlamAgent = softUpdatePaperTargets(rlamAgent)
    % PAPER: Soft update target networks
    
    tau = rlamAgent.tau;
    
    % Update target actor
    rlamAgent.targetActor.W1 = tau * rlamAgent.actor.W1 + (1 - tau) * rlamAgent.targetActor.W1;
    rlamAgent.targetActor.W2 = tau * rlamAgent.actor.W2 + (1 - tau) * rlamAgent.targetActor.W2;
    rlamAgent.targetActor.W3 = tau * rlamAgent.actor.W3 + (1 - tau) * rlamAgent.targetActor.W3;
    rlamAgent.targetActor.W4 = tau * rlamAgent.actor.W4 + (1 - tau) * rlamAgent.targetActor.W4;
    
    % Update target critic (all 6 layers)
    rlamAgent.targetCritic.W1 = tau * rlamAgent.critic.W1 + (1 - tau) * rlamAgent.targetCritic.W1;
    rlamAgent.targetCritic.W2 = tau * rlamAgent.critic.W2 + (1 - tau) * rlamAgent.targetCritic.W2;
    rlamAgent.targetCritic.W3 = tau * rlamAgent.critic.W3 + (1 - tau) * rlamAgent.targetCritic.W3;
    rlamAgent.targetCritic.W4 = tau * rlamAgent.critic.W4 + (1 - tau) * rlamAgent.targetCritic.W4;
    rlamAgent.targetCritic.W5 = tau * rlamAgent.critic.W5 + (1 - tau) * rlamAgent.targetCritic.W5;
    rlamAgent.targetCritic.W6 = tau * rlamAgent.critic.W6 + (1 - tau) * rlamAgent.targetCritic.W6;
end

%% 2.. RLAM-PGPSO (Policy Gradient)

