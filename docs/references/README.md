# CQSAC-PSO Research Papers Collection

This directory contains research papers related to the CQSAC-PSO (Advanced Parameter Exploration CrossQ-SAC for PSO) algorithm and its foundational methodologies.

## Reinforcement Learning Core Algorithms

### Actor-Critic Methods
1. **SAC_Original_2018_1801.01290.pdf**
   - Title: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
   - Authors: Haarnoja, T., Zhou, A., Abbeel, P., Levine, S.
   - Venue: ICML 2018
   - Key Contribution: Original SAC algorithm with entropy regularization

2. **SAC_Applications_2018_1812.05905.pdf**
   - Title: "Soft Actor-Critic Algorithms and Applications"
   - Authors: Haarnoja, T., Zhou, A., Finn, C., Abbeel, P., Levine, S.
   - Venue: ICML 2018
   - Key Contribution: Extended SAC with applications and practical improvements

3. **DDPG_2015_1509.02971.pdf**
   - Title: "Continuous control with deep reinforcement learning"
   - Authors: Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., Wierstra, D.
   - Venue: ICLR 2016
   - Key Contribution: Deep Deterministic Policy Gradient for continuous control

4. **A3C_2016_1602.01783.pdf**
   - Title: "Asynchronous Methods for Deep Reinforcement Learning"
   - Authors: Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., Kavukcuoglu, K.
   - Venue: ICML 2016
   - Key Contribution: Asynchronous Advantage Actor-Critic algorithm

5. **PolicyGradient_ActorCritic_2111.11232.pdf**
   - Title: "Policy Gradient and Actor-Critic Learning in Continuous Time and Space"
   - Authors: Lei et al.
   - ArXiv: 2111.11232
   - Key Contribution: Theoretical analysis of policy gradient methods

### Policy Optimization Methods
6. **PPO_2017_1707.06347.pdf**
   - Title: "Proximal Policy Optimization Algorithms"
   - Authors: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.
   - Venue: ICLR 2017
   - Key Contribution: Stable policy gradient method with clipped objective

### Value-Based Methods
7. **DQN_Atari_2015_1312.5602.pdf**
   - Title: "Playing Atari with Deep Reinforcement Learning"
   - Authors: Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., Riedmüller, M.
   - Venue: NIPS 2013
   - Key Contribution: Deep Q-Learning with experience replay and target networks

## CQSAC-PSO-Specific

8. **SAC_SAPSO_2024_MDPI.pdf**
   - Title: "Soft Actor-Critic Approach to Self-Adaptive Particle Swarm Optimisation"
   - Authors: Von Eschwege, D.H., Engelbrecht, A.P.
   - Journal: Mathematics, Vol. 12, No. 22, 2024
   - Publisher: MDPI (Open Access)
   - Key Contribution: Direct precursor to CQSAC-PSO, combining SAC with PSO parameter adaptation

## Optimization and Training Techniques

### Batch Processing & Normalization
9. **BatchNormalization_2015_1502.03167.pdf**
   - Title: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
   - Authors: Ioffe, S., Szegedy, C.
   - Venue: ICML 2015
   - Key Contribution: BatchNorm technique used in CrossQ implementation

10. **Adam_Optimizer_2014_1412.6980.pdf**
    - Title: "Adam: A Method for Stochastic Optimization"
    - Authors: Kingma, D.P., Ba, J.
    - Venue: ICLR 2015
    - Key Contribution: Adaptive learning rate optimizer

### Experience and Replay
11. **ExperienceReplay_2710.06574.pdf**
    - Title: "The Effects of Memory Replay in Reinforcement Learning"
    - Authors: Novati, G., Koumoutsakos, P.
    - ArXiv: 1710.06574
    - Key Contribution: Analysis of experience replay mechanisms

12. **ExperienceReplayFundamentals_2007.06700.pdf**
    - Title: "Revisiting Fundamentals of Experience Replay"
    - Authors: Novati, G., Koumoutsakos, P.
    - ArXiv: 2007.06700
    - Key Contribution: Comprehensive study of experience replay variants

### Advanced RL Techniques
13. **CrossQ_2019_1902.05605.pdf**
    - Title: "CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity"
    - Authors: Bhatt, A., Palenicek, D., Belousov, B., Argus, M., Amiranashvili, A., Brox, T., Peters, J.
    - Venue: ICLR 2024
    - Key Contribution: CrossQ optimization with BatchNorm and no target networks (used in CQSAC-PSO)

14. **Entropy_Regularization_1912.01557.pdf**
    - Title: "Policy Optimization Reinforcement Learning with Entropy Regularization"
    - Authors: Haarnoja, T., et al.
    - ArXiv: 1912.01557
    - Key Contribution: Theoretical analysis of entropy regularization in SAC

## Swarm Intelligence and PSO

15. **PSO_Survey_1804.05319.pdf**
    - Title: "Particle Swarm Optimization: A survey of historical and recent developments with hybridization perspectives"
    - Authors: Bonyadi, M.R., Michalewicz, Z.
    - ArXiv: 1804.05319
    - Key Contribution: Comprehensive PSO survey covering improvements and variants

## Reward Design and Optimization

16. **MultiObjective_RL_1809.06364.pdf**
    - Title: "Generalizing Across Multi-Objective Reward Functions in Deep Reinforcement Learning"
    - Authors: Pitis, R.
    - ArXiv: 1809.06364
    - Key Contribution: Multi-objective reward handling in deep RL (relevant to CQSAC-PSO's 4-component reward)

## Neural Network Architecture

17. **Transformer_Attention_2017_1706.03762.pdf**
    - Title: "Attention Is All You Need"
    - Authors: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.
    - Venue: NeurIPS 2017
    - Key Contribution: Transformer architecture with self-attention for sequence modeling in RL

18. **ResNet_2015_1512.03385.pdf**
    - Title: "Deep Residual Learning for Image Recognition"
    - Authors: He, K., Zhang, X., Ren, S., Sun, J.
    - Venue: CVPR 2016
    - Key Contribution: Residual connections for training deep networks

## Gradient Descent and Optimization

19. **EvolutionStrategies_2017_1703.03864.pdf**
    - Title: "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"
    - Authors: Salimans, T., Ho, J., Chen, X., Sidor, S., Sutskever, I.
    - ArXiv: 1703.03864
    - Key Contribution: Alternative optimization approach for comparison with RL-based methods

## Usage and Citation

These papers provide the theoretical foundation and technical context for understanding CQSAC-PSO:

- **Core Algorithm**: SAC_Original_2018 + CrossQ_2019
- **State Representation**: Transformer_Attention_2017
- **Parameter Domain**: PSO_Survey_1804
- **Reward Design**: MultiObjective_RL_1809 + Entropy_Regularization_1912
- **Network Architecture**: ResNet_2015 + BatchNormalization_2015
- **Optimization**: Adam_Optimizer_2014 + DDPG_2015
- **Practical Foundation**: SAC_SAPSO_2024 (Direct precursor)

## Paper Statistics

- **Total Papers**: 19
- **Conference Papers**: 10
- **Journal Papers**: 2
- **ArXiv Papers**: 7
- **Publication Year Range**: 2013-2024
- **Total Size**: ~38 MB

## Download Sources

Papers were downloaded from:
- **arXiv.org**: 13 papers (free preprints)
- **MDPI**: 1 paper (open access)
- **Conference/Journal sites**: 5 papers (peer-reviewed versions)

## Recommendations for Reading

**Essential Foundation** (Start Here):
1. SAC_Original_2018
2. CrossQ_2019
3. Transformer_Attention_2017
4. SAC_SAPSO_2024

**Core Understanding**:
5. PSO_Survey_1804
6. DQN_Atari_2015
7. DDPG_2015
8. PPO_2017

**Optimization & Training**:
9. BatchNormalization_2015
10. Adam_Optimizer_2014
11. ExperienceReplayFundamentals_2007

**Advanced Topics**:
12. A3C_2016
13. MultiObjective_RL_1809
14. Entropy_Regularization_1912
15. EvolutionStrategies_2017

---

**Last Updated**: November 10, 2025
**CQSAC-PSO Repository**: C:\Users\PC\Desktop\Copy_of_New Folder - Copy\
