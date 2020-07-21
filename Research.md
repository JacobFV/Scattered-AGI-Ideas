# Research

Some relevant articles that I will apply in my RL systems

- [Nengo](nengo.ai/publications)
  - more biologically accurate networks 

- [Goal-Aware Prediction: Learning to Model What Matters](https://arxiv.org/abs/2007.07170)
  - unaligned objectives: future state reconstruction and policy objective
  - they "direct prediction towards task relevant information, enabling the model to be aware of the current task and encouraging it to only model relevant quantities of the state space, resulting in a learning objective that more closely matches the downstream task"
  - uses residual latent state forward model with the latent state combining goal and ground truth
  - nearing the goal, forward dynamics residuals approach zero

- [An Open-World Simulated Environment for Developmental Robotics](https://arxiv.org/abs/2007.09300)
  - ackowledges the rl focus shifting toward multi-modality, self-supervised learning
  - Introduces "SEDRo, a Simulated Environment for Developmental Robotics which allows a learning agent to have similar experiences that a human infant goes through from the fetus stage up to 12 months"
  - didn't see link to code in paper

- [Learning High-Level Policies for Model Predictive Control](https://arxiv.org/abs/2007.10284)
  - heirarchial dcomposition of MDP's with self-supervised higher level modeling
  - Code: https://github.com/uzh-rpg/high_mpc
  - This paper has been selected for future presentation
 <img src="https://github.com/uzh-rpg/high_mpc/raw/master/docs/figures/MethodOverview.png" width=50%>
 
- [Reinforcement Communication Learning in Different Social Network
Structures](https://arxiv.org/abs/2007.09820)
  - more connected social networks converge to more consistant dialects
  
- [Beyond Prioritized Replay: Sampling States in Model-Based RL via Simulated Priorities](https://arxiv.org/pdf/2007.09569.pdf)
  - actively search for high priority states with gradient *ascent*
  - simulated prioritized trajectories are generally diverse
  
- [ESCELL: Emergent Symbolic Cellular Language](https://arxiv.org/abs/2007.09469)
  - a sender and reciever cooperate to build emergant symbolic language to describe biological cells
  
- Generate temporal superresolution trajectories by training on accelerated sequences
