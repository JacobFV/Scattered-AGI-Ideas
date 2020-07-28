# Research

Some relevant articles that I will apply in my RL systems

- [Free Energy Principle](https://en.wikipedia.org/wiki/Free_energy_principle)
  - minimize variational free energy by adjusting 1) actions 2) world models

- [Active Inference: A Process Theory](https://www.mitpressjournals.org/doi/pdf/10.1162/NECO_a_00912#:~:text=In%20brief%2C%20active%20inference%20separates,model%20of%20(observed)%20outcomes.)
  > "all neuronal processing (and action selection) can be explained by maximizing Bayesian model evidence—or minimizing variational free energy \[. . .\] the fact that a gradient descent appears to be a valid description of neuronal activity means that variational free energy is a Lyapunov function for neuronal dynamics, which therefore conform to Hamilton’s principle of least action

- [Automatic Recall Machines: Internal Replay, Continual Learning and the Brain](https://arxiv.org/abs/2006.12323)
  - "optimizing for not forgetting calls for the generation of samples that are specialized to each real training batch"
  - generates samples from implicit internal memory and trains on reals data most conflicting with generated data
  - backpropagation "inverts" the neural network
  - backpropagation identifies which experiences are most conflicting with existing memories
  - internal replay of unexpected stimulii by backpropagation similar to top-down biasing by the neocortex

- [Nengo Publications](nengo.ai/publications)
  - spiking neural network are more biologically accurate models 
  
- [Predictive Information Accelerates Learning in RL](https://arxiv.org/pdf/2007.12401.pdf)
  - maximizing mutual information between past and future by selective feature attention/compression improves sample efficency
  - defines _predictive information_ as "the mutual information between the past and the future"
  - mathematical formulation for predictive information in paper
  > the task of the agent may be described as finding a representation of the past that is most useful for predicting the future, upon which an optimal policy may more easily be learned.

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
  
- [Complex Skill Acquisition through Simple Skill Adversarial Imitation Learning](https://arxiv.org/abs/2007.10281)
  - "Some skills can be considered as approximate combinations of certain subskill"
  - aims to learn a "latent space structure so that relationships between embeddings of behaviors and embeddings of subskills that comprise these behaviors are captured in a meaningful way"
 
- Generate temporal superresolution trajectories by training on accelerated sequences
