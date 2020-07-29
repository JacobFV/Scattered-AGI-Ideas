# Research

Some relevant articles that I will apply in my RL systems

- [Relational Neural Machines](https://arxiv.org/abs/2002.02193)
  - neurosymbolic integration paper
  > This paper presents Relational Neural Machines, a novel framework allowing to jointly train the parameters of the learners and of a First–Order Logic based reasoner. A Relational Neural Machine is able to recover both classical learning from supervised data in case of pure sub-symbolic learning, and Markov Logic Networks in case of pure symbolic reasoning, while
allowing to jointly train and perform inference in hybrid learning tasks

- [Graph Representation Learning via Graphical Mutual Information Maximization](https://arxiv.org/abs/2002.01169)
  - proposes "Graphical Mutual Information (GMI), to measure the correlation between input graphs and high-level hidden representations."

- [Emergent cooperation through mutual information maximization](https://arxiv.org/abs/2006.11769)
  >  The algorithm is based on the hypothesis that highly correlated actions are a feature of cooperative systems, and hence, we propose the insertion of an auxiliary objective of maximization of the mutual information between the actions of agents in the learning problem \[. . .\] maximization of mutual information among agents promotes the emergence of cooperation in social dilemmas

- [Collective Learning by Ensembles of Altruistic Diversifying Neural Networks](https://arxiv.org/abs/2006.11671)
  > ensembles of interacting neural networks \[. . .\] aim to maximize their own performance but also their functional relations to other networks \[. . .\] outperform independent ones, and that optimal ensemble performance is reached when the coupling between networks increases diversity and degrades the performance of individual networks \[. . .\] even without a global goal for the ensemble, optimal collective behavior emerges from local interactions between networks

- [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547)
  - intelligence is "skill-acquisition efficiency"
  - describes and introduces the Abstraction and Reasoning Corpus "general AI benchmark" 
  > solely measuring skill at any given task falls short of measuring intelligence, because skill is heavily modulated by prior knowledge and experience: unlimited priors or unlimited training data allow experimenters to "buy" arbitrary levels of skills for a system, in a way that masks the system's own generalization power.
  - [human intelligence &ne; AGI](https://arxiv.org/abs/2007.07710)
  - video series
    1. [Foundations](https://youtu.be/3_qGrmD6iQY)
    2. [Human Priors](https://youtu.be/THcuTJbeD34)
    3. [Math](https://youtu.be/cuyM63ugsxI)
    4. [The ARC Challange](https://youtu.be/O9kFX33nUcU)

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

- [Noisy Agents: Self-supervised Exploration by Predicting Auditory Events](https://arxiv.org/abs/2007.13729)
  - "we introduce an intrinsic reward function of predicting sound for RL exploration"
  - "use the prediction errors as intrinsic rewards to guide RL exploration."
  - "The intrinsic rewards could serve as incentives that allow the agent to distinguish novel and fruitful states, but the lack of extrinsic rewards impedes the awareness of auditory events where agents can earn more rewards and need to visit again"
  - " the intrinsic reward can diminish quickly during training, since the learned predictive model usually converges to a stable state representation of the environment"
  > "Since auditory signals are prevalent in real-world scenarios, we believe that combining them with visual signals could help guide exploration in many robotic applications [. . .]. For example, the honk of a car may be a useful signal that a self-driving agent has entered an unexpected situation"

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

# General AI References

## Meta-learning beyond few-shot learning

> When we say a task was “easy” to learn, we usually mean that it didn’t take us too long and that the process was relatively smooth. From a machine learning perspective, this implies rapid convergence. It also implies parameter updates should improve performance monotonically (well, in expectation at least). Oscillating back and forth is equivalent to not knowing what to do.
>
> Both these notions revolve around how we travel from our initialisation to our final destination on the model’s loss surface. The ideal is a going straight down-hill to the parameterisation with smallest loss for the given task. Worst case is taking a long detour with lots of back-and-forts. In Leap, we leverage the following insight:
>
> Transferring knowledge therefore implies influencing the parameter trajectory such that it converges as rapidly and smoothly as possible.
>
> ![surf](http://flennerhag.com/img/leap/surf.png)
>
> Transferring knowledge across learning processes means that learning a new task becomes easier in the sense that we enjoy a shorter and smoother parameter trajectory.
>
> ...
>
> Consequently, we can learn to transfer knowledge across learning processes by learning an initialisation such that the expected distance we have to travel when learning a similar task is as short as possible.
>
> ...
>
> Leap learns an initialisation $\theta_0$ such that the expected distance of any learning process from that task distribution is as short as possible in expectation. Thus, Leap extracts information across a variety of learning processes during meta-training and condenses it into a good initial guess that ensures learning a new task is as easy as possible. Importantly, this initial guess has nothing to do with the details of the final parameterisation on any task, it is meta-learned to facilitate the process of learning those parameters, whatever they might be.
>
> ![evo](http://flennerhag.com/img/leap/evo.png)
>
> Leap learns an initialisation that induces faster learning on tasks from the given task distribution. By minimising the distance we need to travel, we make tasks as ‘easy’ as possible to learn.

http://flennerhag.com/2019-05-09-transferring-knowledge-across-learning-processes/

https://arxiv.org/abs/1812.01054

**This means new modules should be born where they may be least specialized but have fasted mean predicted convergence speed**

- [Mind Your Manners! A Dataset and A Continual Learning Approach for Assessing Social Appropriateness of Robot Actions](https://arxiv.org/abs/2007.12506)
  - dataset of robot action appropriatenes
