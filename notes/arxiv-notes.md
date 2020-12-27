# arxiv-notes

Some relevant articles that I will apply towards AGI

## Papers I need to read:
- [Text Modular Networks: Learning to Decompose Tasks in the Language of Existing Models](https://arxiv.org/abs/2009.00751)
- [Second-order Neural Network Training Using Complex-step Directional Derivative](https://arxiv.org/abs/2009.07098)
- [Naive Artificial Intelligence](https://arxiv.org/abs/2009.02185)
- [Continual Prototype Evolution: Learning Online from Non-Stationary Data Streams](https://arxiv.org/abs/2009.00919)
- [Theory of Mind with Guilt Aversion Facilitates Cooperative Reinforcement Learning](https://arxiv.org/abs/2009.07445)
- [Presentation and Analysis of a Multimodal Dataset for Grounded LanguageLearning](https://arxiv.org/abs/2007.14987)
- [Goal-Aware Prediction: Learning to Model What Matters](https://arxiv.org/abs/2007.07170)
- [Tracking Emotions: Intrinsic Motivation Grounded on Multi-Level Prediction Error Dynamics](https://arxiv.org/abs/2007.14632)
- [DynamicEmbedding: Extending TensorFlow for Colossal-Scale Applications](https://www.arxiv-vanity.com/papers/2004.08366/)
- [Motivation and emotion Textbook](https://en.m.wikiversity.org/wiki/Motivation_and_emotion/)
- [Scale-Localized Abstract Reasoning](https://arxiv.org/abs/2009.09405)
- [From Static to Dynamic Node Embeddings](https://arxiv.org/abs/2009.10017)
- [Hierarchical Affordance Discovery using Intrinsic Motivation](https://arxiv.org/abs/2009.10968)
- [X-LXMERT: Paint, Caption and Answer Questions with Multi-Modal Transformers](https://arxiv.org/abs/2009.11278)
- [Message Passing for Hyper-Relational Knowledge Graphs](https://arxiv.org/abs/2009.10847)
- [Fuzzy Simplicial Networks: A Topology-Inspired Model to Improve Task Generalization in Few-shot Learning](https://arxiv.org/abs/2009.11253)
- [Lifelong Learning Dialogue Systems: Chatbots that Self-Learn On the Job](https://arxiv.org/abs/2009.10750)
- [The Illustrated SimCLR Framework](https://amitness.com/2020/03/illustrated-simclr/)
- [An AGI with Time-Inconsistent Preferences](https://arxiv.org/abs/1906.10536)
- [Towards Graph Representation Learning in Emergent Communication](https://arxiv.org/abs/2001.09063)
- [The Tensor Brain: Semantic Decoding for Perception and Memory](https://arxiv.org/abs/2001.11027)
- [A Machine Consciousness architecture based on Deep Learning and Gaussian Processes](https://arxiv.org/abs/2002.00509)
- [Goal-Directed Planning for Habituated Agents by Active Inference Using a Variational Recurrent Neural Network](https://arxiv.org/abs/2005.14656)
- [Wave Propagation of Visual Stimuli in Focus of Attention](https://arxiv.org/abs/2006.11035)
- [Graph Policy Network for Transferable Active Learning on Graphs](https://arxiv.org/abs/2006.13463)


- [Latent representation prediction networks](https://arxiv.org/abs/2009.09439)
  - learn representations that maximize prediction rollout accuracy instead of minimizing reconstructive loss

- [Entropy, Computing and Rationality](https://arxiv.org/abs/2009.10224)
  > Making decisions freely presupposes that there is some indeterminacy in the environment and in the decision making engine [. . .] Memory, perception, action and thought involve a level of indeterminacy and decision making may be free in such degree. 

- [A Review of Personality in Human‒Robot Interactions](https://arxiv.org/abs/2001.11777)
  > Why is analyzing personality important? "Theories of personality assert that individual human traits can be used to predict human emotions, cognitions and behaviors [...] “Personality traits” is a label to describe a specific set of characteristics that are believed to be the best predictors of an individual’s behavior"
  - give the robot a fitting personality for its role
  - important for "enjoyment, empathy, intelligence, social attraction, credibility and trust, perceived performance, and compliance"
  - personailty builds social connections
  - many physical robotic and behavioral properties affect personailty
  - general findings:
  > 1. Extraverts seem to respond more favorably when interacting with robots.
  > 2. Varying the robots behavior and vocal cues can invoke an extraverted personality.
  > 3. Humans respond more favorably to extravert-type robots, but this relationship is moderated.
  > 4. Humans respond favorably to robots with similar or different personalities from them.
  - considerations:
  1. include context in consideration
  2. get out of the lab
  3. try new tasks (a different task distribution)
  4. Big five isn't the only set of psychometrics

- :star:[THE NEXT BIG THING(S) IN UNSUPERVISED MACHINE LEARNING: FIVE LESSONS FROM INFANT LEARNING](https://arxiv.org/abs/2009.08497)
  1) "Babies’ information processing is guided and constrained from birth"
    - its okay to be feature engineer
  2) "Babies are learning statistical relations across diverse inputs"
    - eg: visual information serves as tiebreaker for otherwise impossible to distinguish audio inputs
    - even for SL tasks, carry some of the higher weights (theoretically processing more abstract information) over from different modalities
  3) "Babies’ input is scaffolded in time"
    - the effect of the "stability/plasticity dilemma" are *critical learning periods* and *catastrophic interference*.
  4) "Babies actively seek out learning opportunities"
    - arousal homeostasis: not too boring nor too alarming
    - effict: interested in familiar things until they are encoded sufficently, then attention shifts to novel things
  5) "Babies learn from other agents"
    - parents provide semi-suervised training
    - fellow babies promote social understanding
  - traditional machine learning has a lot of work to do. MADRL seems the way to go.

- [GRAC: Self-Guided and Self-Regularized Actor-Critic](https://arxiv.org/abs/2009.08973)
  - involved approach to combatting Q-value overestimation
  - take the best Q function actions, increase their likelihood on the policy, sample an action + alot of regularization like TRPO 

- [HTMRL: Biologically Plausible Reinforcement Learning with Hierarchical Temporal Memory](https://arxiv.org/pdf/2009.08880.pdf)
  - echoes of Jeff Hinton's work on sparse distributed representations and prediction.
  - draws attention to normalized (x-mu)/sigma reward instead of raw reward 

- [RLzoo](https://github.com/tensorlayer/RLzoo)
  - tf2 rl zoo

- [DECOUPLING REPRESENTATION LEARNING FROM REINFORCEMENT LEARNING](https://arxiv.org/abs/2009.08319)
  > In an effort to overcome limitations of reward-driven feature learning in deep reinforcement learning (RL) from images, we propose decoupling representation
learning from policy learning.
  - :star: They use intrinsic motivation for feature learning (perhaps described by section 3.3 of "Action and Perception as Divergence Minimization") while using extrinsic motivation for RL

- [A Neural-Symbolic Framework for Mental Simulation](https://arxiv.org/abs/2008.02356)
  - __TO READ__

- [A critical analysis of metrics used for measuring progress in artificial intelligence](https://arxiv.org/abs/2008.02577)
  - metrics on [Papers with Code](https://paperswithcode.com/) inadequately reflect classifier's performance " especially when used with imbalanced datasets"
  - reporting of (benchmark) metrics is "partly inconsistent and partly unspecific"

- [Explore then Execute: Adapting without Rewards via Factorized Meta-Reinforcement Learning](https://arxiv.org/abs/2008.02790)
  - decouple chicken and egg problem by independantly learning execution and exploration policies
  - DREAM exploration objective "identify\[s\] key information in the environment, independent of how this information will exactly be used solve the task"
  - "explores and consequently adapts to new environments, requiring *no reward signal* when the task is specified via an instruction"
  - "we allow each episode to have a different task provided via a different instruction"
  - the agent (in this case, a chef) evaluates the reward of a situation without a reward function
  - metalearning should be utilized to provide information to many related tasks-not just one
  - learn an exploration policy to recover the information from demonstrations and learn an execution policy (and take note of what information was necesary for that policy)

- [The Emergence of Adversarial Communication in Multi-Organism Reinforcement Learning](https://arxiv.org/abs/2008.02616)
  - use differentiable communication channels to accelerate learning and convergence
  - cooperative teams recover from selfish (but not malicious) influence with learnable *filter taps*
  - dynamicly constructed agregation graph neural network defines communication channels according to spatial inter-agent metrics
  - applies inverse communication signal encoder to each point in the 2D environment space to build whitebox analysis communication maps

- [CrowDEA: Multi-view Idea Prioritization with Crowds](https://arxiv.org/abs/2008.02354)
  - prioritized attention respecting a latent criterea (:star: human behavior optimizes a latent criterea: the 'heart')
  - evaluators do not share neither perspectives, criterea, nor values
  - frontier ideas maximize the common's interest
  - builds saliency matrix to interpret results

- [Compositional Networks Enable Systematic Generalization for Grounded Language Understanding](https://arxiv.org/abs/2008.02742)
  > Guided by the notion that compositionality is the central feature of human languages which deep
networks are failing to internalize, we construct a compositional deep network to guide the behavior
of robots \[. . .\] Given a command, a \[automatically discovered\] command-specific network
is assembled from previously-trained components \[. . .\] derived from the
linguistic structure of the command. In this way, the compositional structure of language is reflected in
the compositional structure of the computations executed by the network
  - generalizes compositional concepts on the gSCAN VP-NP dataset
  - replaces data augmentation with compositionality
  - semantic parsing over constiuency or dependency parsing
  - builds compositional RNN graphical network with each node associated with the lexicographical model

- [Forgetful Experience Replay in Hierarchical Reinforcement Learning from Demonstrations](https://arxiv.org/abs/2006.09939)
  - hierarchiel methods and expert demonstrations improve sample efficency
  - hierarchiel model extracts subgoals from sequence
  - Forgetful Experience Replay (ForgER) uses good expert samples to reach its subgoals
  - this allows focusing on good parts of expert demonstration while ignoring mistakes
  - wins MineRL competition (get diamond in MineCraft)

- [Semantic Visual Navigation by Watching YouTube Videos](https://arxiv.org/abs/2006.10034)
  - inverse model converts videos into episodes
  - off the shelf object detectors identify likelihood of objects in scene
  - Q-learns discounted likelihood of object from given starting point
  - :question: can the object detector learn by unsupervised clustering?
  - :star: beyond locating nearness likelihood of object, this approach could identify the likelihood of any perspective in a situation such as when a chess move 'feels right'
  ![](https://matthewchang.github.io/value-learning-from-videos/video-dqn-website_files/vfv.gif)
  - [website and video](https://matthewchang.github.io/value-learning-from-videos/)

- [ShieldNN: A Provably Safe NN Filter for Unsafe NN
Controllers](https://arxiv.org/abs/2006.09564)
  - use neural networks to ensure safety of other neural networks
  - :star: ensembles arrange NN's in parallel; ShieldNN introduces the concept of NN's in series. What about networks?

- [SAMBA: Safe Model-Based & Active Reinforcement Learning](https://arxiv.org/pdf/2006.09436.pdf)
  - model-based approach reduces sample cost
  - auxillary acquisition function samples safe yet uncertain points in the policy transition model 
  - models transitions by gaussian processes (GP's)
  - maximizes information gain by variance reductions in GP
  - biases policy samples to be close to the training set when exploring to maintain safety
  - unsafe states do not maximize information gain (the intrinsic objective) because of little sampling so they are not exploited

- [Automatic Curriculum Learning through Value Disagreement](https://arxiv.org/abs/2006.09641)
  > Our key insight is that if we can sample goals at the frontier of the set of goals that an agent is able to reach, it will provide a significantly stronger learning signal compared to randomly sampled goals. To operationalize this idea, we introduce a goal proposal module that prioritizes goals that maximize the epistemic uncertainty of the Q-function of the policy.

- [Getting Artificial Neural Networks Closer to Animal Brains](https://maraoz.com/2020/07/12/brains-vs-anns/)
  - creative approach to analyzing neural network architecture from a biological perspective

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

- [Approximate Simulation for Template-Based Whole-Body Control](https://arxiv.org/abs/2006.09921)
  - don't model your robot with templates (double inverted pendulum) but model the real deal. This requires less parameter tuning
  - mathematical formulation for humanoid kinematics

- [If I Hear You Correctly: Building and Evaluating Interview Chatbots with Active Listening Skills](https://arxiv.org/abs/2002.01862)

- [ON THE INTERACTION BETWEEN SUPERVISION AND SELF-PLAY IN EMERGENT COMMUNICATION](https://arxiv.org/abs/2002.01093)
  > first training agents via supervised learning on human data followed by self-play outperforms the converse, suggesting that it is not beneficial to emerge languages from scratch. \[. . .\] population based approaches to S2P [supervised self-play] further improves the performance over single-agent methods.
  
- [EgoMap: Projective mapping and structured egocentric memory for Deep RL](https://arxiv.org/abs/2002.02286)
  - builds grid of memory embeddings as agent travels
  

- [Mind Your Manners! A Dataset and A Continual Learning Approach for Assessing Social Appropriateness of Robot Actions](https://arxiv.org/abs/2007.12506)
  - dataset of robot action appropriatenes

[FinTech](https://arxiv.org/pdf/2007.12681.pdf)
[From Robotic Process Automation to Intelligent Process Automation](https://arxiv.org/abs/2007.13257)
  - general overview of IPA
[A Conversational Digital Assistant for Intelligent Process Automation](https://arxiv.org/abs/2007.13256)
  - many same authors from above
  - combination of IPA + RPA
