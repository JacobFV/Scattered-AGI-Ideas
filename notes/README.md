# Thoughts

## 11 August 2020

*Read and reflected on task-mode network, saliency network, default mode network*

The saliency network "has been implicated in the detection and integration of emotional and sensory stimuli, as well as in modulating the switch between the internally directed cognition of the default mode network and the externally directed cognition of the central executive network" ([wikipedia](https://en.wikipedia.org/wiki/Salience_network#Function))

Are these networks two separate policies, one minimizing free energy by acting, another minimizing free energy by (re)modeling?

Also, [Saliency, switching, attention and control: a network model of insula function](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2899886/) presents the saliency network as making the transition from internal to external (ave. 20sec) focus relatively distinct. Can I introduce similar biases in the nodular (intrer or intra) network?

*Read and reflected on [dopaminergetic pathways](https://en.wikipedia.org/wiki/Dopaminergic_pathways#Function)*

The brain is an power consumption optimizing machine. It attempts to minimally explain the physiological reward (along with all other) signals recieved by something like Hebbian learning. The positive surprise of a reward is encoded by the frequency of dopaminergetic pathway activations - hance, energy consumption. When a neuron activates prior to recieving reward along dopaminergetic pathways, a synaptic connection may be strengthened. In attempting to maintain a constant expectation of reward, top-down biasing from the dopaminergetic pathways to the behavior pathways bias activations that locally activation gradients between the pathways and globally optimize neural activations to approach the true reward represented by the latent variable that the dopaminergetic pathways signal.

This model of both local and global energy minimization teaches every pathway and neuron connected to the dopaminergetic pathways what reward to expect. It explains why a brain accostomed to a particular degree of reward demands so much change when physiological reward changes (as in breaking addictions). This model does not explain why the brain seeks out more reward than it is accustomed to. Maybe it does not.

If reward pathways are the critic then they must follow - not preceed - the actor. Since the critic models deopamine receptors, it cannot actually be changed by bottom up activations. But the graphical model attempts to view things that way. This is the case where some observables and actions (external senses and behaviors) explain another observable (internal dopamine).

Considered the free energy minimization more, the model makes dopamine prediction more important than external information. Hence, the brain primarily models dopamine as an effect rather than a cause. I believe the this may be so if reward error is usually greator than other predictive modeling errors.

I could test this hypothesis by observing the self-organized arrangement of graphical nodes and testing if the divergance is positive leading to the reward node. But how can I make reward predictive errors cost more than any other predictive error? Are dense rewards the answer? Should I add a `cost_of_error` hyperparemter to each node? I know, I will encode this information in the intrinsic `weight` of the reward node so that every connect with that node shares the weight variable. 

This seems to affect large behavioral decisions, but what about smaller ones? They are not very informative to global reward and so learn their own unsupervised, locally free-energy minimizing synaptic strengths and activation potentials.

## 10 August 2020

*Compiled* more relevant papers from arxiv. Notable ones:
- *Explore then Execute: Adapting without Rewards via Factorized Meta-Reinforcement Learning* encourages viewing exploration as trying to identify the information from a scene that guided an execution policy
- *Compositional Networks Enable Systematic Generalization for Grounded Language Understanding* makes an RNN network out of lexicographically structured graphical probablistic models. I think this support my model to a nodular network

*I feel* intellect performance varies with the richness of developmental environment and the human environment is most rich of all for agents that interact in the human behavior space.

*Learned about* [ThreeD World](http://threedworld.org/)
  - Supports rigid body, multibody, constrained body, soft body, cloth, and fluid simulation
  - acoustic impact generation
  - many builtin environments, models, and procedural generation tools
  - Unity on the back end, so I may be able to write my own custom elements (like computer)
  - Windows required for advanced physics, but I can offload rendering

*Problem*: After installing with pip, I was able to run a tutorial. However, later vscode installed pylint. Then I could not get it to work. I think either that or failing to `terminate` connections are to blame. I will investigate this more tomorrow.

*To think about* robotic interface
  - 2D image sensor
  - what kind of body? humanoid? are the physics real?

## 1 August 2020

### Digital Heart

Simplifying affective states as a hidden variable in the latent dynamics of every cortical column, the digital heart is coallescing emotive system unifying various domains of processing and nondifferentiably influencing the physiological state which, in turn, biases activation patterns. I still need to rigorously formulate this concept mathematically and detirmine to what extent affective state should be policy vs. environmentally controlled. 
