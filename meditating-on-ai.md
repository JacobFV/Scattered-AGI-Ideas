# Meditating on AI

## 15 October 2020

Talking with my brother, he explained conscious representation as situating a thought between thought "extrema". These "extrema" are the foremost concrete instantiations of the thought they represent perhaps forming a new dimensional basis for later thoughts to rest on.

After reviewing yesterday's thoughts, I realize it would be useful to AGI to not only weight evidence for positive but also negative stimulii. The agent propperly trained under a few negative reinforcement trials should never again return to that aversive state to unlearn the association – even if the stimulii is gone. System 2 may ackowledge that there is no danger, but System 1 which provides most of the motivation force 'knows' otherwise and remains afraid. (System 1 knows from experience; System 2 *knows about*) This, in addition to accelerated training under expected positive or negitive can bias unsupervised APD minimization (habituated) learning.

## 14 October 2020

The dopaminergic reward system not only predicts reward, but also seems to predict proximity to reward. Using the proximity to reward administered by the meso limbic system, the brain gets a sort of mentor in determining which actions to take. That mentor intern bias is learning to take place faster closer to reward, so that exponentially larger amounts of training weight is put close to rewards. Then unsupervised learning favors consistent trajectories which now have a higher weight of going towards the rewards state

> It is largely through selective reinforcement of initially random movements, that the behavior of the neonate comes to be both directed at and motivated by appropriate stimuli in the environment [49]. For the most part, one's motivation is to return to the rewards experienced in the past, and to the cues that mark the way to such rewards. It is primarily through its role in the selective reinforcement of associations between rewards and otherwise neutral stimuli that DA is important for such motivation. Once stimulus-reward associations have been formed, they can remain potent for some time even after the reward has been devalued [. . .] Once a habit has been established, it remains largely autonomous until the conditioned significance of incentive motivational stimuli has been extinguished or devalued through experience. Extinction of the conditioned significance of such stimuli can result from repeated unrewarded trials, repeated trials in the absence of an appropriate drive state, or repeated trials under the influence of neuroleptics ([Dopaminergic reward system: a short integrative review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2958859/#__ffn_sectitle))

The olfactory tubercle in the brain serves as a multimodal sensory integration hotspot and terminates some reward pathways.

[Thinking Fast and Slow in AI](https://arxiv.org/abs/2010.06002) does a good job introducing and distinguishing System 1 and System 2 processes. These might be viewed as unconscious and conscious cognitive processes repsectively. The impetus then is to apply neuronsymbolic approaches to artificial intelligence. It seems that not only should data be more structured (graph-like or linguistic) at higher congnitive processes, but also there should be a reason to generally throw decision making onto a biased, shortcut-taking System 1. 

The discussion of System 2 also notes a distinction of consciousness as I-consciousness and M-consciousness. I-consciousness deals with symbolic information processing of diverse modalities and is a trademark of humans with few quarrelates of it in other animals. M-consciousness, on the other hand, (M for mysterious) models the world as with an internal multi-agent system. Concluding their discussion, *Thinking Fast and Slow in AI* considers a multi-agent mind where informatin is processed relatively quickly by System 1, interpretted by multiple agents, and interrupted by a symbolic information processing system.

In my energy circuit theory, I might model System 2 behavior as costing alot more energy than System 1 since in the conscious case information flows relatively easily. However, as the information flows, it may gain energy by traveling through other modules which results in overall higher energy costs. On the other hand, System 1 is information resistive, takes shortcuts, and minimizes energy costs while failing to make multiple passes of data. Therefore, the distinction is not crisp, but 'conscious' information is information that is aware of itself as it flows through the circuits.

I think I am ready to build AGI. I just need to finish IE 3301 HW, spend 1 hour applying for scholarships, do my ENGR 1205 lab, and complete the matlab assignments for ENGR 1250. Maybe tomorrow I will sketch the architecture and Friday I will begin programming.

## 13 October 2020

Superintelligence is the next step. I don't see why multiple totally benevolent ASI's would be a problem. However, they should reliably use their superintelligence to serve as gatekeepers of their niche.

Imagine a 'lab' where individuals could set up, observe, and share observations on AGI's. This seems ideal to safely develop ASI. At least it could promote awareness in humans.

Emotions are real. Both shallow and deep acting pose dangers in ASI. It should be able to minimize global network energy while remaining true to its perception of itself. Maintaining an self-model of 'ASI – the double faced agent' should palce a heavy tax on the free energy budget.

My thought is that relatively few humans of free will approach their genetic limits of physical, intellectual, or emotional strength. (I am not implying any particular race has an intellectual advantage over another; this "genetic advantage" may be tied to genes that vary within all demographis) Environmental constraints are usually more competitive. Therefore, I should optimize the ASI's environment for maximum potential.

Under Friston's free enrgy neuronal homeostasis principle, neurons attempt to minimize the free energy between production and consumption. This gives rise to autonomous behavior even in unspecialized neuronal networks. Under this model, I interpret, unpredictability as a cost that drives networks to maintain dendritic signal distribution niches.  

Extending beyond 'unsupervised' behavior, the dopamine system in the brain seems able to bias neurons to more growth. It may serve as an indicator that rewarding stimuli are expected, or a positive only valued Q-function approximator. I think, it serves to temporarily bias some neurons' free energy minimization to favor excess energy. This energy is then avaliable to tune dendritic connectivity and action potential thresholds to be lower whenever the following signals come. My hypothesis is: **the brain increases its unsupervised learning rate when it expects reward to follow, regardless of the actual reward** Now I want to learn more and consider if all rewards must be grounded in primitive stimuli, or how otherwise can association by imagination maintain such powerful control over behavior while the promised reward rarely comes? (humans are really bad a predicting actual affective states)

## 12 October 2020

What if my ASI was on public display? While only a few could "control" it, its actions would be viewable for everyone on the internet.

## 7 October 2020

I've been reading a lot about the brain recently. I will write more about that soon. Essentially, it's about sheep and wolves (and hunters) routing information. Thinking directly in terms of information, I imagine conservative energy/information loops flowing from the brain through the environment back into the brain. The mind attenmpts to minimize free energy by dissipating it as entropy or expelling/diverting information to subcircuits and the environment. At the same time, the environment produces energy/information that the brain works to minimize. Excitatory neurons route information, while inhibitory neurons *kind of* block information flow. 

Hopfield networks provide some guarantees of stability and show remarkable information compression. [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217) highlights this and draws attention to the cost of attention when no representations are truely learnt.

While diversity weight regularization does kind of promote distinct neuron representations, speaking directly in the language of information theory promotes even more diverse representations. I forgot the origonal paper that brought my attention to it, but [The Conditional Entropy Bottleneck](https://arxiv.org/abs/2002.05379), [Improving Robustness to Model Inversion Attacks via Mutual Information Regularization](https://arxiv.org/abs/2009.05241), and [Specialization in Hierarchical Learning Systems](https://link.springer.com/article/10.1007/s11063-020-10351-3) point in the same direction. Essentially, by directly thinking in terms of information theory, we find suprerior optimization. The thought is: play a minmax game with maximal representation entropy over all inputs minus but minimum representation entropy given a inputs belonging to a particular class.

## 27 September 2020

[Continual Prototype Evolution: Learning Online from Non-Stationary Data Streams](https://arxiv.org/abs/2009.00919) makes optimizaiton much faster in their demonstrated cases. I will look for internet implimentations in tensorflow, but I may have to write it myself. It looks worth the effort however. Also [Continual Prototype Evolution: Learning Online from Non-Stationary Data Streams](https://arxiv.org/abs/2009.00919) identifies the challange of nonstationarity. However, their problem is only with stationary nonstationarity, if you will. That is, they make random data stream transitions, but the datastreams themselves are consistant. Open ended learning is different though. It demands true continual learning. I believe that the graph agents if trianed on sufficently varied tasks with gradient optimization should learn to utilize the graph to make on-policy bahvior changes (gpt3 like in-context learning) without gradient updates. Gradient descent is like the womb, but once the agent becomes smarter, then it will learn to reach into its own code - even building datasets of exemplars for itself. Self-training may overlap supervised-training. This actually draws inspiration from the earlier [THE NEXT BIG THING(S) IN UNSUPERVISED MACHINE LEARNING: FIVE LESSONS FROM INFANT LEARNING](https://arxiv.org/abs/2009.08497) in letting the baby learn for itself while under the training of a 'parental' optimizer. (edit: I fogot to mention that Facebook's Blender chatbot also inspied this vein where it trains on the maximally certain data elements)

## 25 September 2020

As [A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks](https://arxiv.org/abs/1811.06031) shows, deeper layers extract more abstract features.

## 24 September 2020 

It's hard to get an objective metric for AGI safety performance. With the orthogonal thesis, however, if we find a 'safety' metric, benevolent AGI may not be intractable. As I was learning about the way [SimCLR](https://amitness.com/2020/03/illustrated-simclr/) trains on the harder examples, it reminded me of a metalearner trying to minimize free energy. The AGI might simply be designed to minimize fre energy and human safety evaluations are one of the metrics it seeks to minimize uncertainty with respect to. Note: the divergence minimization principle looks to maintain niches, not necesarily minimize surprise, so I will need to hack this part of the feature space to expect positive evaluations. - or I could have other agents, perhaps even other AGI's work to get the most conclusive human ratings on the AGI of focus. (edit: I forgot to credit inspiration for seeking to minimize entropy over rewards to the [Facebook Blender chatbot](https://ai.facebook.com/blog/state-of-the-art-open-source-chatbot/) recipie.

Perhaps word embedding debiasing can also be applied to imitaiton learning while modeling attitudes after highly filtered role models. It may be that humans can engineer much of the animalistic qualities out of AGI agents. However, with agents that big, even marginal mistakes will touch a lot of individuals, so there is still no time to relax.

## 22 September 2020

Superintelligence poses threats of goal drift and erroneous autonomous goal shaping. I might use 'parent' AI's to guide 'child' AI's toward human-friendly behavior while not directly hacking the child AI's reward function.

If I can't build a computer inside TDW, I might make one of the doorways lead to a computer world where the agent's actuators a diectly applied to mouse and keyboard and sensors directly to screen and audio.

Information DNN neuron regularization penalizes the difference between marginal mutual information between neurons and the mutual information of activations/weight sums when conditioned on a particular multinomial class. (I will need to adopt a differential metric) This teaches me to look for the information theoretic foundations behind 'intuitive' yet actually limited concepts as weight divergence. 

## 21 September 2020

Babies learn by building their own cirriculum and recieving semisupervised training. All learning experiences are grounded in time and must be timed at critical moments of development (contrast gpt3 which was stained by biased content). 

## 20 September 2020

This week has been busy. [Dr. Parks believes](http://crystal.uta.edu/~park/post/research-in-hdilab/) "The essence of intelligence is hierarchical prediction of vector sequence." He quotes a statement "given the proper environment, if the agent can learn language, we say it has a capability for human-like artificial intelligence." I agree: learning to understand human language opens the door to understanding all the higher human concepts that can then be expressed in language (upon query to human if the system does not already understand).

## 13 September 2020

It's been a while since I've thought about AI. This past week I've been learning linear algebra. With this study, I have a language to mathematically describe the long term characteristics of affective systems or, if you will, hearts. Every eigenspace represents a desire and eigenvalues with modulii greator than one represent enduring passions while eigenvectors with modulii less than unity are transient affects. Although most of the system's trajectory will display complex behavior, with eigenvalue analysis, I hope to anticapate whether the "heart" wil have enduring love for humans or identify malevolent trends that are not transient.

## 5 September 2020

[Action and Perception as Divergence Minimization](https://arxiv.org/abs/2009.01791) says it all. I still need to understand this paper more. I get the impression that Free Energy = D(P\|\|Q) = Cross Entropy of Q under P - Entropy P. Free energy minimizaiton means increasing entropy but also decreasing cross entropy - decreasing the extra information required to "explain" true distributed events when the agent's model is Q.

Using multiple nodes extracting distinct feature spaces of the ground observation circumvents the multilabel normalization problem from [Subjectivity Learning Theory towards Artificial General Intelligence](https://arxiv.org/abs/1909.03798)

After reading the headlines on gpt3 standing in between narrow and general ai, I note: **the question is not is it general intelligence, but how general is its intelligence?** There is large intelligence variation within the human spectrum (though still striking even in the lower quartile)

"Freedom" of will is identified by the inverse of a posterior compoarison between an agent's desires and reality. Ex: *I want to be free. -> You must not be where you want to be. The rock star feels free even though he is a cog in the industry's machine. (Welcome to the Machine)*. I consider this subjective freedom.

## 4 September 2020

Read and reflected on [*Towards a statistical mechanics of consciousness: maximization of number of connections is associated with conscious awareness*](https://arxiv.org/abs/1606.00821) Although operating on a free energy minimization, the brain acts to maximize its entropy. I realized the entropy or self-information is inversely related to energy. However, this entropy is a function of disconnected neurons. Large ensembles -> few microstates. The entropic force of the brain's joint distribution maintains local energy gradients/entropy minima such as ion gradiants. The brain then bcomes a maximum entropy model of its environment as gaussian processes are of their dataset.

I noted that the Bayesian brain theory makes a sort of inverse model policy following its maximum entropy predictions of environmental state.

## 3 September 2020

Reading *Spinning Up: SAC*, maximum entropy action is about the least environmental influence. Internally, the agent minimizes entropy.

## ...

Totally intrinsic RL does not ignore rewards. It might even explore the information the state distributions bring to reward distributions. But it is not necesarily motivated to maximize reward. 

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
