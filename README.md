# Artificial General Intelligence

Please visit [jacobfv.github.io/AGI](https://jacobfv.github.io/AGI) for more info


## Major refactoring ahead

- restructure/factor project as follows

All objects are stateless except the `policy` whose state is kept track
of by `tfa.networks.*PolicyNetwork`. The state is packed and unpacked
between iterations (but this does not really matter after tf-conversion)

Inside the agent, all environment observations and actions are represented
as a dictionary. `Agent.obs2dict` and `Agent.dict2act` convert between formats.
Often, these are identity functions.

```
convenience_f_act(f_abs):
    builds a nondifferentiable layer that
    backpropagates through f_abs to get the
    gradients for x. Those gradients are
    scaled to become x_target and z_{t+1}

policy._action(time_step: Timestep, policy_state: NestedArray) -> action: PolicyStep:
    obs = self.obs2dict(time_step.observation)
    states, targets = policy_state
    states = states.update(obs)
    
    """bottom up perception:
    1. organs update their state from environment observations
        energy consuming organs get an update on the amount of available energy 
        from their suppliers and update their internal energy balance.        
        Suppliers do NOT tax children if last round's bill exceed available energy,
        instead the suppliers learn to tax THEIR parents more frequently. 
    2. PredNodes update their internal state"""
    for organ in self.organs:
        parent_states = subdict(d=states[organ.name], ks=organ.parent_names)
        states[organ.name] = organ.bottom_up(parent_states, self_state)
    
    """forward propagation:
    InfoNodes interact with their neighbors. Here:
    1. organs perform logical updates like healing/adapting
    2. PredNodes perform neighbor state space clustering"""
    for organ in self.organs:
        neighbor_states = subdict(d=states[organ.name], ks=organ.neighbor_names)
        self_target, state[organ.name] = organ.forward(neighbor_states, self_state)
        targets[organ.name].append(self_target)
        
    """top down action:
    1. PredNodes message target controllable latents to their children
    2. PredNode send direct environment action outputs or talk to motor organs
    3. Organs output action to environment after performing energy consumption logic
        energy consuming organs send the bill of how much energy they spent back to their suppliers 
     """
    for organ in self.organs.reverse():
        new_targets, states[organ.name] = organ.top_down(targets[organ.name], states[organ.name])
        targets[organ.name] = list()
        for k,v in new_targets:
            targets[k].append(v)
    
    actions, targets = self.dict2act(targets) # structure action and clear lists for env actions
    return actions, (states, targets), None, None
    
======================TODO=========================
======================below========================
Agent.train(traj_batch):
    loss = 0
    for organ_name, organ_traj in traj_batch.items():
        loss = loss + organs[organ_name].train(organ_traj)
        
Organ.train(traj_batch):
    varies for each subclass
    return loss
    
InfoNode.train(traj_batch):
    varies for each subclass
    
PredNode.train(traj_batch):
    if f_abs.trainable or f_pred.trainable: 
        # TODO maximize predictability
    if f_act.trainable:
        # TODO reverse learn from f_abs
    if any(f_trans+[f_abs], lambda x: x.trainable) :
        # TODO miximize predicted latent similarity but
        # only for functions mapping to the top cluster
        
SubclassedOrgan.bottom_up(inputs, state):
    hid = f(states)
    return super(SubclassedOrgan, self).bottom_up(hid)
    
Organ.bottom_up(states):
    for info_node in self.info_nodes:
        states.update(info_node.bottom_up(states))
    return states

SubclassedOrgan.top_down(states):
    hid = f(states)
    return super(SubclassedOrgan, self).bottom_up(hid)
    
Organ.bottom_up(states):
    for info_node in self.info_nodes:
        states.update(info_node.bottom_up(states))
    return states
==========================above=============================
==========================TODO===========================


Agent: tfa.Agent
- name: str
- train: (traj: Trajectory) -> loss: Tensor # nonidempotent. also trains its policy
- policy: tfa.Network
  - obs2dict: (observation: NestedTensor) -> dict[str, NestedTensor]
  - dict2act: (action: dict[str, NestedTensor]) -> NestedTensor
  - organs: dict[str,Organ]
  - _action: (time_step:Timestep, policy_state:NestedArray) -> action:PolicyStep
  - _blank_targets_dict: dict[str,list[NestedTensor]] # empty list to init the targets_dict

InfoNode
- name: str
- parent_names: list[str]
- neighbor_names: list[str]
- child_names: list[str]
- bottom_up: (parent_states: dict[str,NestedTensor], self_state:NestedTensor) -> self_state: NestedTensor
- forward: (neighbor_states: dict[str,NestedTensor], self_state: NestedTensor) -> (self_pred_state: NestedTensor, self_state: NestedTensor)
- top_down: (targets: list[NestedTensor], self_state: NestedTensor) -> (parent_targets: dict[str,NestedTensor], self_state: NestedTensor)
- controllability_mask: Optional[NestedTensor]
- trainable : bool
: PredNode
  - f_abs
  - f_pred
  - f_act
  : DQNNode
  : RewardNode
: Organ : StatelessNode
  - info_nodes : list[InfoNode]
  : Brain
  : NavigableModality : abc.ABC
    "this superclass keeps track of data and loc in its recurrent state
    - interactive: bool # can the agent write values to its audio stream, text tape, or image imagination?
    - f_window : data:nest[Tensor], loc:nest[Tensor] -> subset:nest[Tensor]
    - info_nodes = subset_info_nodes + loc_info_nodes : list[InfoNode]
    - subset_info_nodes : list[InfoNode]
    - loc_info_nodes : list[InfoNode]
    : NavigableSequence
      - loc_info_nodes
        . TODO
      : NavigableText
        - subset_info_nodes
          . obs1 = LambdaInfoNode(gpt2 base)
          . act1 = LambdaInfoNode(gpt2 LM head)
          . obs2 = LambdaInfoNode(T5 base)
          . act2 = LambdaInfoNode(T5 LM head)
          . obs3 = LambdaInfoNode(Blender base)
          . act3 = LambdaInfoNode(Blender LM head)
      : NavigableAudio
        - subset_info_nodes
          . obs1 = LambdaInfoNode(wav2vec 2.0)
          . act1 = LambdaInfoNode(Tacotron 2)
    : NavigableGraph
      : NavigableGrid
        : NavigableImage
  : TDWBody
  : TDWStickyMitten
  : EnergyNode
  : EnergyVessel
  : EnergyReservoir
  : EnergyConverter
  
: GymSpaces
  - type: 'sensor'||'actuator'
  - key: str
  : Box
  : Discreet
  : MultiBinary
  : MultiDiscreet
```
