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

All InfoNodes include `self_state['cost']` in their internal_state. This helps train
the reward system and prioritize training. Most InfoNodes initially set their cost on
`bottom_up` and update it during `forward` and `top_down`. Afterwards, it is recorded
as the final cost for that step.

EnergyNodes present energy in the form of a dense vector. Not all components are useful
to all organs however. Some energy vector components even have bad effects just like the
chemical profile of the bloodstream.

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
    for info_node in self.info_nodes:
        parent_states = subdict(d=states, ks=info_node.parent_names)
        states[info_node.name] = info_node.bottom_up(parent_states, self_state)
    
    """forward propagation:
    InfoNodes interact with their neighbors. Here:
    1. organs perform logical updates like healing/adapting
    2. PredNodes perform neighbor state space clustering"""
    for info_node in self.info_nodes:
        neighbor_states = subdict(d=states, ks=info_node.neighbor_names)
        self_target, state[info_node.name] = info_node.forward(neighbor_states, self_state)
        targets[info_node.name].append(self_target)
        
    """top down action:
    1. PredNodes message target controllable latents to their children
    2. PredNode send direct environment action outputs or talk to motor organs
    3. Organs output action to environment after performing energy consumption logic
        energy consuming organs send the bill of how much energy they spent back to their suppliers 
     """
    for info_node in self.info_nodes.reverse():
        new_targets, states[info_node.name] = info_node.top_down(targets[info_node.name], states[info_node.name])
        targets[info_node.name] = list()
        for k, v in new_targets:
            targets[k].append(v)
    
    # tabulate intrinsic reward
    reward = -sum(state['cost'] for state in states.values())
    
    # structure action and erase env actions in `targets`
    actions, targets = self.dict2act(targets)
    
    return actions, (states, targets), reward, None
    
Agent.train(traj_batch):
    """trains on a batch of trajectories.
    Elements in the batch are purposely unordered
    They are sampled porportional to negative reward
    """
    cost = 0
    
    # train individual organ
    for organ_name, organ_traj in traj_batch.items():
        cost = cost + organs[organ_name].train(organ_traj)
        
    return -cost
        
InfoNode.train(traj_batch):
    varies for each subclass
    return 0
    
Organ.train(traj_batch):
    varies for each subclass
    return 0
    
PredNode.train(traj_batch):
    cost = 0.
    # incrementally increase cost porportionally to
    # the amount of learning that took place
    # It'd be awsome if I coulddirectly penalize 
    # cost += KL[weights_new|weights_old]
    if f_abs.trainable or f_pred.trainable: 
        # TODO maximize predictability
    if f_act.trainable:
        # TODO reverse learn from f_abs
    if any(f_trans+[f_abs], lambda x: x.trainable) :
        # TODO miximize predicted latent similarity but
        # only for functions mapping to the top cluster
    # TODO: somehow compute cost
    return cost

InfoNode.bottom_up(parent_states: dict[str,NestedTensor], self_state:NestedTensor) -> self_state: NestedTensor:
    # reset (or don't continually add to) self_state['cost']
    return self_state
    
InfoNode.forward(neighbor_states: dict[str,NestedTensor], self_state: NestedTensor) -> self_state: NestedTensor:
    return self_state
    
InfoNode.top_down(targets: list[NestedTensor], self_state: NestedTensor) -> (parent_targets: dict[str,NestedTensor], self_state: NestedTensor)
    # self_state['cost'] should be finalize by the end of the method
    return dict(), self_state


PredNode.bottom_up(parent_states: dict[str,NestedTensor], self_state:NestedTensor) -> self_state: NestedTensor:
    self_state['z'] = self.f_abs(x=parent_states, z=self_state['z_prev']) - self_state['z_pred']
    self_state['free_energy'] = tf.reduce_sum(self_state['z']) # I would've preferred KL[Z|Z_pred]. This can happen at training
    return self_state

PredNode.forward(neighbor_states: dict[str,NestedTensor], self_state: NestedTensor) -> self_state: NestedTensor:
    # get neighbor latents
    z_neighbors = [s['z'] for s in neighbor_states.values()]
    z_all = z_neighbors + self_state['z']
    
    # conditional neighbor latent broadcasting
    # simple:
    if in 25-75 percentile: move self_state['z'] to 50th percentile
    # complex:
    cluster = fit_to_normal(data=z_all)
    delta_z = 1 / ((cluster.mean - self_state['z'])**2
                    * cluster.log_prob(self_state['z'] * cluster.variance)
    self_state['z'] += delta_z
    
    # make prediction
    self_state['z_pred'] = self.f_pred(z=self_state['z'])
    
    return self_state

PredNode.top_down(targets: list[NestedTensor], self_state: NestedTensor) -> (parent_targets: dict[str,NestedTensor], self_state: NestedTensor)
    target = random_sample(targets) # NOT mean
    parent_targets = self.f_act(z_target=mean_target, z=self_state['z'])
    return parent_targets, self_state


RewardNode.top_down(targets: list[NestedTensor], self_state: NestedTensor) -> (parent_targets: dict[str,NestedTensor], self_state: NestedTensor)
    targets.append(self.const_set_point)
    return super(RewardNode, self).top_down(*args)
    

Organ.bottom_up(parent_states: dict[str,NestedTensor], self_state:NestedTensor) -> self_state: NestedTensor:
    # use this function for env2organ interaction. save organ2organ for `forward`
    
    # we canonot actually take more energy than is available
    # NOTE: multiple organs might simultaneously ask for an amount that
    # sums to be greator than the energy node's available energy. While
    # there is momentary 'free'/imaginary energy in existance, this is quickly
    # exacted on the next frame step and optimzed against during training 
    recvd_energy = tf.clip(self_state['cost'],
                           0, parent_states[self.energy_node_name]['energy'])
    self_state['cost'] = self_state['cost'] - recvd_energy # start the energy bill with any energy not available last round
    self_state['energy'] = self_state['energy'] + recvd_energy
    
    # example of updating internal state from env observation
    self_state['internal_obs_state'] = somefunc(parent_states['env_obs1'], self_state['internal_obs_state'])
    return self_state
    
Organ.forward(neighbor_states: dict[str,NestedTensor], self_state: NestedTensor) -> self_state: NestedTensor:
    # use this function for organ2organ interaction
    # any neighbor interaction logic here
    self_state['internal_obs_state2'] = somefunc(neighbor_states['liver']['horomoneX'],
                                                 self_state['internal_obs_state2'])
    return self_state
    
Organ.top_down(targets: list[NestedTensor], self_state: NestedTensor) -> (parent_targets: dict[str,NestedTensor], self_state: NestedTensor)
    target = random_sample(targets) # energy organs take the sum rather than random sample
    amount_to_move = target[0]
    energy_demanded = tf.sigm(self_state['energy'] * amount_to_move)
    energy_spent = tf.clip(energy_demanded, 0, self_state['energy'])
    energy_gained = 0 # mouth, energy_reservoir, and a few others actually gain energy
    
    self_state['action1'] = energy_spent # child InfoNodes will read this data
    self_state['cost'] = energy_spent - energy_gained # positive => need more energy
    
    parent_targets = {
        self.energy_node_name: self_state['cost'],
        'env_controller_motion': 'left' if self_state['activity'] > 0.5 else 'right' # example of env action
    }
    
    return parent_targets, self_state
    
    
EnergyNode.bottom_up(parent_states: dict[str,NestedTensor], self_state:NestedTensor) -> self_state: NestedTensor:
    return self_state
    
EnergyNode.forward(neighbor_states: dict[str,NestedTensor], self_state: NestedTensor) -> self_state: NestedTensor:
    # some equalibrium matrix. By default, this is an identity matrix
    self_state['energy'] = M_energy_transitions @ self_state['energy'] 
    return self_state
    
EnergyNode.top_down(targets: list[NestedTensor], self_state: NestedTensor) -> (parent_targets: dict[str,NestedTensor], self_state: NestedTensor):
    consumed_energy = sum(target['cost'] for target in targets)
    self_state['energy'] = self_state['energy'] - consumed_cost
    # somehow get energy (either behave like regular organs and backpropagate
    # the demand or convert it from another energy form. ie: energyA <-> energyB
    
    # make sure to set cost even for energy node
    self_state['cost'] = -negative_amount_of(self_state['energy'])
    
    parent_targets = dict()
    return parent_targets, self_state
    
    
EnergyNode.bottom_up(parent_states: dict[str,NestedTensor], self_state:NestedTensor) -> self_state: NestedTensor:
    # we canonot actually take more energy than is available
    # NOTE: multiple organs might simultaneously ask for an amount that
    # sums to be greator than the energy node's available energy. While
    # there is momentary 'free'/imaginary energy in existance, this is quickly
    # exacted on the next frame step and optimzed against during training 
    recvd_energy = tf.clip(self_state['cost'],
                           0, parent_states[self.energy_node_name]['energy'])
    # start the energy bill with any energy not available last round
    # by not completely reassigning the 'cost' entry, we keep track of unpaid
    # costs from multiple rounds
    self_state['cost'] = self_state['cost'] - recvd_energy 
    self_state['energy'] = self_state['energy'] + recvd_energy
    return self_state
    
EnergyVessel.top_down(targets: list[NestedTensor], self_state: NestedTensor) -> (parent_targets: dict[str,NestedTensor], self_state: NestedTensor):
    parent_targets, self_state = super(EnergyVessel, self).top_down(targets, self_state)
    parent_targets.update({self.energy_node_name: self_state['cost']})
    return parent_targets, self_state


Agent: TFAgent
- __init__: (info_nodes: list[InfoNode]) -> None
- name: str
- train: (traj: Trajectory) -> loss_info: LossInfo # nonidempotent. also trains its policy
- policy: TFPolicy
  - obs2dict: (observation: NestedTensor) -> dict[str, NestedTensor]
  - dict2act: (targets: dict[str, NestedTensor]) -> NestedTensor
  - info_nodes: dict[str,InfoNode]
  - _action: (time_step: Timestep, policy_state: NestedArray) -> action: PolicyStep
  - get_initial_state (batch_size: Optional[Int]) -> NestedTensor

Trajectory: Dataset[NestedTensor]

InfoNode
- name: str
- parent_names: list[str]
- neighbor_names: list[str]
- child_names: list[str]
- bottom_up: (parent_states: dict[str,NestedTensor], self_state:NestedTensor) -> self_state: NestedTensor
- forward: (neighbor_states: dict[str,NestedTensor], self_state: NestedTensor) -> (self_pred_state: NestedTensor, self_state: NestedTensor)
- top_down: (targets: list[NestedTensor], self_state: NestedTensor) -> (parent_targets: dict[str,NestedTensor], self_state: NestedTensor)
- controllability_mask: Optional[NestedTensor]
- trainable: bool
- train: (traj: NestedTensor) -> loss: float
: PredNode
  - __init__(f_abs, f_pred, f_act, name) -> None
  - f_trans: dict[str, Callable[[NestedTensor], NestedTensor]]
  - f_abs: (parents: dict[str,NestedTensor], self_state: NestedTensor) -> self_state: NestedTensor 
  - f_pred: (self_state: NestedTensor) -> self_state: NestedTensor
  - f_act: (targets: list[NestedTensor], self_state: NestedTensor) -> (parent_targets: dict[str,NestedTensor], self_state: NestedTensor)
  - train # complete override
  : DQNNode
    - train # complete override
  : RewardNode
    - top_down # same signature, but adds a strong bias target before calling super.top_down
    
: Organ
  - energy_node_name: str
  - _info_nodes: dict[str,InfoNode]
  : Brain
  : NavigableModality
    "this superclass keeps track of data and loc in its recurrent state
    - interactive: bool # can the agent write values to its audio stream, text tape, or image imagination?
    - f_window: (data: NestedTensor, loc: NestedTensor) -> subset: NestedTensor
    - @property _info_nodes: (self) -> list[InfoNode] = subset_info_nodes + loc_info_nodes
    - data_info_nodes: list[InfoNode]
    - loc_info_nodes: list[InfoNode]
    : NavigableSequence
      - loc_info_nodes
        . navigator: GridNavigatorNode
      : NavigableText
        - data_info_nodes
          . obs1 = LambdaInfoNode(gpt2 base)
          . act1 = LambdaInfoNode(gpt2 LM head)
          . obs2 = LambdaInfoNode(T5 base)
          . act2 = LambdaInfoNode(T5 LM head)
          . obs3 = LambdaInfoNode(Blender base)
          . act3 = LambdaInfoNode(Blender LM head)
      : NavigableAudio
        - data_info_nodes
          . obs1 = LambdaInfoNode(wav2vec 2.0)
          . act1 = LambdaInfoNode(Tacotron 2)
    : NavigableGrid
      - loc_info_nodes
        . navigator: GridNavigatorNode
      : NavigableImage
    : NavigableGraph = NotImplemented
  : TDWBody
  : TDWStickyMitten
  : EnergyNode
    : EnergyVessel
    : EnergyReservoir
: GymSpaceNode
  - type: 'sensor'||'actuator'
  - key: str
  : BoxSpaceNode
  : DiscreetSpaceNode
  : MultiBinarySpaceNode
  : MultiDiscreetSpaceNode

```
