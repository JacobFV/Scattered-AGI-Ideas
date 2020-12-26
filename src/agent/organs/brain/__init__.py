from ... import utils
from .. import NodeOrgan


class Brain(NodeOrgan):

    def __init__(self, **kwargs):
        """
        params:
        name
        agent
        """

        kwargs['freezables'] = self.get_regions
        kwargs['trainables'] = self.get_regions
        kwargs['stepables'] = self.get_regions
        kwargs['name'] = f"{kwargs['agent']}_brain"

        super(Brain, self).__init__(**kwargs)

        # PERIPHERAL REGIONS
        # TODO all code in this block
        self.auditory_perception = None # spectrograph processing
        self.auditory_generation = None # spectrograph generation
        self.monocular_visual_perception = None # single image
        self.joint_visual_perception = None # set of input images
        self.energovascular_cortex = None # top-down biasing for graph of all energy-aware information
        # energy converting, energy storing, energy transfer, lights, energy vessel, energy pump, and
        # energy availability at energy nodes along the graph
        self.gustatory_cortex = None # perception of contact for "taste"
        self.somatosensory_cortex = None # perception of touch, joints, skeleton, muscle, pain, muscle tension
        self.motor_cortex = None # high-level muscle output
        self.cerebellum = None # graph-aware individual muscle outputs with absolute and derivative control granularity

        self.peripheral_regions = [
            self.auditory_perception,
            self.auditory_generation,
            self.monocular_visual_perception,
            self.joint_visual_perception,
            self.energovascular_cortex,
            self.gustatory_cortex,
            self.somatosensory_cortex,
            self.motor_cortex,
            self.cerebellum
        ]

        # HIGHER REGIONS
        # TODO all code in this block
        self.language_understanding_cortex = None # language integrator
        self.language_generation_cortex = None # language differentiator
        self.neocortex = None # graph of cortical columns
        # with overall gradient of peripheral_region connection across the neocortex
        self.allostatic_regulator = None # detirmine if nocioceptive, free-energy related stimulii are present
        self.reward_system = None # bias consciousness global workspace temperature

        self.global_workspace_regions = [
            self.language_understanding_cortex,
            self.language_generation_cortex,
            self.neocortex,
            self.allostatic_regulator,
            self.reward_system
        ] + self.peripheral_regions
        self.multimodal_translator = dict() # TODO

        self._regions = self.global_workspace_regions

    @property
    def get_regions(self):
        return self._regions

    def step(self):
        # pre-ops
        super(Brain, self).step()
        # post-ops

    def train(self):
        # pre-ops
        super(Brain, self).train()
        # post-ops

    def freeze(self):
        # pre-ops
        super(Brain, self).freeze(dir)
        # post-ops

    def unfreeze(self):
        # pre-ops
        super(Brain, self).unfreeze(dir)
        # post-ops
