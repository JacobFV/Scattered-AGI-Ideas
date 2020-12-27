from ... import utils
from .. import NodeOrgan


class Brain(NodeOrgan):

    def __init__(self, **kwargs):
        """
        params:
        name
        organism
        """

        kwargs['freezables'] = self.get_regions
        kwargs['trainables'] = self.get_regions
        kwargs['stepables'] = self.get_regions
        kwargs['name'] = f"{kwargs['organism']}_brain"

        super(Brain, self).__init__(**kwargs)

        # PERIPHERAL REGIONS
        # TODO all code in this block
        # all of these regions should be able to take in dynamic body parts
        # whether predominately input or output, all regions have some kind of diversity optimizing latent code
        self.auditory_perception = None # spectrograph processing
        self.auditory_generation = None # spectrograph generation
        self.visual_perception = None # set of input images
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
            self.visual_perception,
            self.energovascular_cortex,
            self.gustatory_cortex,
            self.somatosensory_cortex,
            self.motor_cortex,
            self.cerebellum
        ]

        # HIGHER REGIONS
        # TODO all code in this block
        self.memory_center = None # store and recall global workspace trajectories
        self.abstract_reasoning_cortex = None # graph of cortical columns
        # with overall gradient of peripheral_region connection across the neocortex
        self.affective_center = None # outputs arousal, motivational intensity, and valency

        # TODO make new class GWT_Region which all brain regions subclass
        self.global_workspace_regions = [
            self.memory_center,
            self.abstract_reasoning_cortex,
            self.affective_center
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
