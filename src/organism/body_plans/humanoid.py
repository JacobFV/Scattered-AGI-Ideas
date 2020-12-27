from ..organism import Organism
from ..organs import circulatory, contact, digestive, ear, energetics, eye, kinesthetic, light, speaker

class Humanoid(Organism):
    """just custom initializer. nothing else different"""

    def __init__(self, **kwargs):
        organs = {
            'rfoot': [None, None],
            ('rfoot', 'rknee'): [None, None, None],
            # TODO: build morphology graph
        }
        super(Humanoid, self).__init__(organs=organs, **kwargs)