"""measures contact for a particular class: eg: 'fine touch' or 'taste' """
from . import NodeOrgan

class NodeContact(NodeOrgan):

    def __init__(self, emb_fn, contact_type, geometry, d_emb, **kwargs):
        super(NodeContact, self).__init__(**kwargs)

        self.emb_fn = emb_fn
        self.contact_type = contact_type
        self.geometry = geometry
        self.d_emb = d_emb

    def set_action(self, action):
        if self.en
