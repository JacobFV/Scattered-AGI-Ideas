from ..utils import Stepable

import random


class GeneExpressionFn:

    @property
    def expression_rate(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()

    def step(self):
        if random.random() <= self.expression_rate:
            self.run()

    def __init__(self, agent):
        self.agent = agent


class SimpleTranscription(GeneExpressionFn):

    @property
    def expression_rate(self):
        return 0.001

    def run(self):
        self.agent.rna = self.agent.dna[0]

class RNAMutation(GeneExpressionFn):

    @property
    def expression_rate(self):
        return 0.005 # TODO see DNAMutation version of this

    def run(self):
        raise NotImplementedError() # TODO: share this code with DNAMutation

class DNAMutation(GeneExpressionFn):

    @property
    def expression_rate(self):
        return 0.0001 # TODO:
        # TODO: read developmental genes and make mutations less early on
        # TODO: also let mutation be a function of free energy at organ
        # TODO: and finally, share this code with RNAMutation

    def run(self):
        raise NotImplementedError() # TODO: share this code with RNAMutation


class GeneLoader(GeneExpressionFn):

    def run(self): # TODO
        # remove self after running once
        raise NotImplementedError()

class ConvenienceInit(GeneExpressionFn):
    """organs and other classes have a staticmethod that returns a
    gene expression function using this convenience initializer

    gene_params make a dictionary from genetic code to python code string names
    and makes appropriate type casts
    """

    def __init__(self, gene_params, call_fn, agent):
        """
        gene_params: list<tuple<genetic_code_name:str, python_code_name:str, cast_fn:callable?>>
        """
        super(ConvenienceInit, self).__init__(agent=agent)

        self.gene_params = gene_params
        self.call_fn = call_fn

    def run(self, agent):
        # TODO: 1) look for gene_params 2) translate to python names 3) call fn
        raise NotImplementedError()