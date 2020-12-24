class GeneExpressionFn:

    @property
    def expression_rate(self):
        raise NotImplementedError()

    def run(agent):
        raise NotImplementedError()


class GeneLoader(GeneExpressionFn):

    def run(self, agent):
        raise NotImplementedError() # TODO
        # remove self after running once

class ConvenienceInit(GeneExpressionFn):
    """organs and other classes have a staticmethod that returns a
    gene expression function using this convenience initializer

    gene_params make a dictionary from genetic code to python code string names
    and makes appropriate type casts
    """

    def __init__(self, gene_params, call_fn):
        """
        gene_params: list<tuple<genetic_code_name:str, python_code_name:str, cast_fn:callable?>>
        """
        self.gene_params = gene_params
        self.call_fn = call_fn

    def run(self, agent):
        # TODO: 1) look for gene_params 2) translate to python names 3) call fn
        raise NotImplementedError()