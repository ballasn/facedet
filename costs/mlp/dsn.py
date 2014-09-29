"""
Functionality for training with dropout.
"""
__authors__ = 'Ian Goodfellow'
__copyright__ = "Copyright 2013, Universite de Montreal"

from pylearn2.costs.cost import DefaultDataSpecsMixin, Cost
from theano.sandbox.rng_mrg import MRG_RandomStreams


class DSNBase(DefaultDataSpecsMixin, Cost):

    supervised = True

    def __init__(self,
                 companion,
                 companion_weight = None):

        if companion_weight == None or companion_weight == {}:
            companion_weight = {}
            for name in companion:
                companion_weight[name] = 1.0


        self.__dict__.update(locals())
        del self.self



class DSN(DSNBase):

    supervised = True

    def __init__(self,
                 companion,
                 companion_weight = None):
        super(DSN, self).__init__ (companion, companion_weight)


    def expr(self, model, data, ** kwargs):
        """
        .. todo::

            WRITEME
        """
        space, sources = self.get_data_specs(model)


        ### We need to decompose the fprop for each layer in order
        ### to add the companion term (create some code duplication)
        space.validate(data)
        (X, Y) = data

        state_below = X

        cost = []
        for layer in model.layers:
            layer_name = layer.layer_name
            state_below = layer.fprop(state_below)
            if layer_name in self.companion:
                self.companion[layer_name].mlp = model
                self.companion[layer_name].set_input_space(layer.get_output_space())
                Y_tmp = self.companion[layer_name].fprop(state_below)
                w = self.companion_weight[layer_name]
                costs += [w * (self.companion[layer_name].cost(Y, Y_tmp))]

        costs += [model.cost(Y, state_below)]
        sum_of_costs = reduce(lambda x, y: x + y, costs)
        return sum_of_costs



class DSN_dropout(DSN) :

    supervised = True

    def __init__(self,
                 companion, companion_weight = None,
                 ### Dropout parameters
                 default_input_include_prob=.5, input_include_probs=None,
                 default_input_scale=2., input_scales=None,
                 per_example=True):

        super(DSN_dropout, self).__init__(companion, companion_weight)

        if input_include_probs is None:
            input_include_probs = {}
        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """
        .. todo::

            WRITEME
        """
        space, sources = self.get_data_specs(model)


        ### We need to decompose the fprop for each layer in order
        ### to add the companion term (create some code duplication)
        space.validate(data)
        (X, Y) = data


        if self.input_include_probs is None:
            self.input_include_probs = {}
        if self.input_scales is None:
            self.input_scales = {}

        model._validate_layer_names(list(self.input_include_probs.keys()))
        model._validate_layer_names(list(self.input_scales.keys()))
        theano_rng = MRG_RandomStreams(max(model.rng.randint(2 ** 15), 1))


        costs = []
        state_below = X
        for layer in model.layers:
            layer_name = layer.layer_name

            if layer_name in self.input_include_probs:
                include_prob = self.input_include_probs[layer_name]
            else:
                include_prob = self.default_input_include_prob

            if layer_name in self.input_scales:
                scale = self.input_scales[layer_name]
            else:
                scale = self.default_input_scale

            state_below = model.apply_dropout(
                state=state_below,
                include_prob=include_prob,
                theano_rng=theano_rng,
                scale=scale,
                mask_value=layer.dropout_input_mask_value,
                input_space=layer.get_input_space(),
                per_example=self.per_example
            )
            state_below = layer.fprop(state_below)

            ### For now, no drop-out on companion
            if layer_name in self.companion:
                ### How to do so at init?
                self.companion[layer_name].mlp = model
                self.companion[layer_name].set_input_space(layer.get_output_space())

                Y_tmp = self.companion[layer_name].fprop(state_below)
                w = self.companion_weight[layer_name]
                costs += [w * (self.companion[layer_name].cost(Y, Y_tmp))]

        costs += [model.cost(Y, state_below)]
        sum_of_costs = reduce(lambda x, y: x + y, costs)
        return sum_of_costs

