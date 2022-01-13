#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from lifelines.utils import concordance_index
from skorch import NeuralNet


class BaseSurvivalNeuralNet(NeuralNet):
    """Implement survival score function for `NeuralNet` class."""

    def score(self, X: np.ndarray, y: np.ndarray):
        try:
            concordance = concordance_index(
                event_times=np.vstack(np.char.split(np.array(y), sep="|"))[
                    :, 1
                ].astype(np.float32),
                # Flip the scoring as required by `lifelines`.
                predicted_scores=np.negative(np.squeeze(self.predict(X))),
                event_observed=np.vstack(np.char.split(np.array(y), sep="|"))[
                    :, 0
                ].astype(int),
            )
        except ValueError as e:
            raise e
        return concordance
