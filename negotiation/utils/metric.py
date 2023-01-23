# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Several metrics that are being evaluated during selfplay.
"""

from typing import List
import time
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
import utils.utils as utils


class TimeMetric(object):
    """Time based metric."""

    def __init__(self):
        self.t = 0
        self.n = 0

    def reset(self):
        self.last_t = time.time()

    def record(self, n=1):
        self.t += time.time() - self.last_t
        self.n += 1

    def value(self):
        self.n = max(1, self.n)
        return 1.0 * self.t / self.n

    def show(self):
        return '%.3fs' % (1. * self.value())


class NumericMetric(object):
    """Base class for a numeric metric."""

    def __init__(self):
        self.k = 0
        self.n = 0

    def reset(self):
        pass

    def record(self, k, n=1):
        self.k += k
        self.n += n

    def value(self):
        self.n = max(1, self.n)
        return 1.0 * self.k / self.n


class PercentageMetric(NumericMetric):
    """Percentage."""

    def show(self):
        return '%2.2f%%' % (100. * self.value())


class AverageMetric(NumericMetric):
    """Average."""

    def show(self):
        return '%.2f' % (1. * self.value())


class TextMetric(object):
    """Text based metric."""

    def __init__(self, text):
        self.text = text
        self.k = 0
        self.n = 0

    def reset(self):
        pass

    def value(self):
        self.n = max(1, self.n)
        return 1. * self.k / self.n

    def show(self):
        return '%.2f' % (1. * self.value())


class NGramMetric(TextMetric):
    """Metric that evaluates n gramms."""

    def __init__(self, text, ngram=-1):
        super(NGramMetric, self).__init__(text)
        self.ngram = ngram

    def record(self, sen):
        n = len(sen) if self.ngram == -1 else self.ngram
        for i in range(len(sen) - n + 1):
            self.n += 1
            target = ' '.join(sen[i:i + n])
            if self.text.find(target) != -1:
                self.k += 1


class UniquenessMetric(object):
    """Metric that evaluates the number of unique sentences."""

    def __init__(self):
        self.seen = set()

    def reset(self):
        pass

    def record(self, sen):
        self.seen.add(' '.join(sen))

    def value(self):
        return len(self.seen)

    def show(self):
        return str(self.value())


class SimilarityMetric(object):
    """Metric that evaluate similarity of the produced sentences."""

    def __init__(self):
        self.reset()
        self.k = 0
        self.n = 0

    def reset(self):
        self.history = []

    def record(self, sen):
        self.n += 1
        sen = ' '.join(sen)
        for h in self.history:
            if h == sen:
                self.k += 1
                break
        self.history.append(sen)

    def value(self):
        self.n = max(1, self.n)
        return 1. * self.k / self.n

    def show(self):
        return '%.2f' % (1. * self.value())


class DiversityDialogueMetric(UniquenessMetric):
    """
    Evaluates diversity over the entire dialogue (not agent-specific)
    """

    def __init__(self):
        super(DiversityDialogueMetric, self).__init__()

    def reset(self, n=1):
        self.seen = set()

    def show(self):
        return '%.2f' % (1. * self.value())


class ParetoOptimalityMetric:
    def __init__(self):
        self.avg_agree = 0
        self.avg_can_improve = 0

    def reset(self):
        pass

    def record(self, ctxs, rewards):
        self.avg_agree += 1
        cnts = np.array([int(x) for x in ctxs[0][0::2]])
        vals1 = np.array([int(x) for x in ctxs[0][1::2]])
        vals2 = np.array([int(x) for x in ctxs[1][1::2]])
        choices = utils.gen_choices(cnts)
        can_improve = False
        score1, score2 = rewards[0], rewards[1]
        cand_scores = []  # for debugging purposes
        for cand1, cand2 in choices:
            cand_score1 = utils.compute_score(vals1, cand1)
            cand_score2 = utils.compute_score(vals2, cand2)
            cand_scores.append((cand_score1, cand_score2)) # for debugging purposes
            if (cand_score1 > score1 and cand_score2 >= score2) or (cand_score1 >= score1 and cand_score2 > score2):
                can_improve = True

        self.avg_can_improve += int(can_improve)
        return int(can_improve)

    def value(self):
        if self.avg_agree == 0:
            return 0
        ratio = (1.0 * self.avg_can_improve / self.avg_agree)
        return 100. * (1. - ratio)

    def show(self):
        return '%.2f' % (1. * self.value())


class DiversityMetric(UniquenessMetric):
    def __init__(self):
        super(DiversityMetric, self).__init__()
        self.n = 0  # length of dialogue
        self.m = 0  # number of dialogues
        self.sum_of_avgs = 0

    def reset(self):
        self.seen = set()
        self.n = 0
        self.m += 1

    def record(self, sen):
        self.seen.add(' '.join(sen))
        self.n += 1

    def record_end_of_dialogue(self):
        self.n = max(1, self.n)
        avg = len(self.seen) / self.n
        self.sum_of_avgs += avg

    def value(self):
        self.m = max(1, self.m)
        return 1. * self.sum_of_avgs / self.m

    def show(self):
        return '%.2f' % (1. * self.value())


class NoveltyMetric(NumericMetric):
    def __init__(self, model):
        super(NoveltyMetric, self).__init__()
        self.model = model
        # self.metric = metric
        # self.history = []

    def reset(self):
        pass

    def record(self, sen: List[str], agent, n=1):
        sen = agent._encode(sen, self.model.word_dict)

        # get the previous hidden state since lang_h has been updated by the write function
        lang_h = agent.lang_hs[-2][-1].unsqueeze(0).unsqueeze(0) if len(agent.lang_hs) > 1 else agent.model.zero_hid(1)
        score = self.model.score(Variable(sen), lang_h, agent.ctx_h, agent.args.temperature, agent.context_mask,
                                 metric='likelihood')
        self.k += score[0].item()
        self.n += n

    def value(self):
        self.n = max(1, self.n)
        return 1.0 * self.k / self.n

    def show(self):
        return '%.2f' % (1. * self.value())


class MetricsContainer(object):
    """A container that stores and updates several metrics."""

    def __init__(self):
        self.metrics = OrderedDict()

    def _register(self, name, ty, *args, **kwargs):
        name = name.lower()
        #assert name not in self.metrics
        self.metrics[name] = ty(*args, **kwargs)

    def register_average(self, name, *args, **kwargs):
        self._register(name, AverageMetric, *args, **kwargs)

    def register_time(self, name, *args, **kwargs):
        self._register(name, TimeMetric, *args, **kwargs)

    def register_percentage(self, name, *args, **kwargs):
        self._register(name, PercentageMetric, *args, **kwargs)

    def register_ngram(self, name, *args, **kwargs):
        self._register(name, NGramMetric, *args, **kwargs)

    def register_similarity(self, name, *args, **kwargs):
        self._register(name, SimilarityMetric, *args, **kwargs)

    def register_uniqueness(self, name, *args, **kwargs):
        self._register(name, UniquenessMetric, *args, **kwargs)

    def register_diversity(self, name, *args, **kwargs):
        self._register(name, DiversityMetric, *args, **kwargs)

    def register_novelty(self, name, *args, **kwargs):
        self._register(name, NoveltyMetric, *args, **kwargs)

    def register_pareto(self, name, *args, **kwargs):
        self._register(name, ParetoOptimalityMetric, *args, **kwargs)

    def register_diversity_dialogue(self, name, *args, **kwargs):
        self._register(name, DiversityDialogueMetric, *args, **kwargs)

    def record(self, name, *args, **kwargs):
        name = name.lower()
        assert name in self.metrics
        self.metrics[name].record(*args, **kwargs)

    def record_pareto(self, name, *args, **kwargs):
        name = name.lower()
        assert name in self.metrics
        return self.metrics[name].record(*args, **kwargs)

    def record_end_of_dialogue(self, name, *args, **kwargs):
        """
        This function should only be called for the diversity metric
        """
        name = name.lower()
        assert name in self.metrics
        assert 'diversity' in name
        self.metrics[name].record_end_of_dialogue(*args, **kwargs)

    def reset(self):
        for m in self.metrics.values():
            m.reset()

    def value(self, name):
        return self.metrics[name].value()

    def show(self):
        return ' '.join(['%s=%s' % (k, v.show()) for k, v in self.metrics.iteritems()])

    def dict(self):
        d = OrderedDict()
        for k, v in self.metrics.items():
            d[k] = v.show()
        return d
