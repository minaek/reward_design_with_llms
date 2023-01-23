from typing import Optional

from .utterance import Utterance
from .lf import LogicalForm


class DialogueState(object):
    def __init__(self, agent, kb):
        self.agent = agent
        self.partner = 1 - agent
        self.kb = kb
        self.time = 0
        init_utterance = Utterance(logical_form=LogicalForm('<start>'), template=['<start>'])
        self.utterance = [init_utterance, init_utterance]
        self.done = set()
        self.proposal = [None, None]
        self.curr_proposal = None

    @property
    def my_act(self):
        return self.utterance[self.agent].lf.intent

    @property
    def partner_act(self):
        return self.utterance[self.partner].lf.intent

    @property
    def partner_utterance(self):
        return self.utterance[self.partner]

    @property
    def partner_template(self):
        try:
            return self.utterance[self.partner].template
        except:
            return None

    @property
    def my_proposal(self):
        return self.proposal[self.agent]

    @my_proposal.setter
    def my_proposal(self, proposal):
        self.proposal[self.agent] = proposal

    @property
    def partner_proposal(self):
        return self.proposal[self.partner]

    def update(self, agent: int, lf: Optional[LogicalForm]):
        if lf is None:
            return

        self.time += 1
        if agent == self.agent:
            self.done.add(lf.intent)

        if hasattr(lf, 'proposal') and lf.proposal is not None:
            self.proposal[agent] = lf.proposal
            self.curr_proposal = lf.proposal
