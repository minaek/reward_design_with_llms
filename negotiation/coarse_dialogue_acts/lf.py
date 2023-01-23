from typing import Dict


class Proposal(Dict):
    book: int
    hat: int
    ball: int


class LogicalForm(object):
    def __init__(self, intent: str, proposal: Dict[int, Proposal] = None, proposal_type: str = None, sender_id: int = None):
        """
        Args:
            intent: The Coarse Dialogue Act type
            proposal: A mapping from agent_id to divisions proposed by the proposal
            proposal_type: A description of the proposal in string form
            sender_id: The agent id of the sender of the proposal
        """
        self.intent = intent
        self.proposal = proposal
        self.proposal_type = proposal_type
        self.sender_id = sender_id

    def to_dict(self):
        attrs = vars(self)
        attrs['intent'] = self.intent
        return attrs

    @property
    def my_proposal(self):
        if self.proposal is None or self.sender_id is None:
            return None
        return self.proposal[self.sender_id]

    def __str__(self):
        attrs = vars(self)
        s = ' '.join(['{}={}'.format(k, v) for k, v in attrs.items()])
        return s
