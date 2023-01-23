from typing import Tuple, List, Union

from .entity import Entity, CanonicalEntity, is_entity
from .dialogue_state import DialogueState
from .utterance import Utterance
from .lf import LogicalForm


Token = Union[str, int, Entity, CanonicalEntity]


class Parser:
    ME = 0
    YOU = 1

    greeting_words = {'hi', 'hello', 'hey', 'hiya', 'howdy'}

    question_words = {'what', 'when', 'where', 'why', 'which', 'who', 'whose', 'how', 'do', 'did', 'does', 'are', 'is',
                      'would', 'will', 'can', 'could', 'any'}

    neg_words = {'no', 'not', "n't", "nothing", "zero", "dont", "worthless"}

    i_words = ('i', 'ill', 'id', 'me', 'mine', 'my')
    you_words = ('u', 'you', 'yours', 'your')
    agreement_words = {'ok', 'okay', 'deal', 'sure', 'fine', 'yes', 'yeah', 'good', 'work', 'great', 'perfect'}
    sentence_delimiter = ('.', ';', '?')

    def __init__(self, agent, kb, lexicon):
        self.agent = agent
        self.partner = 1 - agent
        self.kb = kb
        self.lexicon = lexicon

    def parse(self, sentence_tokens: List[str], dialogue_state: DialogueState) -> Tuple[LogicalForm, bool]:
        """

        Args:
            sentence_tokens: A list of tokens that represent the current sentence
            dialogue_state: The current dialogue state

        Returns: (
            lf: The logical form that represents the sentence
            ambiguous_proposal: If True, indicates that the parser is unsure about the result
        )

        """
        tokens = self.lexicon.link_entity(sentence_tokens)
        utterance = Utterance(raw_text=' ', tokens=tokens)
        intent = self.classify_intent(utterance)

        split = None
        proposal_type = None
        ambiguous_proposal = False
        if intent == 'propose':
            proposal, proposal_type, ambiguous_proposal = self.parse_proposal(tokens, self.kb.item_counts)
            if proposal:
                # NOTE: YOU/ME in proposal is from the partner's perspective
                split = {self.agent: proposal[self.YOU], self.partner: proposal[self.ME]}
                if dialogue_state.partner_proposal and split[self.partner] == dialogue_state.partner_proposal[self.partner]:
                    intent = 'insist'

        lf = LogicalForm(intent, proposal=split, proposal_type=proposal_type, sender_id=self.partner)

        return lf, ambiguous_proposal

    def classify_intent(self, utterance: Utterance) -> str:
        if self.has_item(utterance):
            intent = 'propose'
        elif self.is_agree(utterance):
            intent = 'agree'
        elif self.is_negative(utterance):
            intent = 'disagree'
        elif self.is_question(utterance):
            intent = 'inquire'
        elif self.is_greeting(utterance):
            intent = 'greet'
        else:
            intent = '<unk>'
        return intent

    def parse_proposal(self, tokens, item_counts):
        proposal = {self.ME: {}, self.YOU: {}}
        items = []
        curr_agent = None
        uncertain = False

        def pop_items(agent, items):
            if agent is None or len(items) == 0:
                return
            for item, count in items:
                proposal[agent][item] = count
            del items[:]

        for i, token in enumerate(tokens):
            pop_items(curr_agent, items)
            if token in self.i_words:
                curr_agent = self.ME
            elif token in self.you_words:
                curr_agent = self.YOU
            # Reset
            elif token in self.sentence_delimiter:
                curr_agent = None
            elif self._is_item(token):
                item = token.canonical.value
                count, guess = self.parse_count(tokens, i, item_counts[item])
                uncertain = uncertain or guess
                items.append((item, count))
        # Clean up. Assuming it's for 'me' if no subject is mentioned.
        if len(items) > 0:
            if curr_agent is None:
                curr_agent = self.ME
                uncertain = True
            pop_items(curr_agent, items)

        if not proposal[self.ME] and not proposal[self.YOU]:
            return None, None, None

        # print 'explict proposal:', proposal
        proposal_type = self.proposal_to_str(proposal, item_counts)

        # Inform: don't need item
        if proposal[self.ME] and not proposal[self.YOU] and sum(proposal[self.ME].values()) == 0:
            for item, count in item_counts.items():
                # Take everything else
                if item not in proposal[self.ME]:
                    proposal[self.ME][item] = count

        # Merge proposal
        proposal = self.merge_proposal(proposal, item_counts, self.ME)
        # proposal: inferred proposal for both agents (after merge)
        # proposal_type: proposal mentioned in the utterance (before merge)
        return proposal, proposal_type, uncertain

    def proposal_to_str(self, proposal, item_counts):
        s = []
        for agent in (self.ME, self.YOU):
            ss = ['me' if agent == self.ME else 'you']
            if agent in proposal:
                p = proposal[agent]
                # TODO: sort items
                for item in ('book', 'hat', 'ball'):
                    count = 'none' if (not item in p) or p[item] == 0 \
                        else 'all' if p[item] == item_counts[item] \
                        else 'number'
                    ss.append(count)
            else:
                ss.extend(['none'] * 3)
            s.append(','.join(ss))
        return ', '.join(s)

    def merge_proposal(self, proposal, item_counts, speaking_agent):
        # Complete proposal
        for agent in proposal:
            if len(proposal[agent]) > 0:
                for item in item_counts:
                    if not item in proposal[agent]:
                        proposal[agent][item] = 0

        partner = 1 - speaking_agent
        for item, count in item_counts.items():
            my_count = proposal[speaking_agent].get(item)
            if my_count is not None:
                proposal[partner][item] = count - my_count
            else:
                partner_count = proposal[partner].get(item)
                if partner_count is not None:
                    proposal[speaking_agent][item] = count - partner_count
                # Should not happened: both are None
                else:
                    print('WARNING: trying to merge proposals but both counts are none.')
                    proposal[speaking_agent][item] = count
                    proposal[partner][item] = 0
        return proposal

    def parse_count(self, tokens: List[Token], i: int, total: int) -> Tuple[int, bool]:
        """Parse count of an item at index `i`.

        Args:
            tokens: all tokens in the utterance
            i: position of the item token
            total: total count of the item

        Returns:
            count (int)
            guess (bool): True if we are uncertain about the parse

        """
        count = None
        # Search backward
        for j in range(i - 1, -1, -1):
            token = tokens[j]
            if count is not None or token in self.sentence_delimiter or self._is_item(token):
                break
            elif self._is_number(token):
                count = min(token.canonical.value, total)
            elif token in self.neg_words:
                count = 0
            elif token in ('a', 'an'):
                count = 1
            elif token == 'both':
                count = 2
            elif token in ('the', 'all'):
                count = total

        if count is None:
            # Search forward
            for j in range(i + 1, len(tokens)):
                token = tokens[j]
                if count is not None or token in self.sentence_delimiter or self._is_item(token):
                    break
                elif count in self.neg_words:
                    count = 0

        if count is None:
            return total, True
        else:
            count = min(count, total)
            return count, False

    def has_item(self, utterance: Utterance) -> bool:
        for token in utterance.tokens:
            if self._is_item(token):
                return True
        return False

    def is_agree(self, utterance: Utterance) -> bool:
        return any(t in self.agreement_words for t in utterance.tokens) and not self.is_negative(utterance)

    def is_negative(self, utterance: Utterance) -> bool:
        return any(t in self.neg_words for t in utterance.tokens)

    def is_question(self, utterance: Utterance) -> bool:
        tokens = utterance.tokens
        if len(tokens) < 1:
            return False
        last_word = tokens[-1]
        first_word = tokens[0]
        return last_word == '?' or first_word in self.question_words

    def is_greeting(self, utterance: Utterance) -> bool:
        return any(t in self.greeting_words for t in utterance.tokens)

    def _is_item(self, token: Token) -> bool:
        return is_entity(token) and token.canonical.type == 'item'

    def _is_number(self, token: Token) -> bool:
        return is_entity(token) and token.canonical.type == 'number'
