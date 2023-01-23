from typing import Iterable, List

from utils.data import Dictionary
from coarse_dialogue_acts.lf import LogicalForm

NON_PROPOSE_ACTS = (
    'agree',
    'disagree',
    #'inquire',
    #'greet',
    '<unk>',
    '<pad>',
    '<selection>',
    'YOU:',
    'THEM:',
    '<eos>'
)

OBJECT_TO_ITEM = (
    ('book', 'item0'),
    ('hat', 'item1'),
    ('ball', 'item2')
)


class ActDictionary(Dictionary):
    def __init__(self, selection_size: int = 3):
        """
        Args:
            selection_size: Defaults to ObjectDivisionDomain.selection_length() // 2
        """
        super().__init__(init=False)
        self.selection_size = selection_size
        self.initialize()

    def initialize(self):
        """
        Generate a dictionary from a game domain, which contains all of the possible coarse
        dialogue acts, along with every possible proposal that an agent could make, regardless
        of the environment (aka the number/values of the objects in the game)

        Args:
            domain: Information about the negotiation game

        Returns: An ActDictionary with all possible coarse dialogue acts
        """

        def generate(item_id, selection=[]):
            if item_id >= self.selection_size:
                selection = ' '.join(selection)
                self.add_word(f'propose: {selection}')
                self.add_word(f'insist: {selection}')   # adding insist CDA
                return
            for i in range(5):
                selection.append('item%d=%d' % (item_id, i))
                generate(item_id + 1, selection)
                selection.pop()

        generate(0)

        for token in NON_PROPOSE_ACTS:
            self.add_word(token)

    def act2idx(self, lfs: Iterable[LogicalForm]) -> List[int]:
        indices = []
        for lf in lfs:
            if lf.intent == 'propose':
                selection = ' '.join(f'{item}={lf.my_proposal[obj]}' for obj, item in OBJECT_TO_ITEM)
                indices.append(self.get_idx(f'propose: {selection}'))
            elif lf.intent == 'insist':   # adding insist CDA
                selection = ' '.join(f'{item}={lf.my_proposal[obj]}' for obj, item in OBJECT_TO_ITEM)
                indices.append(self.get_idx(f'insist: {selection}'))
            else:
                indices.append(self.get_idx(lf.intent))
        return indices


if __name__ == '__main__':
    from domain import ObjectDivisionDomain
    domain = ObjectDivisionDomain()
    dictionary = ActDictionary()

    lf1 = LogicalForm('propose', proposal={0: {'hat': 1, 'book': 2, 'ball': 3}}, sender_id=0)
    lf2 = LogicalForm('agree')
    lf3 = LogicalForm('<unk>')
    lf4 = LogicalForm('<selection>')
    lf5 = LogicalForm('YOU:')
    indices = dictionary.act2idx([lf1, lf2, lf3, lf4, lf5])
    tokens = dictionary.i2w(indices)
    assert tokens == ['propose: item0=2 item1=1 item2=3', 'agree', '<unk>', '<selection>', 'YOU:']


###
# 4 books 1 hat 1 ball
# 4 0 1 6 1 4
#<dialogue>
# THEM: i need the hat and 2 books <eos> 2 books 0 hat 1 ball, 55: 2 books 1 hat 0 ball
# YOU: i'll give you all the books if i can have the hat . <eos> 5: 0 books 1 hat 0 ball
# THEM: i have to have the hat and at least 1 book <eos> 30: 1 books 1 hat 0 balls
# YOU: unless you are willing to give up that hat i just cant see us coming up with anything <eos> 101: 4 books 0 hat 1 ball
# THEM: the hat is non negotiable , i cannot make a deal without the hat <eos> NONE
# YOU: no deal <eos>
# THEM: no deal <eos>
# YOU: <selection>
# </dialogue>



# 2 books 1 hats 1 ball
# 2 1 3 1 1 5
#<dialogue>
# YOU: i would like 2 hats and the ball <eos> 11:  0 books 2 hats 1 ball
# THEM: what about one hat and the ball ? <eos> 6: 0 books 1 hat 1 ball
# YOU: one book , one hat , and the ball <eos> 31: 1 book 1 hat 1 ball
# THEM: okay that is fine . <eos> 125
# YOU: thanks <eos> 129
# THEM: <selection>
# </dialogue> #

# 1 book 3 hats 1 ball
#<dialogue>
# THEM: hi i would like the book and ball and you can have the hats <eos> 52: 1 book 0 hats 1 ball
# YOU: i can give you either the book or the ball <eos> 31: insist: 0 books 3 hats 0 balls
# THEM: ill take the book <eos> 50 1 book 0 hats 0 balls
# YOU: ok i will take the hats and ball <eos> 33 insist: 0 books 3 hats 1 ball
# THEM: deal <eos> 250
# YOU: <selection>
# </dialogue>

####