from typing import List


class KB:
    """
    Knowledge Base for an instance of the Object Division Game
    """
    def __init__(self, example_input: List[int]):
        """
        Args:
            example_input: A list of the 6 integers between the <input> tags
        """
        self.item_counts = {
            'book': example_input[0],
            'hat': example_input[2],
            'ball': example_input[4],
        }
