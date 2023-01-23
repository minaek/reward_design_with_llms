import random
from enum import Enum
from typing import Tuple


class Update(Enum):
    RL = 'rl'
    SL = 'sl'
    NONE = 'none'


class Trainer:
    def select_update_types(self, ep: int) -> Tuple[Update, Update]:
        """
        Select the current update
        Args:
            ep: The current episode iteration number (starting at 1)

        Returns: A tuple representing the update type for each agent
        """
        pass


class FreezeTrainer(Trainer):
    def __init__(self, args):
        self.p_freeze = args.p_freeze

    def freeze_update(self, update_type: Update) -> Tuple[Update, Update]:
        """
        Freeze bob based on the training strategy
        Args:
            update_type: The update type for Alice

        Returns: The update type for both Alice and Bob
        """
        freeze = self.p_freeze is None or random.random() < self.p_freeze
        if freeze:
            return update_type, Update.NONE
        return update_type, update_type


class RandTrainer(FreezeTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.p_sl_update = args.p_sl_update

    def select_update_types(self, ep: int) -> Tuple[Update, Update]:
        p = random.random()

        if p < self.p_sl_update:
            return self.freeze_update(Update.SL)
        else:
            return self.freeze_update(Update.RL)


class ScheduledTrainer(FreezeTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.self_play_updates = args.self_play_updates
        self.supervised_updates = args.supervised_updates

    def select_update_types(self, ep: int) -> Tuple[Update, Update]:
        total_updates = self.self_play_updates + self.supervised_updates
        if ep % total_updates < self.self_play_updates:
            return self.freeze_update(Update.RL)
        else:
            return self.freeze_update(Update.SL)


def get_trainer(trainer_type: str, args) -> Trainer:
    if trainer_type == 'rand':
        return RandTrainer(args)
    elif trainer_type == 'sched':
        return ScheduledTrainer(args)
    raise ValueError('Invalid trainer type')
