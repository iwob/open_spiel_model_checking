

class ActionSelector:
    def select_actions(self, actions: list[(float, str)], cur_player: int, coalition: set[int]) -> list[(float, str)]:
        """Selects actions to keep as enabled in the game tree.

        :param actions: a sorted list of action names and their evaluation for the current player
        :param cur_player: id of the current player.
        :param coalition: a set ids of players in the coalition.
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.select_actions(*args)


class DualActionSelector(ActionSelector):
    """Action selector that uses one selector for players in the coalition, and another to players in the anti-coalition."""
    def __init__(self, coalition_policy: ActionSelector, anticoalition_policy: ActionSelector):
        self.coalition_policy = coalition_policy
        self.anticoalition_policy = anticoalition_policy

    def select_actions(self, actions: list[(float, str)], cur_player: int, coalition: set[int]) -> list[(float, str)]:
        if cur_player in coalition:
            return self.coalition_policy(actions, cur_player, coalition)
        else:
            return self.anticoalition_policy(actions, cur_player, coalition)


class SelectKBestActions(ActionSelector):
    def __init__(self, k):
        """
        :param k: Number of the best actions to be returned.
        """
        self.k = k

    def select_actions(self, actions: list[(float, str)], cur_player: int, coalition: set[int]) -> list[(float, str)]:
        return actions[:self.k]


class SelectAtMostTwoActions(ActionSelector):
    """Original action selection scheme as implemented in the Master's thesis."""
    def __init__(self, epsilon_ratio=0.95):
        """
        :param epsilon_ratio: If a ratio between the first and second highest q-value is higher
         or equal than this number, then two actions are selected; otherwise only one.
        """
        self.epsilon_ratio = epsilon_ratio

    def select_actions(self, actions: list[(float, str)], cur_player: int, coalition: set[int]) -> list[(float, str)]:
        if len(actions) <= 1:
            return actions

        if actions[1][0] == 0 or actions[0][0] * actions[1][0] < 0:  # second is zero or signs are different
            return actions[:1]
        elif actions[0][0] < 0 and actions[0][0] / actions[1][0] >= self.epsilon_ratio:  # both negative
            return actions[:2]
        elif actions[0][0] > 0 and actions[1][0] / actions[0][0] >= self.epsilon_ratio:
            return actions[:2]
        else:
            return actions[:1]


class SelectAllActions(ActionSelector):
    """Action selector that returns all actions."""
    def select_actions(self, actions: list[(float, str)], cur_player: int, coalition: set[int]) -> list[(float, str)]:
        return actions


class SelectKRandomly(ActionSelector):
    """Action selector that returns k randomly selected actions with the uniform probability."""
    def __init__(self, k):
        """
        :param k: Number of the best actions to be returned.
        """
        self.k = k

    def select_actions(self, actions: list[(float, str)], cur_player: int, coalition: set[int]) -> list[(float, str)]:
        raise NotImplementedError
