import pyspiel
from textwrap import indent
from game_mnk import GameInterface


def _get_history_desc(game_state, moves: list):
    text = f"--  History: {','.join(moves)}"
    if game_state is not None:
        text += f"\n--  Game state: {str(game_state)}"""
    return text


def _get_init_conditions(game_state, moves: list):
    assert len(moves) <= 5, "Number of moves is too high, the maximum is 5!"
    # Example input: ['Deal:0', 'Deal:1', 'Pass' , 'Bet', 'Pass']
    card_p0 = "-1"
    card_p1 = "-1"
    action_0 = "none"
    action_1 = "none"
    action_2 = "none"
    if len(moves) >= 1:
        card_p0 = moves[0].split(":")[1]
    if len(moves) >= 2:
        card_p1 = moves[1].split(":")[1]
    if len(moves) >= 3:
        action_0 = "bet" if moves[2] == "Bet" else "pass"
    if len(moves) >= 4:
        action_1 = "bet" if moves[3] == "Bet" else "pass"
    if len(moves) >= 5:
        action_2 = "bet" if moves[4] == "Bet" else "pass"
    turn = {0: "deal0", 1: "deal1", 2: "turn0", 3: "turn1", 4: "turn2", 5: "rewards"}[len(moves)]
    text = f"""\
{_get_history_desc(game_state, moves)}
Environment.turn = {turn} and Environment.action_0 = {action_0} and Environment.action_1 = {action_1} and
Environment.action_2 = {action_2} and Environment.card_p0 = {card_p0} and Environment.card_p1 = {card_p1} and
 Environment.reward = 0 and
 Player0.null = true and Player1.null = true;
"""
    return text


def generate_specification(game_state, moves: list, formulae_to_check: str):
    spec = """\
Semantics=SingleAssignment;

Agent Environment
    Obsvars:
        turn : {deal0, deal1, turn0, turn1, turn2, rewards, finish};
        action_0 : {bet, pass, none};
        action_1 : {bet, pass, none};
        action_2 : {bet, pass, none};
    end Obsvars
    Vars:
    	card_p0 : -1 .. 2;
    	card_p1 : -1 .. 2;
    	reward : -2 .. 2;
    end Vars
    Actions = { };
    Protocol: end Protocol
    Evolution:
        -- Dealing the first card
    	turn=deal1 if turn=deal0;
    	card_p0=0 if turn=deal0;
    	card_p0=1 if turn=deal0;
    	card_p0=2 if turn=deal0;
    	-- Dealing the second card
    	turn=turn0 if turn=deal1;
    	card_p1=0 if turn=deal1 and (! card_p0=0);
    	card_p1=1 if turn=deal1 and (! card_p0=1);
    	card_p1=2 if turn=deal1 and (! card_p0=2);
        -- Player 0 bets or passes
        turn=turn1 if turn=turn0;
    	action_0=bet if turn=turn0 and Player0.Action=bet;
    	action_0=pass if turn=turn0 and Player0.Action=pass;
    	-- Player 1 bets or passes
    	turn=rewards if turn=turn1 and ((action_0=bet and Player1.Action=bet) or (action_0=pass and Player1.Action=pass));
        turn=turn2 if turn=turn1 and (! ((action_0=bet and Player1.Action=bet) or (action_0=pass and Player1.Action=pass)));
    	action_1=bet if turn=turn1 and Player1.Action=bet;
    	action_1=pass if turn=turn1 and Player1.Action=pass;
    	-- If actions different then Player 0 once again can bet or pass
    	turn=rewards if turn=turn2;
    	action_2=bet if turn=turn2 and Player0.Action=bet;
    	action_2=pass if turn=turn2 and Player0.Action=pass;
    	turn=finish if turn=rewards;
    	reward = 2 if turn=rewards and ((action_1=bet and action_1=bet and card_p0 > card_p1) or
                      (action_0=pass and action_1=bet and action_2=bet and card_p0 > card_p1));
        reward = 1 if turn=rewards and ((action_0=pass and action_1=pass and card_p0 > card_p1) or
                      (action_0=bet and action_1=pass));
        reward = -1 if turn=rewards and ((action_0=pass and action_1=pass and card_p0 < card_p1) or
                      (action_0=pass and action_1=bet and action_2=pass));
        reward = -2 if turn=rewards and ((action_0=bet and action_1=bet and card_p0 < card_p1) or
                       (action_0=pass and action_1=bet and action_2=bet and card_p0 < card_p1));
    end Evolution
end Agent

Agent Player0
	Lobsvars = {card_p0};
    Vars:
        null : boolean; -- for syntax reasons only
    end Vars
    Actions = {pass, bet, none};
    Protocol:
        Environment.turn=turn0: {pass, bet};
        Environment.turn=turn2: {pass, bet};
        Other : { none }; -- technicality
    end Protocol
    Evolution:

        null=true if null=true;
    end Evolution
end Agent

Agent Player1
    Lobsvars = {card_p1};
    Vars:
        null : boolean; -- for syntax reasons only
    end Vars
    Actions = {pass, bet, none};
    Protocol:
        Environment.turn=turn1: {pass, bet};
        Other : { none }; -- technicality
    end Protocol
    Evolution:
        null=true if null=true;
    end Evolution
end Agent

Evaluation
    custom_condition if
        Environment.turn=finish and ((! Environment.card_p0 > Environment.card_p1) or Environment.reward >= 1);
        --Environment.turn=finish and Environment.reward >= 1;
    reward_2 if
        Environment.turn=finish and Environment.reward = 2;
    reward_1 if
        Environment.turn=finish and Environment.reward = 1;
    reward_neg1 if
        Environment.turn=finish and Environment.reward = -1;
    reward_neg2 if
        Environment.turn=finish and Environment.reward = -2;
end Evaluation

InitStates
""" + indent(_get_init_conditions(game_state, moves), " "*4) +\
f"""\
end InitStates

Groups
    player0 = {{Player0}};
    player1 = {{Player1}};
end Groups

Formulae
    {formulae_to_check}
end Formulae"""
    return spec


class GameKuhnPoker(GameInterface):
    def __init__(self):
        GameInterface.__init__(self, players={"player0": 0, "player1": 1})

    def get_name(self):
        return "kuhn_poker"

    def load_game(self):
        return pyspiel.load_game("kuhn_poker")

    def formal_subproblem_description(self, game_state, history, formulae_to_check: str = None) -> str:
        if formulae_to_check is None:
            formulae_to_check, _ = self.get_default_formula_and_coalition()
        if isinstance(history, str):
            history = self.get_moves_from_history_str(history)
        return generate_specification(game_state, history, formulae_to_check)

    def termination_condition(self, history: str):
        """Determines when the branching of the game search space will conclude."""
        pass

    def get_moves_from_history_str(self, history: str) -> list[str]:
        """Converts a single history string to a list of successive actions."""
        if history == "":
            return []
        else:
            return history.split(',')  # E.g. input to process: "Deal:0,Deal:1,Pass,Bet,Pass"

    @classmethod
    def get_default_formula_and_coalition(cls):
        return "<player0> F (reward_2 or reward_1 or reward_neg1);", {0}
