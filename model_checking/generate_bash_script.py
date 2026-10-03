

def generate_E1():
    benchmarks = [
        ("mnk3,3,3", "--game mnk -n 3 -m 3 -k 3"),
        ("mnk4,4,3", "--game mnk -n 4 -m 4 -k 3"),
        ("mnk5,5,3", "--game mnk -n 5 -m 5 -k 3"),
        ("mnk5,5,4", "--game mnk -n 5 -m 5 -k 4"),
        ("mnk6,6,3", "--game mnk -n 6 -m 6 -k 3"),
        ("nim2;3;4", "--game nim --piles \"2;3;4\""),  # winning position
        ("nim2;3;5", "--game nim --piles \"2;3;5\""),  # losing position
        ("nim2;19;16", "--game nim --piles \"2;19;16\""),  # winning position
        ("nim2;19;17", "--game nim --piles \"2;19;17\""),  # losing position
    ]
    # benchmarks["tourality"] = [
    #     "--game mcmas_model schlingloff_1.ispl",
    # ]
    action_selectors = [
        ("1-best", "--action_selector1 1-best --action_selector2 all"),
        # ("all", "--action_selector1 all --action_selector2 1-best"),
    ]

    output_dir = "E1[pure-mcts]"
    prefix = "E1[pure-mcts]"
    rollout_count = 5
    max_game_depth = 1000000000
    initial_simulations = 1000000000

    def generate_config_run(config_name, benchmark, action_selector, max_simulations, rollout_count):
        return f"""tsp python3 mcts_v4.py
    --quiet 1
    {benchmark}
    --max_game_depth {max_game_depth}
    {action_selector}
    --submodels_dir "{output_dir}/{config_name}"
    --output_file "{output_dir}/summary/{config_name}.txt"
    --rollout_count {rollout_count}
    --max_simulations {max_simulations}
    --initial_simulations {initial_simulations}
    --use_mcts_outcome_information 1
    --use_reward_in_terminal_states
    --num_games 5
    --timeout 3600""".replace("\n", " ")


    text = f"""#!/bin/bash
    
    source ../../open_spiel_model_checking/venv_model_checking/bin/activate
    
    output_dir="{output_dir}"
    
    mkdir -p "$output_dir/summary"
    
    """

    for b in benchmarks:
        for a_s in action_selectors:
            for max_sim in [("s100", 100)]:
                for roll_count in [("r5", 5)]:
                    config_name = f"{prefix}_{b[0]}_{a_s[0]}_{max_sim[0]}{roll_count[0]}"
                    text += "\n"
                    text += generate_config_run(config_name, b[1], a_s[1], max_sim[1], roll_count[1])
                    text += "\n"


    with open(f"{prefix}.sh", "w") as f:
        f.write(text)


def generate_E2():
    benchmarks = [
        ("mnk3,3,3", "--game mnk -n 3 -m 3 -k 3"),
        ("mnk4,4,3", "--game mnk -n 4 -m 4 -k 3"),
        ("mnk5,5,3", "--game mnk -n 5 -m 5 -k 3"),
        ("mnk5,5,4", "--game mnk -n 5 -m 5 -k 4"),
        ("mnk6,6,3", "--game mnk -n 6 -m 6 -k 3"),
        ("nim1;4;5", "--game nim --piles \"1;4;5\""),  # winning position
        ("nim2;4;5", "--game nim --piles \"2;4;5\""),  # losing position
        ("nim2;19;16", "--game nim --piles \"2;19;16\""),  # winning position
        ("nim2;19;17", "--game nim --piles \"2;19;17\""),  # losing position
    ]
    # benchmarks["tourality"] = [
    #     "--game mcmas_model schlingloff_1.ispl",
    # ]
    action_selectors = [
        ("1-best", "--action_selector1 1-best --action_selector2 all"),
        ("all", "--action_selector1 all --action_selector2 1-best"),
    ]

    output_dir = "E2[pure-mcts-steps]"
    prefix = "E2[pure-mcts-steps]"
    rollout_count = 5
    max_game_depth = 1000000000
    # initial_simulations = 1000

    def generate_config_run(config_name, benchmark, action_selector, inital_sim, max_simulations, rollout_count):
        return f"""tsp python3 mcts_v4.py
    --quiet 1
    {benchmark}
    --max_game_depth {max_game_depth}
    {action_selector}
    --submodels_dir "{output_dir}/{config_name}"
    --output_file "{output_dir}/summary/{config_name}.txt"
    --rollout_count {rollout_count}
    --max_simulations {max_simulations}
    --initial_simulations {inital_sim}
    --use_mcts_outcome_information 1
    --use_reward_in_terminal_states
    --num_games 2
    --timeout 3600""".replace("\n", " ")

    text = f"""#!/bin/bash

    source ../../open_spiel_model_checking/venv_model_checking/bin/activate

    output_dir="{output_dir}"

    mkdir -p "$output_dir/summary"

    """

    for b in benchmarks:
        for a_s in action_selectors:
            for inital_sim in [("in1000", 1000), ("in0", 0)]:
                for max_sim in [("s200", 200)]:
                    for roll_count in [("r5", 5)]:
                        config_name = f"{prefix}_{b[0]}_{a_s[0]}_{inital_sim[0]}{max_sim[0]}{roll_count[0]}"
                        text += "\n"
                        text += generate_config_run(config_name, b[1], a_s[1], inital_sim[1], max_sim[1], roll_count[1])
                        text += "\n"

    with open(f"{prefix}.sh", "w") as f:
        f.write(text)


if __name__ == "__main__":
    # generate_E1()
    generate_E2()
