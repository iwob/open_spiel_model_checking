import os
import numpy as np
import pandas as pd

import evoplotter.reporting
from pathlib import Path
import shutil

def load_properties(filepath, sep='=', comment_char='#'):
    """
    Read the file passed as parameter as a properties file.
    """
    props = {}
    with open(filepath, "rt") as f:
        for line in f:
            l = line.strip()
            if l and not l.startswith(comment_char):
                key_value = l.split(sep)
                key = key_value[0].strip()
                value = sep.join(key_value[1:]).strip().strip('"')
                props[key] = value
    return props


def move_row_to_last_id(df, row_id):
    row_to_move = df.loc[row_id]
    print("row_to_move.to_frame(): ", row_to_move.to_frame())
    # Remove the row and append it to the end
    # df = df.drop(row_id).append(row_to_move, ignore_index=True)
    df = df.drop(row_id)
    df = pd.concat([df, row_to_move.to_frame().T])
    return df

def load_properties_from_dir(dir_path):
    res = []
    for file in os. listdir(dir_path):
        res.append(load_properties(os.path.join(dir_path, file)))
    return res

def map_max_depth_to_depth_percent(n, x):
    d = {
            4: {4: 0.25, 8: 0.5, 12: 0.75, 16: 1.0},
            5: {6: 0.25, 13: 0.5, 19: 0.75, 25: 1.0},
            6: {9: 0.25, 18: 0.5, 27: 0.75, 36: 1.0},
            7: {12: 0.25, 25: 0.5, 37: 0.75, 49: 1.0},
            8: {16: 0.25, 32: 0.5, 48: 0.75, 64: 1.0},
            9: {20: 0.25, 41: 0.5, 61: 0.75, 81: 1.0},
            10: {25: 0.25, 50: 0.5, 75: 0.75, 100: 1.0},
         }
    if n not in d:
        return None
    else:
        d2 = d[n]
        if x not in d2:
            return None
        else:
            return d2[x]

expected_results_dict = {
    "mnk(3,3,3)": False,
    "mnk(4,4,3)": True,
    "mnk(5,5,3)": True,
    "mnk(5,5,4)": False,
    "mnk(6,6,3)": True,
    "mnk(7,7,3)": True,
    "mnk(8,8,3)": True,
    "mnk(9,9,3)": True,
    "mnk(10,10,3)": True,
    "nim(1,4,5)": False,
    "nim(2,3,4)": True,
    "nim(2,3,5)": True,
    "nim(2,4,5)": True,
    "nim(2,19,17)": False,
    "nim(2,19,16)": True
}


def get_benchmark_name(d):
    if d["game"] == "mnk":
        n = int(d['m,n,k'].split(',')[0][1:])
        m = int(d['m,n,k'].split(',')[1][1:])
        k = int(d['m,n,k'].split(',')[2][:-1].strip())
        return f"mnk({n},{m},{k})"
    elif d["game"] == "nim":
        return f"nim({d["piles"].replace(";", ",")})"
    elif d["game"] == "mcmas_model":
        return "tourality_"
    else:
        raise Exception(f"Unknown game: {d['game']}")

def process_dict(d):
    name = get_benchmark_name(d)
    return {
            'benchmark': name,
            'expected_results': expected_results_dict[name],
            'max_game_depth': int(d['max_game_depth']),
            'action_selector1': d['action_selector1'],
            'action_selector2': d['action_selector2'],
            'avg.time_total': float(d['avg.time_total']) if 'avg.time_total' in d else None,
            'avg.num_submodels': float(d['avg.num_submodels']) if 'avg.num_submodels' in d else None,
            'sum.result_0': int(d['sum.result_0']),
            'sum.result_1': int(d['sum.result_1']),
            'sum.timeouts': int(d['sum.timeouts']),
            'results_tuple': (int(d['sum.result_1']), int(d['sum.result_0']), int(d['sum.timeouts'])),
            'stddev.time_total': float(d['stddev.time_total']) if 'stddev.time_total' in d else None,
            'max_simulations': int(d['max_simulations']),
            'rollout_count': int(d['rollout_count']),
            'initial_simulations': int(d['initial_simulations']),
            }


def create_report(data, report_dir_path, report_name):
    results_dir = report_dir_path
    if results_dir.exists():
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # text3 = data3.style \
    #     .format(precision=6, thousands=" ", decimal=".", na_rep="--") \
    #     .background_gradient(subset=[('Time')], axis=None) \
    #     .background_gradient(subset=[('TimeProcessor')], axis=None) \
    #     .applymap(lambda x: 'color: black; background-color: white' if pd.isnull(x) else '') \
    #     .to_latex(convert_css=True, hrules=True)


    report = evoplotter.reporting.ReportPDF(packages=["multirow"],
                                            geometry_params="[paperwidth=55cm, paperheight=100cm, margin=0.3cm]")

    s0 = evoplotter.reporting.SectionRelative("Full data")
    s0.add(data["table_full"])

    s1_exp1 = evoplotter.reporting.SectionRelative("Experiment 1")
    ss1 = evoplotter.reporting.SectionRelative(r"General results")
    ss1.add("""Naming convention:\\\\
    action\_selector1=1-best (proponent explores only 1 action)\\\\
    action\_selector1=all (proponent explores all actions)\\\\
    """)
    ss1.add(data["table_basic_time"])

    ss1_1 = evoplotter.reporting.SectionRelative(r"True benchmarks (satisfying the property)")
    ss1_1.add(data["table_true_01_timeouts"])
    ss1_1.add(data["table_true_01_time"])
    ss1_1.add(data["table_true_01_decision"])

    ss1_2 = evoplotter.reporting.SectionRelative(r"False benchmarks (not satisfying the property)")
    ss1_2.add(data["table_false_01_timeouts"])
    ss1_2.add(data["table_false_01_time"])
    ss1_2.add(data["table_false_01_decision"])
    # ss1.add(data["table_depthPercent_1"])
    # ss1.add(data["table_decision1_1"])
    # ss1.add(data["table_depthPercent_cases0_1"])
    # ss1.add(data["table_decision1_cases0_1"])

    # ss2 = evoplotter.reporting.SectionRelative(r"action\_selector1=all (proponent takes all actions)")
    # ss2.add(r"\begin{minipage}{25cm}"
    #         r"IsReduced=False means that a benchmark was processed as is by STV (a baseline we compare with)." + "\n"
    #         r"\end{minipage}\vspace{0.5cm}" + "\n\n" + r"\noindent")
    # ss2.add(data["table_depthPercent_all"])
    # ss2.add(data["table_decision1_all"])
    # ss2.add(data["table_depthPercent_cases0_all"])
    # ss2.add(data["table_decision1_cases0_all"])
    #
    # ss3 = evoplotter.reporting.SectionRelative("Other tables")
    # ss3.add(r"\begin{minipage}{25cm}"
    #         r" "
    #         r"" + "\n" + r"\end{minipage}\vspace{0.5cm}" + "\n\n" + r"\noindent")
    # ss3.add(data["table_time_by_selectors_k3"])
    # ss3.add(data["table_time_by_selectors_knon3"])
    # ss3.add(data["table_decision1_by_selectors_k3"])
    # ss3.add(data["table_decision1_by_selectors_knon3"])
    # ss3.add(data["table_timeouts_by_selectors_k3"])
    # ss3.add(data["table_timeouts_by_selectors_knon3"])
    # ss_errors = evoplotter.reporting.SectionRelative("Runs with STV errors (exit code != 0)")
    # ss_errors.add(text_errors)
    s1_exp1.add(ss1)
    s1_exp1.add(ss1_1)
    s1_exp1.add(ss1_2)
    # s1_exp1.add(ss2)
    # s1_exp1.add(ss3)
    report.add(s0)
    report.add(s1_exp1)
    f = results_dir / f"{report_name}.tex"
    report.save_and_compile(f, output_dir=f.parent)


def get_latex_table_default(df):
    return df.style \
           .format(precision=2, thousands=" ", decimal=".", na_rep="--") \
           .to_latex(convert_css=True, hrules=True)  #float_format=lambda x: "{:.2f}".format(x)


def get_latex_table_pivot1(df, values, columns, index=None, drop_list=None):
    if index is None:
        index = ["benchmark"]
    df = df.pivot_table(values=values, index=index, columns=columns, observed=True)  # , dropna=False
    if drop_list is not None:
        df = df.drop(drop_list, axis=1)
    # cmap="Greys",
    text = df.style \
        .format(precision=2, thousands=" ", decimal=".", na_rep="--") \
        .background_gradient(axis=None) \
        .applymap(lambda x: 'color: black; background-color: white' if pd.isnull(x) else '') \
        .to_latex(convert_css=True, hrules=True)
    # text = text.replace(r"action_selector1", "").replace("_", r"\_") + r"\\"
    text = text.replace("_", r"\_") + r"\\"
    return text


def get_latex_table_pivot1_other(df, values, columns):
    df = df.pivot_table(values=values, index=["benchmark"], columns=columns)  #, dropna=False
    if len(df.columns) < 3:
        return "Not enough columns"
    if "sum.result_0" in values:
        result_x = "sum.result_0"
    elif "sum.result_1" in values:
        result_x = "sum.result_1"
    else:
        raise Exception("sum.result_* was not specified!")
    # cmap="Greys",
    # .format(subset=["sum.timeouts", "sum.result_1"], precision=1, thousands=" ", decimal=".", na_rep="--") \
    # idx[:, ('sum.result_1', '1-best')]
    # .format(subset=[('sum.timeouts', '1-best'), ('sum.timeouts', 'all')], precision=1, thousands=" ", decimal=".", na_rep="--") \
    text = df.style \
        .format(subset=[('avg.time_total', '1-best'), ('avg.time_total', 'all')], precision=1, thousands=" ", decimal=".", na_rep="--") \
        .format(subset=[(result_x, '1-best'), (result_x, 'all')], precision=1, thousands=" ", decimal=".", na_rep="--") \
        .background_gradient(subset=[(result_x, '1-best'), (result_x, 'all')], axis=None) \
        .background_gradient(subset=[('avg.time_total', '1-best'), ('avg.time_total', 'all')], axis=None) \
        .applymap(lambda x: 'color: black; background-color: white' if pd.isnull(x) else '') \
        .to_latex(convert_css=True, hrules=True)
    text = text.replace(r"action_selector1", "").replace("_", r"\_") + r"\\"
    return text


def process_final(summary_folders, report_dir_path, report_name):
    data = {}
    dicts = []
    for sf in summary_folders:
        for p in load_properties_from_dir(sf):
            dicts.append(process_dict(p))


    df = pd.DataFrame.from_records(dicts)
    # df = df[df["max_simulations"] == 5000]
    df.sort_values(by=["benchmark", "action_selector1", "initial_simulations"], inplace=True)
    df_true = df[df["expected_results"] == True]
    df_false = df[df["expected_results"] == False]
    df_false = df[df["expected_results"] == False]
    print(df.dtypes)
    data["table_full"] = get_latex_table_default(df)
    data["table_basic_time"] = get_latex_table_pivot1(df, index=["benchmark", "expected_results"],
                                                      values=["avg.time_total", "sum.result_0", "sum.result_1"],
                                                      columns=["action_selector1", "initial_simulations"])

    data["table_true_01_timeouts"] = get_latex_table_pivot1(df_true, index=["benchmark"],
                                                      values=["sum.timeouts"],
                                                      columns=["action_selector1", "initial_simulations", "max_simulations"])
    data["table_true_01_time"] = get_latex_table_pivot1(df_true, index=["benchmark"],
                                                        values=["avg.time_total", "stddev.time_total"],
                                                        columns=["action_selector1", "initial_simulations", "max_simulations"])
    drop_list = [("sum.result_0", "1-best"), ("sum.result_1", "all")]
    data["table_true_01_decision"] = get_latex_table_pivot1(df_true, index=["benchmark"],
                                                        values=["sum.result_0", "sum.result_1"],
                                                        columns=["action_selector1", "initial_simulations", "max_simulations"],
                                                        drop_list=drop_list)

    # df_false_2 = df.copy()
    # df_false_2.drop(["expected_results"], axis=1, inplace=True)
    data["table_false_01_timeouts"] = get_latex_table_pivot1(df_false, index=["benchmark"],
                                                         values=["sum.timeouts"],
                                                         columns=["action_selector1", "initial_simulations"])
    data["table_false_01_time"] = get_latex_table_pivot1(df_false, index=["benchmark"],
                                                        values=["avg.time_total", "stddev.time_total"],
                                                        columns=["action_selector1", "initial_simulations"])
    drop_list = [("sum.result_0", "1-best"), ("sum.result_1", "all")]
    data["table_false_01_decision"] = get_latex_table_pivot1(df_false, index=["benchmark"],
                                                        values=["sum.result_0", "sum.result_1"],
                                                        columns=["action_selector1", "initial_simulations"],
                                                        drop_list=drop_list)
    # data["table_basic_decision"] = get_latex_table_pivot1(df, values=["avg.time_total"], columns=["action_selector1", "initial_simulations"])
    # data["table_time_by_selectors_knon3"] = get_latex_table_pivot1(df_exp_false, values=["avg.time_total"], columns=["action_selector1"])
    # data["table_decision1_by_selectors_k3"] = get_latex_table_pivot1_other(df_exp_true, values=["avg.time_total", "sum.result_1"], columns=["action_selector1"])
    # data["table_decision1_by_selectors_knon3"] = get_latex_table_pivot1_other(df_exp_false, values=["avg.time_total", "sum.result_0"], columns=["action_selector1"])
    # data["table_timeouts_by_selectors_k3"] = get_latex_table_pivot1(df_exp_true, values=["sum.timeouts"], columns=["action_selector1"])
    # data["table_timeouts_by_selectors_knon3"] = get_latex_table_pivot1(df_exp_false, values=["sum.timeouts"], columns=["action_selector1"])
    # print(df)

    # df2_1 = df[(df["expected_results"] == True) & (df["action_selector1"] == "1-best")]
    # df2_all = df[(df["expected_results"] == True) & (df["action_selector1"] == "all")]
    # df3_1 = df[(df["expected_results"] == False) & (df["action_selector1"] == "1-best")]
    # df3_all = df[(df["expected_results"] == False) & (df["action_selector1"] == "all")]

    # data["table_depthPercent_1"] = get_latex_table_pivot1(df2_1, values=["avg.time_total"], columns=["depth_percent"])
    # data["table_decision1_1"] = get_latex_table_pivot1(df2_1, values=["sum.result_1"], columns=["depth_percent"])
    # data["table_depthPercent_cases0_1"] = get_latex_table_pivot1(df3_1, values=["avg.time_total"], columns=["depth_percent"])
    # data["table_decision1_cases0_1"] = get_latex_table_pivot1(df3_1, values=["sum.result_0"], columns=["depth_percent"])

    # data["table_depthPercent_all"] = get_latex_table_pivot1(df2_all, values=["avg.time_total"], columns=["depth_percent"])
    # data["table_decision1_all"] = get_latex_table_pivot1(df2_all, values=["sum.result_1"], columns=["depth_percent"])
    # data["table_depthPercent_cases0_all"] = get_latex_table_pivot1(df3_all, values=["avg.time_total"], columns=["depth_percent"])
    # data["table_decision1_cases0_all"] = get_latex_table_pivot1(df3_all, values=["sum.result_0"], columns=["depth_percent"])

    create_report(data, report_dir_path=report_dir_path, report_name=report_name)



def process_final_E3(summary_folders, report_dir_path, report_name):
    data = {}
    dicts = []
    for sf in summary_folders:
        for p in load_properties_from_dir(sf):
            dicts.append(process_dict(p))


    df = pd.DataFrame.from_records(dicts)
    # df = df[df["max_simulations"] == 5000]
    df.sort_values(by=["benchmark", "action_selector1", "initial_simulations"], inplace=True)
    df_true = df[df["expected_results"] == True]
    df_false = df[df["expected_results"] == False]
    df_false = df[df["expected_results"] == False]
    print(df.dtypes)
    data["table_full"] = get_latex_table_default(df)
    data["table_basic_time"] = get_latex_table_pivot1(df, index=["benchmark", "expected_results"],
                                                      values=["avg.time_total", "sum.result_0", "sum.result_1"],
                                                      columns=["action_selector1", "initial_simulations", "max_simulations"])

    data["table_true_01_timeouts"] = get_latex_table_pivot1(df_true, index=["benchmark"],
                                                      values=["sum.timeouts"],
                                                      columns=["action_selector1", "initial_simulations", "max_simulations"])
    data["table_true_01_time"] = get_latex_table_pivot1(df_true, index=["benchmark"],
                                                        values=["avg.time_total", "stddev.time_total"],
                                                        columns=["action_selector1", "initial_simulations", "max_simulations"])
    drop_list = [("sum.result_0", "1-best"), ("sum.result_1", "all")]
    data["table_true_01_decision"] = get_latex_table_pivot1(df_true, index=["benchmark"],
                                                        values=["sum.result_0", "sum.result_1"],
                                                        columns=["action_selector1", "initial_simulations", "max_simulations"],
                                                        drop_list=drop_list)
    # drop_list = [("avg.num_submodels", "1-best"), ("avg.num_submodels", "all")]
    data["table_true_01_submodels"] = get_latex_table_pivot1(df_true, index=["benchmark"],
                                                              values=["avg.num_submodels"],
                                                              columns=["action_selector1", "initial_simulations",
                                                                       "max_simulations"])

    # df_false_2 = df.copy()
    # df_false_2.drop(["expected_results"], axis=1, inplace=True)
    data["table_false_01_timeouts"] = get_latex_table_pivot1(df_false, index=["benchmark"],
                                                         values=["sum.timeouts"],
                                                         columns=["action_selector1", "initial_simulations", "max_simulations"])
    data["table_false_01_time"] = get_latex_table_pivot1(df_false, index=["benchmark"],
                                                        values=["avg.time_total", "stddev.time_total"],
                                                        columns=["action_selector1", "initial_simulations", "max_simulations"])
    drop_list = [("sum.result_0", "1-best"), ("sum.result_1", "all")]
    data["table_false_01_decision"] = get_latex_table_pivot1(df_false, index=["benchmark"],
                                                        values=["sum.result_0", "sum.result_1"],
                                                        columns=["action_selector1", "initial_simulations", "max_simulations"],
                                                        drop_list=drop_list)
    data["table_false_01_submodels"] = get_latex_table_pivot1(df_false, index=["benchmark"],
                                                        values=["avg.num_submodels"],
                                                        columns=["action_selector1", "initial_simulations", "max_simulations"])

    results_dir = report_dir_path
    if results_dir.exists():
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    report = evoplotter.reporting.ReportPDF(packages=["multirow"],
                                            geometry_params="[paperwidth=55cm, paperheight=100cm, margin=0.3cm]")

    s0 = evoplotter.reporting.SectionRelative("Full data")
    s0.add(data["table_full"])

    s1_exp1 = evoplotter.reporting.SectionRelative("Experiment 1")
    ss1 = evoplotter.reporting.SectionRelative(r"General results")
    ss1.add("""Naming convention:\\\\
        action\_selector1=1-best (proponent explores only 1 action)\\\\
        action\_selector1=all (proponent explores all actions)\\\\
        """)
    ss1.add(data["table_basic_time"])

    ss1_1 = evoplotter.reporting.SectionRelative(r"True benchmarks (satisfying the property)")
    ss1_1.add(data["table_true_01_timeouts"])
    ss1_1.add(data["table_true_01_time"])
    ss1_1.add(data["table_true_01_decision"])
    ss1_1.add(data["table_true_01_submodels"])

    ss1_2 = evoplotter.reporting.SectionRelative(r"False benchmarks (not satisfying the property)")
    ss1_2.add(data["table_false_01_timeouts"])
    ss1_2.add(data["table_false_01_time"])
    ss1_2.add(data["table_false_01_decision"])
    ss1_2.add(data["table_false_01_submodels"])

    s1_exp1.add(ss1)
    s1_exp1.add(ss1_1)
    s1_exp1.add(ss1_2)
    report.add(s0)
    report.add(s1_exp1)
    f = results_dir / f"{report_name}.tex"
    report.save_and_compile(f, output_dir=f.parent)


# summary_folders = ["EXPERIMENTS_AAMAS27/E1[pure-mcts]/summary"]
# process_final(summary_folders, report_dir_path=Path("EXPERIMENTS_AAMAS27/REPORTS/final_report_E1"), report_name="final_report_E1")


# summary_folders = ["EXPERIMENTS_AAMAS27/E2[pure-mcts-steps]/summary", "EXPERIMENTS_AAMAS27/E2[pure-mcts-steps]_s200/summary"]
# process_final(summary_folders, report_dir_path=Path("EXPERIMENTS_AAMAS27/REPORTS/final_report_E2"), report_name="final_report_E2_s5000")



summary_folders = ["EXPERIMENTS_AAMAS27/E3[mcsa]/summary"]
process_final_E3(summary_folders, report_dir_path=Path("EXPERIMENTS_AAMAS27/REPORTS/final_report_E3"), report_name="final_report_E3")
