import os
import numpy as np
import pandas as pd

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

def process_dict(d):
    return {
            'm,n,k': d['m,n,k'],
            'n': int(d['m,n,k'].split(',')[0][1:]),
            'k': int(d['m,n,k'].split(',')[2][:-1].strip()),
            'max_game_depth': int(d['max_game_depth']),
            'depth_percent': map_max_depth_to_depth_percent(int(d['m,n,k'].split(',')[0][1:]), int(d['max_game_depth'])),
            'action_selector1': d['action_selector1'],
            'action_selector2': d['action_selector2'],
            'avg.time_total': float(d['avg.time_total']) if 'avg.time_total' in d else None,
            'avg.num_submodels': float(d['avg.num_submodels']) if 'avg.num_submodels' in d else None,
            # 'sum.result_0': int(d['sum.result_0']),
            'sum.result_1': int(d['sum.result_1']),
            'sum.timeouts': int(d['sum.timeouts']),
            'stddev.time_total': float(d['stddev.time_total']) if 'stddev.time_total' in d else None,
            }

def process_first():
    data_opp = pd.DataFrame.from_records([process_dict(p) for p in load_properties_from_dir("exp_07_opponent_all/summary") + load_properties_from_dir("exp_07_experiments_opp/summary")],
                                         index='m,n,k')
    data_pro = pd.DataFrame.from_records([process_dict(p) for p in load_properties_from_dir("exp_07_proponent_all/summary") + load_properties_from_dir("exp_07_experiments_pro/summary")],
                                         index='m,n,k')
    data_opp_bench2 = pd.DataFrame.from_records([process_dict(p) for p in load_properties_from_dir("exp_07_opponent_all_bench2/summary")],
                                         index='m,n,k')
    # data_pro_bench2 = pd.DataFrame.from_records([process_dict(p) for p in load_properties_from_dir("exp_07_proponent_all_bench2/summary")],
    #                                      index='m,n,k')

    # data_opp['m,n,k (copy)'] = data_opp.index

    print("data_pro")
    print(data_pro)
    print("data_opp")
    print(data_opp)
    # print("data_pro_bench2")
    # print(data_pro_bench2)
    print("data_opp_bench2")
    print(data_opp_bench2)
    data_opp_bench2.rename(columns={'avg.time_total': 'avg.time_total (pro=1-best)'}, inplace=True)
    # data_opp_bench2.rename(columns={'sum.result_0': 'sum.result_0 (pro=1-best)'}, inplace=True)
    data_opp_bench2.rename(columns={'sum.result_1': 'sum.result_1 (pro=1-best)'}, inplace=True)
    data_opp_bench2.rename(columns={'avg.num_submodels': 'avg.num_submodels (pro=1-best)'}, inplace=True)

    # data['avg.time_total (pro=all)'] = data['m,n,k'].map(data_pro['avg.time_total'])


    data = data_opp.merge(data_pro, how='outer', on='m,n,k', suffixes=(' (pro=1-best)', ' (pro=all)'), indicator=False)
    data = data_opp.merge(data_pro, how='outer', on='m,n,k', suffixes=(' (pro=1-best)', ' (pro=all)'), indicator=False)
    # data = data_opp.merge(data_pro, how='outer', on='m,n,k', suffixes=(' (pro=1-best)', ' (pro=all)'), indicator=False)

    # data = pd.concat([data, data_opp_bench2])
    data["avg.time_total (pro=1-best)"] = pd.to_numeric(data["avg.time_total (pro=1-best)"])
    data["avg.time_total (pro=all)"] = pd.to_numeric(data["avg.time_total (pro=all)"])
    data["MCMAS"] = np.nan
    data.loc['(4, 4, 3)', 'MCMAS'] = 91.218


    def sort_fun(x):
        vec = []
        for v in x.values:
            x1 = int(v.split(',')[0].replace('(', '').strip())
            x2 = int(v.split(',')[1].strip())
            x3 = int(v.split(',')[2].replace(')', '').strip())
            vec.append(x1 * 10**4 + x2 * 10**2 + x3)
        print(vec)
        return pd.Index(vec)

    # data = data.sort_index(key=lambda x: pd.Index([int(v.split(',')[0].replace('(', '')) for v in x.values]))
    data = data.sort_index(key=sort_fun)

    print()
    print("Final data frame:")
    print(data.dtypes)
    print(data.to_string(float_format=lambda x: "{:.1f}".format(x)))

    # Some changing places
    data = move_row_to_last_id(data, "(4, 4, 4)")
    data = move_row_to_last_id(data, "(5, 5, 4)")
    data = move_row_to_last_id(data, "(5, 5, 5)")
    data = move_row_to_last_id(data, "(6, 6, 5)")

    print()
    print("Final data frame 2:")
    print(data.dtypes)
    print(data.to_string(float_format=lambda x: "{:.1f}".format(x)))

    # Results from MCMAS
    # mnk(4,4,3).ispl has been parsed successfully.
    # Global syntax checking...
    # 1
    # 1
    # 1
    # Done
    # Encoding BDD parameters...
    # Building partial transition relation...
    # Building BDD for initial states...
    # Building reachable state space...
    # Checking formulae...
    # Verifying properties...
    #   Formula number 1: (<cross>F (crosswins && (! noughtwins))), is TRUE in the model
    # done, 1 formulae successfully read and checked
    # execution time = 91.218
    # number of reachable states = 1.01658e+07
    # BDD memory in use = 166719696


    # mnk(6,6,5).ispl has been parsed successfully.
    # Global syntax checking...
    # 1
    # 1
    # 1
    # Done
    # Encoding BDD parameters...
    # Building partial transition relation...
    # Building BDD for initial states...
    # Building reachable state space...
    # NOTE: terminated after 2h



def process_second():
    data_opp = pd.DataFrame.from_records([process_dict(p) for p in load_properties_from_dir("exp_07_experiments_pro_f/summary")])

    print("data_opp")
    print(data_opp)
    data_opp_bench2.rename(columns={'avg.time_total': 'avg.time_total (pro=1-best)'}, inplace=True)
    # data_opp_bench2.rename(columns={'sum.result_0': 'sum.result_0 (pro=1-best)'}, inplace=True)
    data_opp_bench2.rename(columns={'sum.result_1': 'sum.result_1 (pro=1-best)'}, inplace=True)
    data_opp_bench2.rename(columns={'avg.num_submodels': 'avg.num_submodels (pro=1-best)'}, inplace=True)

    # data['avg.time_total (pro=all)'] = data['m,n,k'].map(data_pro['avg.time_total'])

    data = data_opp.merge(data_pro, how='outer', on='m,n,k', suffixes=(' (pro=1-best)', ' (pro=all)'), indicator=False)
    data = data_opp.merge(data_pro, how='outer', on='m,n,k', suffixes=(' (pro=1-best)', ' (pro=all)'), indicator=False)
    # data = data_opp.merge(data_pro, how='outer', on='m,n,k', suffixes=(' (pro=1-best)', ' (pro=all)'), indicator=False)

    # data = pd.concat([data, data_opp_bench2])
    data["avg.time_total (pro=1-best)"] = pd.to_numeric(data["avg.time_total (pro=1-best)"])
    data["avg.time_total (pro=all)"] = pd.to_numeric(data["avg.time_total (pro=all)"])
    data["MCMAS"] = np.nan
    data.loc['(4, 4, 3)', 'MCMAS'] = 91.218

    def sort_fun(x):
        vec = []
        for v in x.values:
            x1 = int(v.split(',')[0].replace('(', '').strip())
            x2 = int(v.split(',')[1].strip())
            x3 = int(v.split(',')[2].replace(')', '').strip())
            vec.append(x1 * 10 ** 4 + x2 * 10 ** 2 + x3)
        print(vec)
        return pd.Index(vec)

    # data = data.sort_index(key=lambda x: pd.Index([int(v.split(',')[0].replace('(', '')) for v in x.values]))
    data = data.sort_index(key=sort_fun)

    print()
    print("Final data frame:")
    print(data.dtypes)
    print(data.to_string(float_format=lambda x: "{:.1f}".format(x)))

    # Some changing places
    data = move_row_to_last_id(data, "(4, 4, 4)")
    data = move_row_to_last_id(data, "(5, 5, 4)")
    data = move_row_to_last_id(data, "(5, 5, 5)")
    data = move_row_to_last_id(data, "(6, 6, 5)")

    print()
    print("Final data frame 2:")
    print(data.dtypes)
    print(data.to_string(float_format=lambda x: "{:.1f}".format(x)))


import evoplotter.reporting
from pathlib import Path
import shutil


def create_report(data, report_dir_path):
    results_dir = report_dir_path
    if results_dir.exists():
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    # text3 = data3.style \
    #     .format(precision=6, thousands=" ", decimal=".", na_rep="--") \
    #     .background_gradient(subset=[('Time')], axis=None) \
    #     .background_gradient(subset=[('TimeProcessor')], axis=None) \
    #     .applymap(lambda x: 'color: black; background-color: white' if pd.isnull(x) else '') \
    #     .to_latex(convert_css=True, hrules=True)


    report = evoplotter.reporting.ReportPDF(packages=["multirow"],
                                            geometry_params="[paperwidth=35cm, paperheight=100cm, margin=0.3cm]")

    s0 = evoplotter.reporting.SectionRelative("Full data")
    s0.add(data["table_full"])

    s1_exp1 = evoplotter.reporting.SectionRelative("Experiment 1")
    ss1 = evoplotter.reporting.SectionRelative(r"action\_selector1=1-best (proponent chooses only 1 action)")
    ss1.add(data["table_depthPercent_1"])
    ss1.add(data["table_decision1_1"])
    ss1.add(data["table_depthPercent_cases0_1"])
    ss1.add(data["table_decision1_cases0_1"])

    ss2 = evoplotter.reporting.SectionRelative(r"action\_selector1=all (proponent takes all actions)")
    ss2.add(r"\begin{minipage}{25cm}"
            r"IsReduced=False means that a benchmark was processed as is by STV (a baseline we compare with)." + "\n"
            r"\end{minipage}\vspace{0.5cm}" + "\n\n" + r"\noindent")
    ss2.add(data["table_depthPercent_all"])
    ss2.add(data["table_decision1_all"])
    ss2.add(data["table_depthPercent_cases0_all"])
    ss2.add(data["table_decision1_cases0_all"])

    ss3 = evoplotter.reporting.SectionRelative("Processing time ratios")
    ss3.add(r"\begin{minipage}{25cm}"
            r"This table is computed based on the table in Section~\ref{sec:exp1-processed-data}. r/nr-ratio is computed "
            r"" + "\n" + r"\end{minipage}\vspace{0.5cm}" + "\n\n" + r"\noindent")
    # ss3.add(text3)
    # ss_errors = evoplotter.reporting.SectionRelative("Runs with STV errors (exit code != 0)")
    # ss_errors.add(text_errors)
    s1_exp1.add(ss1)
    s1_exp1.add(ss2)
    # s1_exp1.add(ss3)
    report.add(s0)
    report.add(s1_exp1)
    f = results_dir / "report_final.tex"
    report.save_and_compile(f, output_dir=f.parent)


def get_latex_table_default(df):
    return df.to_latex(float_format=lambda x: "{:.2f}".format(x))


def get_latex_table_pivot1(df, values):
    df = df.pivot_table(values=values, index=["n", "k"], columns=["depth_percent"], dropna=False)
    # cmap="Greys",
    text = df.style \
        .format(precision=2, thousands=" ", decimal=".", na_rep="--") \
        .background_gradient(axis=None) \
        .applymap(lambda x: 'color: black; background-color: white' if pd.isnull(x) else '') \
        .to_latex(convert_css=True, hrules=True)
    text = text.replace("_", r"\_") + r"\\"  # .replace("0000", "")
    return text


def process_final(summary_folders, report_dir_path):
    data = {}
    dicts = []
    for sf in summary_folders:
        for p in load_properties_from_dir(sf):
            dicts.append(process_dict(p))


    df = pd.DataFrame.from_records(dicts)
    df.sort_values(by=["n", "k", "max_game_depth", "action_selector1", "action_selector2"], inplace=True)
    print(df.dtypes)
    data["table_full"] = get_latex_table_default(df)
    print(df)

    df2_1 = df[(df["k"] == 3) & (df["action_selector1"] == "1-best")]
    df2_2 = df[(df["k"] == 3) & (df["action_selector1"] == "all")]
    df3_1 = df[(df["k"] != 3) & (df["action_selector1"] == "1-best")]
    df3_2 = df[(df["k"] != 3) & (df["action_selector1"] == "all")]

    data["table_depthPercent_1"] = get_latex_table_pivot1(df2_1, values=["avg.time_total"])
    data["table_decision1_1"] = get_latex_table_pivot1(df2_1, values=["sum.result_1"])
    data["table_depthPercent_cases0_1"] = get_latex_table_pivot1(df3_1, values=["avg.time_total"])
    data["table_decision1_cases0_1"] = get_latex_table_pivot1(df3_1, values=["sum.result_1"])

    data["table_depthPercent_all"] = get_latex_table_pivot1(df2_2, values=["avg.time_total"])
    data["table_decision1_all"] = get_latex_table_pivot1(df2_2, values=["sum.result_1"])
    data["table_depthPercent_cases0_all"] = get_latex_table_pivot1(df3_2, values=["avg.time_total"])
    data["table_decision1_cases0_all"] = get_latex_table_pivot1(df3_2, values=["sum.result_1"])

    create_report(data, report_dir_path=report_dir_path)



# summary_folders = ["exp_final/summary", "exp_final_correct_timeout/summary"]
summary_folders = ["exp_final_tmp/summary_1", "exp_final_tmp/summary_2"]
process_final(summary_folders, report_dir_path=Path("exp_final_report"))


summary_folders = ["exp_final_r2s2000/summary"]
process_final(summary_folders, report_dir_path=Path("exp_final_report_r2s2000"))

# process_first()
# process_second()


