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

def process_dict(d):
    return {
            'm,n,k': d['m,n,k'],
            # 'max_game_depth': d['max_game_depth'],
            'avg.time_total': d['avg.time_total'],
            # 'sum.result_0': d['sum.result_0'],
            'sum.result_1': d['sum.result_1'],
            # 'stddev.time_total': d['stddev.time_total'],
            'avg.num_submodels': d['avg.num_submodels'],
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


def create_report():
    results_dir = Path("results_final")
    # text3 = data3.style \
    #     .format(precision=6, thousands=" ", decimal=".", na_rep="--") \
    #     .background_gradient(subset=[('Time')], axis=None) \
    #     .background_gradient(subset=[('TimeProcessor')], axis=None) \
    #     .applymap(lambda x: 'color: black; background-color: white' if pd.isnull(x) else '') \
    #     .to_latex(convert_css=True, hrules=True)


    report = evoplotter.reporting.ReportPDF(packages=["multirow"],
                                            geometry_params="[paperwidth=35cm, paperheight=100cm, margin=0.3cm]")
    s1_exp1 = evoplotter.reporting.SectionRelative("Experiment 1")
    ss1 = evoplotter.reporting.SectionRelative("Original data")
    # ss1.add(text)
    ss2 = evoplotter.reporting.SectionRelative("Processed data", label="sec:exp1-processed-data")
    ss2.add(r"\begin{minipage}{25cm}"
            r"Aggregating runs of the experiment by pairs (Benchmark, IsReduced) and averaging processing times. "
            r"IsReduced=True means that a random strategy was generated for all agents in the coalition, their local"
            r" models were reduced/cut to reflect this, new file representing model was generated, and then finally processed by STV. "
            r"IsReduced=False means that a benchmark was processed as is by STV (a baseline we compare with)." + "\n"
                                                                                                                 r"\end{minipage}\vspace{0.5cm}" + "\n\n" + r"\noindent")
    # ss2.add(text2)

    ss3 = evoplotter.reporting.SectionRelative("Processing time ratios")
    ss3.add(r"\begin{minipage}{25cm}"
            r"This table is computed based on the table in Section~\ref{sec:exp1-processed-data}. r/nr-ratio is computed "
            r"as (time reduced) / (time not reduced). A r/nr-ratio equal to 1 means that reduction had no impact on time. "
            r"We would like for the r/nr-ratio to be as small as possible, since that would imply that verifying a single "
            r"strategy is more efficient than verifying the whole problem. "
            r"" + "\n" + r"\end{minipage}\vspace{0.5cm}" + "\n\n" + r"\noindent")
    # ss3.add(text3)
    # ss_errors = evoplotter.reporting.SectionRelative("Runs with STV errors (exit code != 0)")
    # ss_errors.add(text_errors)
    # s1_exp1.add(ss1)  # a table with original data is too big; this data is accessible as a csv
    s1_exp1.add(ss2)
    s1_exp1.add(ss3)
    report.add(s1_exp1)
    f = results_dir / "report_final.tex"
    report.save_and_compile(f, output_dir=f.parent)


def process_final(summary_folders):
    dicts = []
    for sf in summary_folders:
        for p in load_properties_from_dir(sf):
            dicts.append(process_dict(p))

    data_opp = pd.DataFrame.from_records(dicts)
    print(data_opp)

    create_report()



summary_folders = ["results_final/summary", "results_final/summary_correct_timeout"]
process_final(summary_folders)

# process_first()
# process_second()


