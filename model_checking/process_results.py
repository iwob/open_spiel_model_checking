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

x = os. listdir("exp_07_opponent_all/summary")

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


process_first()
# process_second()


