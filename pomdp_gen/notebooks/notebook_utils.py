import os, sys
sys.path.append(os.getcwd()+"/../..")
sys.path.append(os.getcwd()+"/..")
from bit_flip_experiments import *
from utils import *
from game import *
import pandas as pd

DIR_PREFIX = "../qalgorithm_synthesis/" # set proper string before running
MAX_HORIZON = 7
selected_backends = ["fake_sydney2", "fake_manhattan1", "fake_manhattan2", "fake_cambridge2"]
TAKE_BEST=True


def get_color(alg_index, algs_order, current_palette):
    for (index, alg_index_) in enumerate(algs_order):
        if alg_index_ == alg_index:
            return current_palette[index]
    assert False
    

def get_highest_advantage(lines):
    best_advantage = None
    for line in lines:
        current_advantage = line[2] - line[1]
        if best_advantage is None:
            best_advantage = current_advantage
        else:
            if current_advantage > best_advantage:
                best_advantage = current_advantage
    print(best_advantage)  
    for line in lines:
        current_advantage = line[2] - line[1]
        assert current_advantage <= best_advantage
        if current_advantage == best_advantage:
            print(line)
    assert best_advantage is not None
    return best_advantage

def get_backend_properties_df(backend):
    Precision.PRECISION = 7
    Precision.update_threshold()
    backends = []
    p0s = []
    p1s = []
    diffs = []
    abs_diffs = []
    probs_accum = []
    couplers_success_probs = []

    num_embeddings = get_num_embeddings(backend, DIR_PREFIX + "lambdas/")
    noise_model = get_ibm_noise_model(backend)
    for embedding_index in range(num_embeddings):
        backends.append(backend + str(embedding_index))
        embedding = load_embedding(backend, embedding_index, DIR_PREFIX)
        noise_data = noise_model.get_qubit_readout_error(embedding[2])
        probs = noise_data[0].probabilities
        success0 = probs[0][1]
        success1 = probs[1][1]
        p0s.append(success0)
        p1s.append(success1)
        diffs.append(success0-success1)
        abs_diffs.append(abs(success0 - success1))
        probs_accum.append(success0+success1)
    
        coupler0 = noise_model.get_noise_object(embedding[0], embedding[2], Instruction.CNOT).get_success_probability()
        coupler1 = noise_model.get_noise_object(embedding[1], embedding[2], Instruction.CNOT).get_success_probability()
        couplers_success_probs.append(coupler0 + coupler1)
    
    
    
    df = pd.DataFrame.from_dict({
        'hardware_spec': backends,
        'success0': p0s,
        'success1': p1s,
        'success_diff': diffs,
        'abs_diff': abs_diffs,
        'accum_prob': probs_accum,
        'couplers_success': couplers_success_probs
    })
    return df

def generate_all_lambdas_file(instruction_set=""):
    ''' Generates a file that merges all results of running bellman equation. The file contains the following columns: lambda, backend, embedding_index, horizon and time taken
        Set instruction_set="" for instruction set I0, otherwise set to "1"
    '''

    assert instruction_set == "" or instruction_set == "1"

    # declare a list for every column of the csv file we are generating
    lambdas = []
    backends = []
    embedding_indices = []
    horizons = []
    times = []
    for backend in backends_w_embs:
        lines = open(f"{DIR_PREFIX}lambdas{instruction_set}/{backend}.txt").readlines()[1:] # upper bound
        for line_ in lines:
            line = line_.replace("\n", "")
            elements = line.split(',')
            embedding_index = int(elements[0])
            horizon = int(elements[1])
            lambda_ = float(elements[2])
            t = int(elements[3])
            lambdas.append(lambda_)
            backends.append(backend)
            embedding_indices.append(embedding_index)
            times.append(t)
            horizons.append(horizon)
                
                
    df = pd.DataFrame.from_dict({
        'hardware_spec': backends,
        'embedding': embedding_indices,
        'horizon': horizons,
        'lambda': lambdas,
        'time': times
    })
    df.to_csv(f"{DIR_PREFIX}analysis_results{instruction_set}/all_lambdas.csv")

def get_different_algorithms_data(instruction_set="", serialize_dump=False):
    ''' Parses all the algorithms found for an instruction set, and compares control flow graph to determine different algorithms. If serialize_dump=True it dumps the algorithms into a file.
    '''
    all_algorithms = dict()
    comments = dict()
    for i in range(4, MAX_HORIZON+1):
        all_algorithms[i] = []
        comments[i] = []

    for backend in backends_w_embs:
        num_embeddings = len(get_backend_embeddings(backend))
        for index in range(num_embeddings):
            for horizon in range(4, MAX_HORIZON+1):
                try:
                    module_path = DIR_PREFIX + f'algorithms{instruction_set}/{backend}_{index}_{horizon}'
                    algorithms = load_algorithms_file(module_path)
                except:
                    print(f"Error: algorithms{instruction_set}/{backend}_{index}_{horizon}")
                if len(algorithms)  == 1:
                    for alg in algorithms:
                        assert (alg is not None)
                        if not (alg in all_algorithms[horizon]):
                            all_algorithms[horizon].append(alg)
                            comments[horizon].append(backend+'-'+str(index))
                        else:
                            for (index_old_alg, old_alg) in enumerate(all_algorithms[horizon]):
                                if old_alg == alg:
                                    comments[horizon][index_old_alg] = f"{comments[horizon][index_old_alg]},{backend}-{index}"
                                    break
                else:
                    assert len(algorithms) == 0
                

    if serialize_dump:
        for i in range(4, MAX_HORIZON+1):
            serialize_algorithms(all_algorithms[i], DIR_PREFIX + f'analysis_results{instruction_set}/diff{i}.py')
            dump_algorithms(all_algorithms[i], DIR_PREFIX + f'analysis_results{instruction_set}/diff{i}_ibm.py', for_ibm=True, comments=comments[i]) 
    return comments

def get_embedding(backend, embedding_index, instruction_set=""):
    ''' returns a dictionary of the embedding mapping virtual addresses to physical addresses
    '''
    f = open(DIR_PREFIX + f"inverse_mappings{instruction_set}/{backend}_{embedding_index}.txt")
    lines = f.readlines()
    result = dict()
    assert(len(lines) == 3)
    for line in lines:
        elements = line.split(' ')
        result[int(elements[1])] = int(elements[0])
    f.close()
    return result

def get_df_horizon_describe(df):    
    df_melted = df.melt(id_vars=["hardware_spec", "program"], value_vars=['success0', 'success1', 'success_diff', 'accum_prob', 'couplers_success'])
    result = df_melted.groupby(['program', 'variable']).describe()
    return result

def get_horizon_df(horizon, comments, is_one=False, dump=False, instruction_set="1"):
    Precision.PRECISION = 7
    Precision.update_threshold()
    backends = []
    horizons = []
    algorithms_index = []
    p0s = []
    p1s = []
    diffs = []
    probs_accum = []
    couplers_success_probs = []
    for (alg_index, comment) in enumerate(comments[horizon]):
            algo_backends = comment.split(",")
            for backend in algo_backends:
                temp = backend.split('-')
                backend_name = temp[0]
                embedding_index = int(temp[1])
                horizons.append(horizon)
                backends.append(backend)
                algorithms_index.append(alg_index)
                embedding = load_embedding(backend_name, embedding_index, DIR_PREFIX, is_one=is_one)
                noise_model = get_ibm_noise_model(backend_name)
                noise_data = noise_model.get_qubit_readout_error(embedding[2])
                probs = noise_data[0].probabilities
                success0 = round(probs[0][1],3)
                success1 = round(probs[1][1],3)
                p0s.append(success0)
                p1s.append(success1)
                diffs.append(success0-success1)
                probs_accum.append(success0+success1)
    
                coupler0 = noise_model.get_noise_object(embedding[0], embedding[2], Instruction.CNOT).get_success_probability()
                coupler1 = noise_model.get_noise_object(embedding[1], embedding[2], Instruction.CNOT).get_success_probability()
                couplers_success_probs.append(round(coupler0 + coupler1,3))
    
    
    
    df = pd.DataFrame.from_dict({
        'hardware_spec': backends,
        'program': algorithms_index,
        'success0': p0s,
        'success1': p1s,
        'success_diff': diffs,
        'accum_prob': probs_accum,
        'couplers_success': couplers_success_probs
    }) 
    df['couplers_success'] = df['couplers_success']/2
    df['accum_prob'] = df['accum_prob']/2

    if dump:
        df.to_csv(f"{DIR_PREFIX}analysis_results{instruction_set}/h{horizon}-per-algo.csv", index=False)
    return df
    
def get_num_algorithms(horizon, instruction_set=""):
    algorithms = load_algorithms_file(DIR_PREFIX + f'analysis_results{instruction_set}/diff{horizon}.py')
    return len(algorithms)

def get_similar_algorithms(horizon, algorithm_index, instruction_set=""):
    print(f"difference of algorithm={algorithm_index} at horizon={horizon}")
    backends_vs_file = open(DIR_PREFIX + f"analysis_results{instruction_set}/backends_vs.csv")
    lines = backends_vs_file.readlines()[1:]

    backends_to_acc = dict()
    for line in lines:
        elements = line.split(',')
        horizon_ = int(elements[0])
        algorithm_index_ = int(elements[1])
        backend = elements[2]
        if backend not in backends_to_acc.keys():
            backends_to_acc[backend] = dict()
        acc = float(elements[3])
        if (horizon_ == horizon):
            assert algorithm_index_ not in backends_to_acc[backend].keys()
            backends_to_acc[backend][algorithm_index_] = acc

    num_algorithms = get_num_algorithms(horizon, instruction_set=instruction_set)
    for alg_index_ in range(num_algorithms):
        accs_diffs = []
        for (backend, b_dict) in backends_to_acc.items():
            accs_diffs.append(abs(backends_to_acc[backend][algorithm_index] - backends_to_acc[backend][alg_index_] ))

        print(f"diff with alg. {alg_index_}: max={max(accs_diffs)} min={min(accs_diffs)} std={np.std(accs_diffs)} avg={np.average(accs_diffs)}")
    backends_vs_file.close()

def remove_numbers(name):
    numbers = [str(i) for i in range(0, 10)]

    result = ""
    for c in name:
        if c not in numbers:
            result += c
    return result

def get_traditional_df(take_best=False, instruction_set=""):
    backends = []
    horizons = []
    accuracies = []
    hardwares = []
    f = open(DIR_PREFIX + "analysis_results/backends_vs.csv")
    lines = f.readlines()[1:]
    best_accs = dict()
    for line in lines:
        elements = line.split(",")
        horizon = int(elements[0])
        alg_index = int(elements[1])
        backend_ = elements[2]
        if horizon < 7:
            if (alg_index == 0) and (backend_ in selected_backends):
                backend = backend_.replace("fake_", "").capitalize()
                acc = float(elements[3])
                
                if backend not in best_accs.keys():
                    best_accs[backend] = -1
                best_accs[backend] = max(best_accs[backend], acc)

                if (instruction_set == "1" and horizon == 6) or (instruction_set == ""):
                    backends.append(backend)
                    hardwares.append(remove_numbers(backend))
                    horizons.append(horizon)
                    if take_best:
                        accuracies.append(best_accs[backend])
                    else:
                        accuracies.append(acc)

    f.close()
    df = pd.DataFrame.from_dict({
        'hardware_spec': backends,
        'hardware': hardwares,
        'horizon': horizons,
        'accuracy': accuracies
    }) 

    return df

def get_backend_algorithm(backend_, horizon, emb_index, comments):
    backend = f"{backend_}-{emb_index}"
    algorithms = comments[horizon]
    for (index, comment) in enumerate(algorithms):
        elements = comment.split(',')
        if backend in elements:
            return index
    assert False

def get_lambda_from_file(backend, embedding, horizon, instruction_set=""):
    filename = DIR_PREFIX + f"lambdas{instruction_set}/{backend}.txt"
    f = open(filename)
    lines = f.readlines()[1:]
    for line in lines:
        elements = line.split(",")
        current_embedding = int(elements[0])
        current_horizon = int(elements[1])
        if current_embedding == embedding:
            if current_horizon == horizon:
                f.close()
                return round(float(elements[2]),3)
            
    f.close()
    assert False

def get_df_visualizing_lambdas(comments, algs_union, filter_out_non_advantage=False, only_own_alg=False, instruction_set="", take_best=True, get_only_trad=False):
    backends = []
    hardwares = []
    horizons = []
    algorithm_types = []
    alg_classes = []
    accs = []

    horizon_lines = {4: [], 5:[], 6: [], 7:[]}
    if instruction_set == "":
        min_horizon = 4
    else:
        min_horizon = 6
    for backend in backends_w_embs:
        num_embeddings = len(get_backend_embeddings(backend))
        for embedding in range(num_embeddings):
            for horizon in range(min_horizon, 8):
                alg_index = get_backend_algorithm(backend, horizon, embedding, comments)
                backend_acc = get_lambda_from_file(backend, embedding, horizon, instruction_set=instruction_set)
                if only_own_alg:
                    # here we load the lambda computed through bellman equation
                    algorithm_types.append("new")
                    backends.append(f"{backend}{embedding}")
                    hardwares.append(backend.replace("fake_", ""))
                    horizons.append(horizon)
                    accs.append(backend_acc)
                    alg_classes.append(algs_union[horizon][alg_index])
                else:
                    # here we consider the accuracy of the simulators
                    traditional_acc = get_algorithm_acc(horizon, 0, f"{backend}{embedding}", take_best=take_best)
                    if (instruction_set=="1" or (algs_union[horizon][alg_index] != 0 ))and (backend_acc - traditional_acc > 0):
                        if (not get_only_trad):
                            backends.append(f"{backend}{embedding}")
                            hardwares.append(backend.replace("fake_", ""))
                            horizons.append(horizon)
                            algorithm_types.append("trad")
                            accs.append(traditional_acc)
                            assert algs_union[horizon][0] == 0
                            alg_classes.append(int(algs_union[horizon][0]))
                            horizon_lines[horizon].append((f"{backend}{embedding}", min(backend_acc, traditional_acc), max(backend_acc, traditional_acc),algs_union[horizon][alg_index]))

                            algorithm_types.append("new")
                            backends.append(f"{backend}{embedding}")
                            hardwares.append(backend.replace("fake_", ""))
                            horizons.append(horizon)
                            accs.append(backend_acc)
                            if instruction_set == "1":
                                alg_classes.append(1)    
                            else:
                                alg_classes.append(int(algs_union[horizon][alg_index]))
                    elif not filter_out_non_advantage:
                        backends.append(f"{backend}{embedding}")
                        hardwares.append(backend.replace("fake_", ""))
                        horizons.append(horizon)
                        algorithm_types.append("trad")
                        accs.append(traditional_acc)
                        assert algs_union[horizon][0] == 0
                        if get_only_trad:
                            alg_classes.append(-1)
                        else:
                            alg_classes.append(algs_union[horizon][0])
                

    df = pd.DataFrame.from_dict({
        'hardware_spec': backends,
        'hardware': hardwares,
        'horizon': horizons,
        'program_type': algorithm_types,
        'alg_class': alg_classes,
        'accuracy': accs
    })
    return df, horizon_lines

def get_df_plots(horizon_to_algorithm, take_best=False, instruction_set=""):
    backends = []
    horizons = []
    accuracies = []
    hardwares = []
    f = open(DIR_PREFIX + f"analysis_results{instruction_set}/backends_vs.csv")
    lines = f.readlines()[1:]
    best_accs = dict()
    for line in lines:
        elements = line.split(",")
        horizon = int(elements[0])
        if (instruction_set == "" and horizon < 7) or (instruction_set == "1" and horizon == 6):
            alg_index = int(elements[1])
            backend_ = elements[2]
            algorithm = horizon_to_algorithm[horizon]
            if (alg_index == algorithm) and (backend_ in selected_backends):
                backend = backend_.replace("fake_", "").capitalize()
                hardwares.append(remove_numbers(backend))
                acc = float(elements[3])
                backends.append(backend)
                horizons.append(horizon)
                if backend not in best_accs.keys():
                    best_accs[backend] = -1
                best_accs[backend] = max(best_accs[backend], acc)
                if take_best:
                    accuracies.append(best_accs[backend])
                else:
                    accuracies.append(acc)

    f.close()
    df = pd.DataFrame.from_dict({
        'hardware_spec': backends,
        'hardware': hardwares,
        'horizon': horizons,
        'accuracy': accuracies
    }) 

    return df
    

# ONLY used for instruction set I0
def get_horizon_backend_algorithm(horizon, embedding, comments):
    ''' return the algorithm index that is optimal for an embedding
    '''
    for (index_algo, comment) in enumerate(comments[horizon]):
        backends = comment.split(",")
        for backend_ in backends:
            elements = backend_.split("-")
            if elements[0]+elements[1] == embedding:
                return index_algo
    assert False

def get_stats(horizon, comments, algs_union, get_trad_best=False):
    # open file where we have all the simulations results, all algorithms executed in all embeddings
    all = open(DIR_PREFIX + "analysis_results/backends_vs.csv")
    all_lines = all.readlines()[1:]
    programs_to_backends_to_accs = dict()
    for line in all_lines:
        elements = line.split(",")
        assert len(elements) == 5
        horizon_ = int(elements[0])
        alg_index = int(elements[1])
        hardware = elements[2]
        acc = float(elements[3])

        if horizon_ <= horizon:
            if alg_index not in programs_to_backends_to_accs.keys():
                programs_to_backends_to_accs[alg_index] = dict()
            if get_trad_best:
                if hardware in programs_to_backends_to_accs[alg_index].keys():
                    # in this case we are computing the max accuracy achieved in all horizons_ <= horizon
                    programs_to_backends_to_accs[alg_index][hardware] = max(acc, programs_to_backends_to_accs[alg_index][hardware])
                else:
                    programs_to_backends_to_accs[alg_index][hardware] = acc
            elif horizon_ == horizon:
                assert hardware not in programs_to_backends_to_accs[alg_index].keys()
                programs_to_backends_to_accs[alg_index][hardware] = acc
    all.close()

    programs_accuracies = dict()
    backends_from_accs = dict()
    for hardware in backends_w_embs:
        num_embeddings = get_num_embeddings(hardware, DIR_PREFIX + "lambdas/")
        for i in range(num_embeddings):
            backend = hardware + str(i)

            # get index of the optimal algorithm for this backend
            alg_index =  get_horizon_backend_algorithm(horizon, backend, comments)
            if alg_index not in programs_accuracies.keys():
                programs_accuracies[alg_index] = []
                assert alg_index not in backends_from_accs.keys()
                backends_from_accs[alg_index] = []
            # get accuracies achieved in all backends where this algorithm was the optimal
            programs_accuracies[algs_union[horizon][alg_index]].append(programs_to_backends_to_accs[alg_index][backend] - programs_to_backends_to_accs[0][backend])
            backends_from_accs[algs_union[horizon][alg_index]].append(backend)
    
    return programs_accuracies, backends_from_accs

def get_best_backends(accs_diffs, diff_backends):
    max_val = 0
    max_algs = []
    for (alg_index, values) in accs_diffs.items():
        if len(values) > 0:
            new_val = max(values)
            if new_val > max_val:
                max_algs = [alg_index]
                max_val = new_val
            elif new_val == max_val:
                max_algs.append(alg_index)
    print("algorithms: ", max_algs)
    for alg in max_algs:
        for (index, value) in enumerate(accs_diffs[alg]):
            if value == max_val:
                print(diff_backends[alg][index])

def get_algorithm_acc(horizon, alg_index, backend, instruction_set="", take_best=True):
    f = open(DIR_PREFIX + f"analysis_results{instruction_set}/backends_vs.csv")
    lines = f.readlines()[1:]
    acc = None
    for line in lines:
        elements = line.split(",")
        horizon_ = int(elements[0])
        alg_index_ = int(elements[1])
        backend_ = elements[2]
        if ((take_best and horizon_ <= horizon) or ((not take_best and horizon==horizon_))) and (alg_index == alg_index_) and (backend_ == backend):
            if acc is None:
                acc = float(elements[3])
            else:
                acc = max(acc, float(elements[3]))
    f.close()
    return acc

def get_chosen_algorithm_df(horizon, alg_index, instruction_set="", take_best=False):
    backends = []
    horizons = []
    accuracies = []
    hardwares = []
    for backend_ in selected_backends:
        acc = get_algorithm_acc(horizon, alg_index, backend_, instruction_set=instruction_set, take_best=take_best)
        backend = backend_.replace("fake_", "").capitalize()
        for h in range(4, 7):
            backends.append(backend)
            hardwares.append(remove_numbers(backend))
            horizons.append(h)
            accuracies.append(acc)
    df = pd.DataFrame.from_dict({
        'hardware_spec': backends,
        'hardware': hardwares,
        'horizon': horizons,
        'accuracy': accuracies
    }) 

    return df

# ONLY used for instruction set I1

def get_trad_new_algo_stats(take_best=True):
    Precision.PRECISION = 7
    Precision.update_threshold()
    backends = []
    horizons = []
    algorithms_index = []
    p0s = []
    p1s = []
    diffs = []
    probs_accum = []
    couplers_success_probs = []
    horizon = 6

    for backend in backends_w_embs:
        num_embeddings = get_num_embeddings(backend, DIR_PREFIX + "lambdas/")
        for embedding_index in range(num_embeddings):
            temp = backend.split('-')
            backend_name = temp[0]
            horizons.append(horizon)
            backends.append(backend)
            embedding = load_embedding(backend_name, embedding_index, DIR_PREFIX)
            noise_model = get_ibm_noise_model(backend_name)
            noise_data = noise_model.get_qubit_readout_error(embedding[2])
            probs = noise_data[0].probabilities
            success0 = probs[0][1]
            success1 = probs[1][1]
            p0s.append(success0)
            p1s.append(success1)
            diffs.append(success0-success1)
            probs_accum.append(success0+success1)

            coupler0 = noise_model.get_noise_object(embedding[0], embedding[2], Instruction.CNOT).get_success_probability()
            coupler1 = noise_model.get_noise_object(embedding[1], embedding[2], Instruction.CNOT).get_success_probability()
            couplers_success_probs.append(coupler0 + coupler1)

            trad_acc = get_algorithm_acc(horizon, 0, f"{backend}{embedding_index}", take_best=take_best)
            new_algo_acc = get_lambda_from_file(backend, embedding_index, horizon, instruction_set="1")
            if trad_acc >= new_algo_acc:
                algorithms_index.append(0)
            else:
                algorithms_index.append(1)

    df = pd.DataFrame.from_dict({
        'hardware_spec': backends,
        'program': algorithms_index,
        'success0': p0s,
        'success1': p1s,
        'success_diff': diffs,
        'accum_prob': probs_accum,
        'couplers_success': couplers_success_probs
    }) 
    df['couplers_success'] = df['couplers_success']/2
    df['accum_prob'] = df['accum_prob']/2
    return df