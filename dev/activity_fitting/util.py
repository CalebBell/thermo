from tqdm import tqdm
from fluids.numerics import linspace
from math import ceil
from thermo import UNIFAC, NRTL, RegularSolution, FloryHuggins, ChemicalConstantsPackage
from chemicals.identifiers import dippr_compounds, search_chemical
from chemicals.utils import hash_any_primitive
from functools import wraps
from joblib import Parallel, delayed
import csv
from itertools import combinations

def batch_pairs(chemicals, batch_size):
    """
    Generate batches of chemical pairs from a list of chemicals.

    This function creates all possible pairs of chemicals and then yields
    these pairs in batches of a specified size.

    Parameters
    ----------
    chemicals : list
        A list of chemical objects or identifiers.
    batch_size : int
        The number of chemical pairs in each batch.

    Yields
    ------
    list
        A batch of chemical pairs, where each pair is a tuple of two chemicals.

    Notes
    -----
    The function uses itertools.combinations to generate all possible pairs
    efficiently. If the number of pairs is not divisible by the batch_size,
    the last batch will be smaller than the others.

    Examples
    --------
    >>> chemicals = ['A', 'B', 'C', 'D']
    >>> list(batch_pairs(chemicals, 2))
    [[('A', 'B'), ('A', 'C')], [('A', 'D'), ('B', 'C')], [('B', 'D'), ('C', 'D')]]
    >>> list(batch_pairs(chemicals, 3))
    [[('A', 'B'), ('A', 'C'), ('A', 'D')], [('B', 'C'), ('B', 'D'), ('C', 'D')]]
    """
    total_combinations = list(combinations(chemicals, 2))
    for i in range(0, len(total_combinations), batch_size):
        yield total_combinations[i:i+batch_size]

STATS_METADATA =  ['MAE', 'STDEV']
WRITE_STATS = True

def write_to_csv(filename, data, fieldnames):
    """
    Write data to a CSV file.

    Parameters
    ----------
    filename : str
        The name of the file to write to.
    data : list of dict
        A list of dictionaries, where each dictionary represents a row of data.
    fieldnames : list
        A list of strings specifying the order of columns in the CSV file.

    Notes
    -----
    This function will overwrite the file if it already exists.
    It uses the csv module's DictWriter for writing the data.

    Examples
    --------
    >>> data = [
    ...     {'name': 'Alice', 'age': 30, 'city': 'New York'},
    ...     {'name': 'Bob', 'age': 25, 'city': 'Los Angeles'}
    ... ]
    >>> fieldnames = ['name', 'age', 'city']
    write_to_csv('example.csv', data, fieldnames)
    """
    if WRITE_STATS:
        fieldnames = fieldnames + STATS_METADATA
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore', delimiter='\t')
        
        writer.writeheader()  # Write the header row
        for row in data:
            writer.writerow(row)  # Write each data row
    
def create_constants_to_fit(required=('UNIFAC_groups',), compounds=list(dippr_compounds())):
    """
    Formaldehyde doesn't have groups

    >>> create_constants_to_fit(required=('UNIFAC_groups',), compounds=['50-00-0', '64-17-5']).CASs
    ['64-17-5']

    Regular solution
    >>> create_constants_to_fit(required=('Vml_STPs','solubility_parameters'), compounds=['50-00-0', '64-17-5']).solubility_parameters
    [21189.410882509575, 26088.180739851363]

    """
    use_IDs = []
    for c in compounds:
        try:
            search_chemical(c)  # Verifying the chemical exists
            use_IDs.append(c)
        except:
            pass
    
    # Get all chemical constants
    all_constants = ChemicalConstantsPackage.constants_from_IDs(IDs=use_IDs)
    valid_chemicals = []
    for i in range(all_constants.N):
        valid = True
        for requirement in required:
            if not getattr(all_constants, requirement)[i]:
                valid = False
                break
        if valid:
            valid_chemicals.append(all_constants.CASs[i])
    return ChemicalConstantsPackage.constants_from_IDs(IDs=valid_chemicals)

_constants_as_chemicals_cache = {}
def constants_as_chemicals(required, compounds=list(dippr_compounds())):
    '''
    >>> constants_as_chemicals(['UNIFAC_groups', 'Vml_STPs','solubility_parameters'], compounds=['124-18-5', '64-17-5'])
    [{'UNIFAC_groups': {1: 2, 2: 8}, 'Vml_STPs': 0.00019583836334716688, 'solubility_parameters': 15800.134474346805}, {'UNIFAC_groups': {1: 1, 2: 1, 14: 1}, 'Vml_STPs': 5.867599253092197e-05, 'solubility_parameters': 26088.180739851363}]
    '''
    if "CASs" not in required:
        required = list(required) + ['CASs']
    key = hash_any_primitive((required, compounds))
    if key in _constants_as_chemicals_cache:
        return _constants_as_chemicals_cache[key]
    constants = create_constants_to_fit(required=required, compounds=compounds)
    chemicals = []
    for i in range(len(constants.CASs)):
        chemical = {}
        for attr in required:
            chemical[attr] = getattr(constants, attr)[i]
        chemicals.append(chemical)
    _constants_as_chemicals_cache[key] = chemicals
    return chemicals



def get_binary_xs(pts=20):
    """
    >>> get_binary_xs(3)
    [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
    """
    x0s = linspace(0, 1, pts)
    many_xs = [[xi, 1-xi] for xi in x0s]
    return many_xs


REGISTERED_TASKS = {}

def register_task(model, name, key, metadata, required_constants):
    def decorator(func):
        if func.__name__ not in REGISTERED_TASKS:
            REGISTERED_TASKS[func.__name__] = []
        REGISTERED_TASKS[func.__name__].append({
            'model': model,
            'name': name,
            'key': key,
            'metadata': metadata,
            'func': func,
            'required_constants': required_constants,
        })
        return func
    return decorator

RS = 'Regular Solution'
REGULAR_SOLUTION_METADATA = ['CAS1', 'CAS2', 'lambda12', 'lambda21']
REGULAR_SOLUTION_SYMMETRIC_298 = 'RSSYM298'
REGULAR_SOLUTION_ASYMMETRIC_298 = 'RSASYM298'
REGULAR_SOLUTION_SYMMETRIC_373 = 'RSSYM373'
REGULAR_SOLUTION_ASYMMETRIC_373 = 'RSASYM373'
REGULAR_SOLUTION_SYMMETRIC_TFIT = 'RSSYMTFIT'
REGULAR_SOLUTION_ASYMMETRIC_TFIT = 'RSASYMTFIT'

def fit_regular_solution_binary(Ts, xs, gammas, Vs, SPs, symmetric=False):
    res, stats = RegularSolution.regress_binary_parameters(gammas=gammas, Ts=Ts, xs=xs, SPs=SPs, Vs=Vs, multiple_tries=True, do_statistics=True, symmetric=symmetric)
    res.update(stats)
    return res

def fit_all_regular_solution_options_to_gammas(Ts, xs, gammas, Vs, SPs):
    res_asymmetric = fit_regular_solution_binary(Ts, xs, gammas, Vs, SPs, symmetric=False)
    res_symmetric = fit_regular_solution_binary(Ts, xs, gammas, Vs, SPs, symmetric=True)
    res_symmetric['lambda21'] = res_symmetric['lambda12']
    return res_asymmetric, res_symmetric

def fit_all_regular_solution_to_model(Vs, SPs, model, xs=None, Ts=[298.15, 373.15], 
            names=[(REGULAR_SOLUTION_SYMMETRIC_298, REGULAR_SOLUTION_ASYMMETRIC_298), (REGULAR_SOLUTION_SYMMETRIC_373, REGULAR_SOLUTION_ASYMMETRIC_373)]):
    if xs is None:
        xs = get_binary_xs()
    params = {}
    all_gammas = []
    all_Ts = []
    all_xs = []
    for T, key_pairs in zip(Ts, names):
        many_gammas_expect = [model.to_T_xs(T=T, xs=x_binary).gammas() for x_binary in xs]
        Ts_pts = [T]*len(xs)
        res_asymmetric, res_symmetric = fit_all_regular_solution_options_to_gammas(Ts_pts, xs, many_gammas_expect, Vs, SPs)
        params[key_pairs[0]] = res_symmetric
        params[key_pairs[1]] = res_asymmetric
        all_gammas.extend(many_gammas_expect)
        all_Ts.extend(Ts_pts)
        all_xs.extend(xs)
    # Fit temperatures all
    res_asymmetric, res_symmetric = fit_all_regular_solution_options_to_gammas(all_Ts, all_xs, all_gammas, Vs, SPs)
    params[REGULAR_SOLUTION_SYMMETRIC_TFIT] = res_symmetric
    params[REGULAR_SOLUTION_ASYMMETRIC_TFIT] = res_asymmetric
    return params

REGULAR_SOLUTION_UNIFAC_CONSTANTS = ['UNIFAC_groups', 'Vml_STPs','solubility_parameters']
@register_task(RS, "UNIFAC Symmetric 298K", REGULAR_SOLUTION_SYMMETRIC_298, REGULAR_SOLUTION_METADATA, REGULAR_SOLUTION_UNIFAC_CONSTANTS)
@register_task(RS, "UNIFAC Asymmetric 298K", REGULAR_SOLUTION_ASYMMETRIC_298, REGULAR_SOLUTION_METADATA, REGULAR_SOLUTION_UNIFAC_CONSTANTS)
@register_task(RS, "UNIFAC Symmetric 373K", REGULAR_SOLUTION_SYMMETRIC_373, REGULAR_SOLUTION_METADATA, REGULAR_SOLUTION_UNIFAC_CONSTANTS)
@register_task(RS, "UNIFAC Asymmetric 373K", REGULAR_SOLUTION_ASYMMETRIC_373, REGULAR_SOLUTION_METADATA, REGULAR_SOLUTION_UNIFAC_CONSTANTS)
@register_task(RS, "UNIFAC Symmetric T-Fit", REGULAR_SOLUTION_SYMMETRIC_TFIT, REGULAR_SOLUTION_METADATA, REGULAR_SOLUTION_UNIFAC_CONSTANTS)
@register_task(RS, "UNIFAC Asymmetric T-Fit", REGULAR_SOLUTION_ASYMMETRIC_TFIT, REGULAR_SOLUTION_METADATA, REGULAR_SOLUTION_UNIFAC_CONSTANTS)
def fit_binary_regular_solution_to_UNIFAC(Vml_STPs, solubility_parameters, UNIFAC_groups):
    model = UNIFAC.from_subgroups(T=298.15, xs=[0.5, 0.5], chemgroups=UNIFAC_groups)
    fits = fit_all_regular_solution_to_model(Vml_STPs, solubility_parameters, model)
    return fits

REGULAR_SOLUTION_UNIFAC_DORTMUND_CONSTANTS = ['UNIFAC_Dortmund_groups', 'Vml_STPs','solubility_parameters']
@register_task(RS, "UNIFAC Dortmund Symmetric 298K", REGULAR_SOLUTION_SYMMETRIC_298, REGULAR_SOLUTION_METADATA, REGULAR_SOLUTION_UNIFAC_DORTMUND_CONSTANTS)
@register_task(RS, "UNIFAC Dortmund Asymmetric 298K", REGULAR_SOLUTION_ASYMMETRIC_298, REGULAR_SOLUTION_METADATA, REGULAR_SOLUTION_UNIFAC_DORTMUND_CONSTANTS)
@register_task(RS, "UNIFAC Dortmund Symmetric 373K", REGULAR_SOLUTION_SYMMETRIC_373, REGULAR_SOLUTION_METADATA, REGULAR_SOLUTION_UNIFAC_DORTMUND_CONSTANTS)
@register_task(RS, "UNIFAC Dortmund Asymmetric 373K", REGULAR_SOLUTION_ASYMMETRIC_373, REGULAR_SOLUTION_METADATA, REGULAR_SOLUTION_UNIFAC_DORTMUND_CONSTANTS)
@register_task(RS, "UNIFAC Dortmund Symmetric T-Fit", REGULAR_SOLUTION_SYMMETRIC_TFIT, REGULAR_SOLUTION_METADATA, REGULAR_SOLUTION_UNIFAC_DORTMUND_CONSTANTS)
@register_task(RS, "UNIFAC Dortmund Asymmetric T-Fit", REGULAR_SOLUTION_ASYMMETRIC_TFIT, REGULAR_SOLUTION_METADATA, REGULAR_SOLUTION_UNIFAC_DORTMUND_CONSTANTS)
def fit_binary_regular_solution_to_UNIFAC_DORTMUND(Vml_STPs, solubility_parameters, UNIFAC_Dortmund_groups):
    model = UNIFAC.from_subgroups(T=298.15, xs=[0.5, 0.5], chemgroups=UNIFAC_Dortmund_groups, version=1)
    fits = fit_all_regular_solution_to_model(Vml_STPs, solubility_parameters, model)
    return fits




NRTL_MODEL = 'NRTL'
NRTL_METADATA_B = ['CAS1', 'CAS2', 'tau12_b', 'tau21_b', 'alpha12', 'alpha21']
NRTL_METADATA_A = ['CAS1', 'CAS2', 'tau12_a', 'tau21_a', 'alpha12', 'alpha21']

NRTL_FORCED_ALPHA_298 = 'NRTLFA298'
NRTL_SYMMETRIC_ALPHA_298 = 'NRTLSYMALPHA298'
NRTL_ASYMMETRIC_ALPHA_298 = 'NRTLASYMALPHA298'

NRTL_FORCED_ALPHA_323 = 'NRTLFA323'
NRTL_SYMMETRIC_ALPHA_323 = 'NRTLSYMALPHA323'
NRTL_ASYMMETRIC_ALPHA_323 = 'NRTLASYMALPHA323'

NRTL_FORCED_ALPHA_373 = 'NRTLFA373'
NRTL_SYMMETRIC_ALPHA_373 = 'NRTLSYMALPHA373'
NRTL_ASYMMETRIC_ALPHA_373 = 'NRTLASYMALPHA373'

def fit_nrtl_binary(Ts, xs, gammas, symmetric_alpha=True, force_alpha=None):
    kwargs = {
        'gammas': gammas,
        'xs': xs,
        'symmetric_alphas': symmetric_alpha,
        'multiple_tries': True
    }
    if force_alpha is not None:
        kwargs['force_alpha'] = force_alpha

    res, stats = NRTL.regress_binary_parameters(**kwargs)
    res.update(stats)
    tau12, tau21 = res['tau12'], res['tau21']
    alpha12 = res.get('alpha12', force_alpha)
    alpha21 = res.get('alpha21', alpha12)
    
    res['alpha12'] = alpha12
    res['alpha21'] = alpha21
    res['tau12_a'] = tau12
    res['tau21_a'] = tau21
    res['tau12_b'] = Ts[0] * tau12
    res['tau21_b'] = Ts[0] * tau21
    return res

def fit_all_nrtl_options_to_gammas(Ts, xs, gammas):
    res_forced_alpha = fit_nrtl_binary(Ts, xs, gammas, force_alpha=0.3)
    res_fit_sym_alpha = fit_nrtl_binary(Ts, xs, gammas, symmetric_alpha=True)
    res_fit_asym_alpha = fit_nrtl_binary(Ts, xs, gammas, symmetric_alpha=False)
    return res_forced_alpha, res_fit_sym_alpha, res_fit_asym_alpha

def fit_all_nrtl_to_model(model, xs=None, Ts=[298.15, 323.15, 373.15], 
                          names=[
                              (NRTL_FORCED_ALPHA_298, NRTL_SYMMETRIC_ALPHA_298, NRTL_ASYMMETRIC_ALPHA_298),
                              (NRTL_FORCED_ALPHA_323, NRTL_SYMMETRIC_ALPHA_323, NRTL_ASYMMETRIC_ALPHA_323),
                              (NRTL_FORCED_ALPHA_373, NRTL_SYMMETRIC_ALPHA_373, NRTL_ASYMMETRIC_ALPHA_373)
                          ]):
    if xs is None:
        xs = get_binary_xs()
    params = {}
    for T, key_tuple in zip(Ts, names):
        many_gammas_expect = [model.to_T_xs(T=T, xs=x_binary).gammas() for x_binary in xs]
        Ts_pts = [T] * len(xs)
        results = fit_all_nrtl_options_to_gammas(Ts_pts, xs, many_gammas_expect)
        for key, result in zip(key_tuple, results):
            params[key] = result
    return params

NRTL_UNIFAC_CONSTANTS = ['UNIFAC_groups']
@register_task(NRTL_MODEL, "UNIFAC Forced Alpha 298K", NRTL_FORCED_ALPHA_298, NRTL_METADATA_A, NRTL_UNIFAC_CONSTANTS)
@register_task(NRTL_MODEL, "UNIFAC Symmetric Alpha 298K", NRTL_SYMMETRIC_ALPHA_298, NRTL_METADATA_A, NRTL_UNIFAC_CONSTANTS)
@register_task(NRTL_MODEL, "UNIFAC Asymmetric Alpha 298K", NRTL_ASYMMETRIC_ALPHA_298, NRTL_METADATA_A, NRTL_UNIFAC_CONSTANTS)
@register_task(NRTL_MODEL, "UNIFAC Forced Alpha 323K", NRTL_FORCED_ALPHA_323, NRTL_METADATA_A, NRTL_UNIFAC_CONSTANTS)
@register_task(NRTL_MODEL, "UNIFAC Symmetric Alpha 323K", NRTL_SYMMETRIC_ALPHA_323, NRTL_METADATA_A, NRTL_UNIFAC_CONSTANTS)
@register_task(NRTL_MODEL, "UNIFAC Asymmetric Alpha 323K", NRTL_ASYMMETRIC_ALPHA_323, NRTL_METADATA_A, NRTL_UNIFAC_CONSTANTS)
@register_task(NRTL_MODEL, "UNIFAC Forced Alpha 373K", NRTL_FORCED_ALPHA_373, NRTL_METADATA_A, NRTL_UNIFAC_CONSTANTS)
@register_task(NRTL_MODEL, "UNIFAC Symmetric Alpha 373K", NRTL_SYMMETRIC_ALPHA_373, NRTL_METADATA_A, NRTL_UNIFAC_CONSTANTS)
@register_task(NRTL_MODEL, "UNIFAC Asymmetric Alpha 373K", NRTL_ASYMMETRIC_ALPHA_373, NRTL_METADATA_A, NRTL_UNIFAC_CONSTANTS)
def fit_binary_nrtl_to_UNIFAC(UNIFAC_groups):
    model = UNIFAC.from_subgroups(T=298.15, xs=[0.5, 0.5], chemgroups=UNIFAC_groups)
    fits = fit_all_nrtl_to_model(model)
    return fits

NRTL_UNIFAC_DORTMUND_CONSTANTS = ['UNIFAC_Dortmund_groups']
@register_task(NRTL_MODEL, "Dortmund UNIFAC Forced Alpha 298K", NRTL_FORCED_ALPHA_298, NRTL_METADATA_A, NRTL_UNIFAC_DORTMUND_CONSTANTS)
@register_task(NRTL_MODEL, "Dortmund UNIFAC Symmetric Alpha 298K", NRTL_SYMMETRIC_ALPHA_298, NRTL_METADATA_A, NRTL_UNIFAC_DORTMUND_CONSTANTS)
@register_task(NRTL_MODEL, "Dortmund UNIFAC Asymmetric Alpha 298K", NRTL_ASYMMETRIC_ALPHA_298, NRTL_METADATA_A, NRTL_UNIFAC_DORTMUND_CONSTANTS)
@register_task(NRTL_MODEL, "Dortmund UNIFAC Forced Alpha 323K", NRTL_FORCED_ALPHA_323, NRTL_METADATA_A, NRTL_UNIFAC_DORTMUND_CONSTANTS)
@register_task(NRTL_MODEL, "Dortmund UNIFAC Symmetric Alpha 323K", NRTL_SYMMETRIC_ALPHA_323, NRTL_METADATA_A, NRTL_UNIFAC_DORTMUND_CONSTANTS)
@register_task(NRTL_MODEL, "Dortmund UNIFAC Asymmetric Alpha 323K", NRTL_ASYMMETRIC_ALPHA_323, NRTL_METADATA_A, NRTL_UNIFAC_DORTMUND_CONSTANTS)
@register_task(NRTL_MODEL, "Dortmund UNIFAC Forced Alpha 373K", NRTL_FORCED_ALPHA_373, NRTL_METADATA_A, NRTL_UNIFAC_DORTMUND_CONSTANTS)
@register_task(NRTL_MODEL, "Dortmund UNIFAC Symmetric Alpha 373K", NRTL_SYMMETRIC_ALPHA_373, NRTL_METADATA_A, NRTL_UNIFAC_DORTMUND_CONSTANTS)
@register_task(NRTL_MODEL, "Dortmund UNIFAC Asymmetric Alpha 373K", NRTL_ASYMMETRIC_ALPHA_373, NRTL_METADATA_A, NRTL_UNIFAC_DORTMUND_CONSTANTS)
def fit_binary_nrtl_to_UNIFAC_DORTMUND(UNIFAC_Dortmund_groups):
    model = UNIFAC.from_subgroups(T=298.15, xs=[0.5, 0.5], chemgroups=UNIFAC_Dortmund_groups, version=1)
    fits = fit_all_nrtl_to_model(model)
    return fits



def process_batch(batch):
    """
    Process a batch of chemical pairs for all tasks.

    Parameters
    ----------
    batch : list of tuple
        A list of chemical pairs to process. Each pair is a tuple of two chemical dictionaries.

    Returns
    -------
    dict
        A dictionary containing results for all tasks, with task names as keys.
    """
    batch_results = {task['name']: [] for tasks in REGISTERED_TASKS.values() for task in tasks}
    for chem1, chem2 in batch:
        pair_results = process_chemical_pair(chem1, chem2)
        for task_name, result in pair_results.items():
            batch_results[task_name].append(result)
    return batch_results

def process_chemical_pair(chem1, chem2):
    # Prepare kwargs for the task functions
    kwargs = {}
    for constant in chem1.keys():
        if constant != 'CASs':
            kwargs[constant] = [chem1.get(constant), chem2.get(constant)]
    # Compute the parameters once
    all_parameters = {}
    for func_name, tasks in REGISTERED_TASKS.items():
        for task in tasks:
            if task['func'].__name__ not in all_parameters:
                kwargs_copy = {k: v for k, v in kwargs.items() if k in task['required_constants']}
                all_parameters[task['func'].__name__] = task['func'](**kwargs_copy)
    # Collect results for all tasks
    pair_results = {}
    for func_name, tasks in REGISTERED_TASKS.items():
        for task in tasks:
            result = all_parameters[task['func'].__name__][task['key']]
            result.update({'CAS1': chem1['CASs'], 'CAS2': chem2['CASs']})
            pair_results[task['name']] = result
    return pair_results

def main():
    batch_size = 20
    n_jobs = 16  # Adjust this based on your system

    # Prepare chemical data
    compounds =list(dippr_compounds())[0:80]
    # compounds=['124-18-5', '64-17-5', '108-88-3', '7732-18-5', '106-97-8']
    chemicals = constants_as_chemicals(
        set.union(*[set(task['required_constants']) for tasks in REGISTERED_TASKS.values() for task in tasks]),
        compounds=compounds
    )

    # Create batches of chemical pairs
    chemical_batches = list(batch_pairs(chemicals, batch_size))

    # Initialize a dictionary to collect results per task
    task_results = {task['name']: [] for tasks in REGISTERED_TASKS.values() for task in tasks}

    # Process batches

    # Process batches in parallel
    batch_results_list = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_batch)(batch) for batch in tqdm(chemical_batches, desc="Processing batches", unit="batch")
    )
    print('Completed parallel fitting')
    print('-'*1000)

    # Collect results per task
    for batch_results in batch_results_list:
        for task_name, results in batch_results.items():
            task_results[task_name].extend(results)
    

            
    # Write results to CSV per task
    for func_name, tasks in REGISTERED_TASKS.items():
        for task in tasks:
            filename = f"{task['model']} {task['name']}.csv".replace(' ', '_')
            write_to_csv(filename, task_results[task['name']], task['metadata'])
            print(f"Task {task['name']} completed and saved to {filename}")




if __name__ == "__main__":
    main()
# main()