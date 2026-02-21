from multiprocessing import Pool
from tqdm import tqdm

def run_experiment_sweep(param_grid, worker_function, n_workers=4):
    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(worker_function, param_grid),
                total=len(param_grid)
            )
        )

    return results