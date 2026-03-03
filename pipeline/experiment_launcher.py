from multiprocessing import Pool
from tqdm import tqdm

def run_experiment_sweep(param_grid, worker_function, n_workers=4):
    """Run a series of worker tasks using a multiprocessing pool.

    Parameters
    ----------
    param_grid : iterable
        Iterable of parameter tuples to feed to ``worker_function``.
    worker_function : callable
        A function (or partial) accepting a single tuple argument.
    n_workers : int, optional
        Number of processes to spawn (default is 4).
    """
    with Pool(n_workers, ) as pool:
        results = list(
            tqdm(
                pool.imap(worker_function, param_grid),
                total=len(param_grid)
            )
        )

    return results