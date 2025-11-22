import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from tqdm import trange

from openelm.configs import CVTMAPElitesConfig, MAPElitesConfig, QDConfig, EnvConfig
from openelm.environments import BaseEnvironment, Genotype
from openelm.algorithms.map_elites import Map
from openelm.evaluation_metrics import get_metric_names

Phenotype = Optional[np.ndarray]
MapIndex = Optional[tuple]
Individual = Tuple[np.ndarray, float]


class Pool:
    """The pool stores a set of solutions or individuals."""

    def __init__(self, pool_size: int):
        """Initializes an empty pool.

        Args:
            pool_size (int): The number of solutions to store in the pool.
            history_length (int): The number of historical solutions
                to maintain in the pool.
        """
        self.pool_size = pool_size
        self.pool = []

    def add(self, solution, fitness):
        """Adds a solution to the pool.

        If the pool is full, the oldest solution is removed. The solution
        is also added to the history.

        Args:
            solution: The solution to add to the pool.
        """
        # if there are not any individual yet, add it to the pool
        if len(self.pool) < self.pool_size:
            self.pool.append((solution, fitness))
            self.pool.sort(key=lambda x: x[1], reverse=True)
            return

        # if new fitness is better than the worst, add it to the pool
        if fitness > self.pool[-1][1]:
            if len(self.pool) >= self.pool_size:
                self.pool.pop(len(self.pool) - 1)
            self.pool.append((solution, fitness))
            # sort the pool by fitness
            self.pool.sort(key=lambda x: x[1], reverse=True)


class GABase:
    """
    Base class for a genetic algorithm
    """

    def __init__(
        self,
        env,
        config: QDConfig,
        env_config,
        data_id: int,
        init_pool: Optional[Pool] = None,
    ):
        """
        The base class for a genetic algorithm, implementing common functions and search.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (QDConfig): The configuration for the algorithm.
            init_pool (Pool, optional): A pool to use for the algorithm. If not passed,
            a new pool will be created. Defaults to None.
        """
        self.env: BaseEnvironment = env
        self.config: QDConfig = config
        self.env_config = env_config
        self.save_history = self.config.save_history
        self.save_snapshot_interval = self.config.save_snapshot_interval
        self.history_length = self.config.history_length
        self.start_step = 0
        self.save_np_rng_state = self.config.save_np_rng_state
        self.load_np_rng_state = self.config.load_np_rng_state
        self.rng = np.random.default_rng(self.config.seed)
        self.rng_generators = None
        self.start_step = 0
        self.data_id = data_id

        # self.history will be set/reset each time when calling `.search(...)`
        self.history: dict = defaultdict(list)
        self.fitness_history: dict = defaultdict(list)

        self._init_pool(init_pool, self.config.log_snapshot_dir)
        self._init_discretization()
        self._init_maps(log_snapshot_dir=self.config.log_snapshot_dir)
        print(f"MAP of size: {self.fitnesses.dims} = {self.fitnesses.map_size}")

    def _init_discretization(self):
        """Initializes the discretization of the behavior space."""
        raise NotImplementedError

    def _get_map_dimensions(self):
        """Returns the dimensions of the map."""
        raise NotImplementedError

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        raise NotImplementedError

    def visualize(self):
        """Visualizes the map."""
        pass

    def _init_maps(
        self, init_map: Optional[Map] = None, log_snapshot_dir: Optional[str] = None
    ):
        # perfomance of niches
        if init_map is None:
            self.map_dims = self._get_map_dimensions()
            self.fitnesses: Map = Map(
                dims=self.map_dims,
                fill_value=-np.inf,
                dtype=float,
                history_length=self.history_length,
            )
        else:
            self.map_dims = init_map.dims
            self.fitnesses = init_map

        # niches' sources
        self.genomes: Map = Map(
            dims=self.map_dims,
            fill_value=0.0,
            dtype=object,
            history_length=self.history_length,
        )
        
        # index over explored niches to select from
        self.nonzero: Map = Map(dims=self.map_dims, fill_value=False, dtype=bool)

        log_path = Path(log_snapshot_dir + f"data-{self.data_id}")
        if log_snapshot_dir and os.path.isdir(log_path):
            stem_dir = log_path.stem

            assert (
                "step_" in stem_dir
            ), f"loading directory ({stem_dir}) doesn't contain 'step_' in name"
            self.start_step = (
                int(stem_dir.replace("step_", "")) + 1
            )  # add 1 to correct the iteration steps to run

            with open(log_path / "config.json") as f:
                old_config = json.load(f)

            snapshot_path = log_path / "maps.pkl"
            assert os.path.isfile(
                snapshot_path
            ), f'{log_path} does not contain map snapshot "maps.pkl"'
            # first, load arrays and set them in Maps
            # Load maps from pickle file
            with open(snapshot_path, "rb") as f:
                maps = pickle.load(f)
            assert (
                self.genomes.array.shape == maps["genomes"].shape
            ), f"expected shape of map doesn't match init config settings, got {self.genomes.array.shape} and {maps['genomes'].shape}"

            self.genomes.array = maps["genomes"]
            self.fitnesses.array = maps["fitnesses"]
            self.nonzero.array = maps["nonzero"]
            # check if one of the solutions in the snapshot contains the expected genotype type for the run
            assert not np.all(
                self.nonzero.array is False
            ), "snapshot to load contains empty map"

            assert (
                self.env.config.env_name == old_config["env_name"]
            ), f'unmatching environments, got {self.env.config.env_name} and {old_config["env_name"]}'

            # compute top indices
            if hasattr(self.fitnesses, "top"):
                top_array = np.array(self.fitnesses.top)
                for cell_idx in np.ndindex(
                    self.fitnesses.array.shape[1:]
                ):  # all indices of cells in map
                    nonzero = np.nonzero(
                        self.fitnesses.array[(slice(None),) + cell_idx] != -np.inf
                    )  # check full history depth at cell
                    if len(nonzero[0]) > 0:
                        top_array[cell_idx] = nonzero[0][-1]
                # correct stats
                self.genomes.top = top_array.copy()
                self.fitnesses.top = top_array.copy()
            self.genomes.empty = False
            self.fitnesses.empty = False

            history_path = log_path / "history.pkl"
            if self.save_history and os.path.isfile(history_path):
                with open(history_path, "rb") as f:
                    self.history = pickle.load(f)
            with open((log_path / "fitness_history.pkl"), "rb") as f:
                self.fitness_history = pickle.load(f)

            if self.load_np_rng_state:
                with open((log_path / "np_rng_state.pkl"), "rb") as f:
                    self.rng_generators = pickle.load(f)
                    self.rng = self.rng_generators["qd_rng"]
                    self.env.set_rng_state(self.rng_generators["env_rng"])

            print("Loading finished")

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        raise NotImplementedError

    def _init_pool(
        self, init_map: Optional[Pool] = None, log_snapshot_dir: Optional[str] = None
    ):
        if init_map is None and log_snapshot_dir is None:
            self.pool = Pool(self.config.pool_size)
        elif init_map is not None and log_snapshot_dir is None:
            self.pool = init_map
        elif init_map is None and log_snapshot_dir is not None:
            self.pool = Pool(self.config.pool_size)
            log_path = Path(log_snapshot_dir)
            if log_snapshot_dir and os.path.isdir(log_path):
                stem_dir = log_path.stem

                assert (
                    "step_" in stem_dir
                ), f"loading directory ({stem_dir}) doesn't contain 'step_' in name"
                self.start_step = (
                    int(stem_dir.replace("step_", "")) + 1
                )  # add 1 to correct the iteration steps to run

                snapshot_path = log_path / "pool.pkl"
                assert os.path.isfile(
                    snapshot_path
                ), f'{log_path} does not contain map snapshot "pool.pkl"'
                # first, load arrays and set them in Maps
                # Load maps from pickle file
                with open(snapshot_path, "rb") as f:
                    self.pool = pickle.load(f)

        print("Loading finished")

    def random_selection(self) -> MapIndex:
        """Randomly select a niche (cell) in the map that has been explored."""
        return random.choice(self.pool.pool)

    def search(self, init_steps: int, total_steps: int, atol: float = 0.0) -> str:
        """
        Run the genetic algorithm.

        Args:
            initsteps (int): Number of initial random solutions to generate.
            totalsteps (int): Total number of steps to run the algorithm for,
                including initial steps.
            atol (float, optional): Tolerance for how close the best performing
                solution has to be to the maximum possible fitness before the
                search stops early. Defaults to 1.

        Returns:
            str: A string representation of the best perfoming solution. The
                best performing solution object can be accessed via the
                `current_max_genome` class attribute.
        """

        if self.niches_filled() == 0:
            max_fitness = -np.inf
            max_genome = None
        else:  # take max fitness in case of filled loaded snapshot
            max_fitness = self.max_fitness()
            max_index = np.where(self.fitnesses.latest == max_fitness)
            max_genome = self.genomes[max_index]
        if self.save_history:
            self.history = defaultdict(list)

        start_step = int(self.start_step)
        total_steps = int(total_steps)
        tbar = trange(start_step, total_steps, initial=start_step, total=total_steps)
        for n_steps in tbar:
            if n_steps < init_steps:
                # Initialise by generating initsteps random solutions
                new_individuals: list[Genotype] = self.env.random()
            else:
                # Randomly select a batch of individuals
                batch: list[Genotype] = []
                for _ in range(self.env.batch_size):
                    item = self.random_selection()
                    batch.append(item)
                # Mutate
                batch = [indv[0] for indv in batch]
                new_individuals = self.env.mutate(batch)

            for individual in new_individuals:
                # Evaluate fitness
                fitness = self.env.fitness(individual)
                if np.isinf(fitness):
                    continue
                self.pool.add(individual, fitness)
            
            fitnesses = np.array([individual[1] for individual in self.pool.pool])

            log_info = {f"Data {self.data_id}/Max GA fitness": fitnesses.max(), 
                        f"Data {self.data_id}/Mean GA fitness": fitnesses.mean(), 
                        f"Data {self.data_id}/Min GA fitness": fitnesses.min(), 
                        f"Step_data_{self.data_id}": n_steps,
            }
            
            self.fitness_history["max_ga"].append(fitnesses.max())
            self.fitness_history["mean_ga"].append(fitnesses.mean()) 
            self.fitness_history["min_ga"].append(fitnesses.min())   

            for name in get_metric_names():
                metric_arr = np.array([indiv[0].eval_metrics[name] for indiv in self.pool.pool])
                max_metric = metric_arr.max()
                mean_metric = metric_arr.min()
                min_metric = metric_arr.min()
                log_info[f"Data {self.data_id}/Max GA {name}"] = max_metric
                log_info[f"Data {self.data_id}/Mean GA {name}"] = mean_metric
                log_info[f"Data {self.data_id}/Min GA {name}"] = min_metric

            max_ga_fitness = fitnesses.max()
            tbar.set_description(f"{max_ga_fitness=:.4f}")


            if (
                self.save_snapshot_interval is not None
                and n_steps != 0
                and n_steps % self.save_snapshot_interval == 0
            ):
                self.save_results(step=n_steps)


        # Evaluate GA individuals in MAP-Elites
        print("="*40)
        print("Evaluate GA individuals in MAP-Elites")
        print("="*40)
        new_individuals = [individual[0] for individual in self.pool.pool]
        max_genome, max_fitness = self.update_map(
            new_individuals, max_genome, max_fitness
        )

        niches_filled = self.niches_filled()
        max_fitness = self.max_fitness()
        min_fitness = self.min_fitness()
        mean_fitness = self.mean_fitness()
        qd_score = self.qd_score()
        coverage = niches_filled / self.fitnesses.map_size

        log_info[f"Data {self.data_id}/Max fitness"] = max_fitness 
        log_info[f"Data {self.data_id}/Min fitness"] = min_fitness
        log_info[f"Data {self.data_id}/Mean fitness"] = mean_fitness
        log_info[f"Data {self.data_id}/Coverage"] = coverage
        log_info[f"Data {self.data_id}/QD Score"] = qd_score
        
        for name in get_metric_names():
            metric_arr = np.array([indiv.eval_metrics[name] if indiv != 0.0 else 0 for indiv in self.genomes.array.flatten()])
            qd_metric = metric_arr.sum()
            max_metric = metric_arr.max()
            mean_metric = metric_arr.mean()
            min_metric = metric_arr.min()
            log_info[f"Data {self.data_id}/QD {name}"] = qd_metric
            log_info[f"Data {self.data_id}/Max {name}"] = max_metric
            log_info[f"Data {self.data_id}/Mean {name}"] = mean_metric
            log_info[f"Data {self.data_id}/Min {name}"] = min_metric
        
        self.fitness_history["max"].append(self.max_fitness())
        self.fitness_history["min"].append(self.min_fitness())
        self.fitness_history["mean"].append(self.mean_fitness())
        self.fitness_history["qd_score"].append(self.qd_score())
        self.fitness_history["coverage"].append(coverage)

        self.current_max_genome = max_genome
        self.save_results(step=n_steps)
        self.visualize()
        return str(max_genome), log_info, self.fitness_history
    
    def update_map(self, new_individuals, max_genome, max_fitness):
        """
        Update the map if new individuals achieve better fitness scores.

        Args:
            new_individuals (list[Genotype]) : List of new solutions
            max_fitness : current maximum fitness

        Returns:
            max_genome : updated maximum genome
            max_fitness : updated maximum fitness

        """
        # `new_individuals` is a list of generation/mutation. We put them
        # into the behavior space one-by-one.
        for individual in new_individuals:
            fitness = self.env.fitness(individual)
            if np.isinf(fitness):
                continue
            phenotype = individual.to_phenotype()[np.array(self.env_config.behavior)] # Get the selected behavior to update the map
            map_ix = self.to_mapindex(phenotype)

            # if the return is None, the individual is invalid and is thrown
            # into the recycle bin.
            if map_ix is None:
                self.recycled[self.recycled_count % len(self.recycled)] = individual
                self.recycled_count += 1
                continue

            if self.save_history:
                # TODO: thresholding
                self.history[map_ix].append(individual)

            self.nonzero[map_ix] = True

            # If new fitness greater than old fitness in niche, replace.
            if fitness > self.fitnesses[map_ix]:
                self.fitnesses[map_ix] = fitness
                self.genomes[map_ix] = individual

            # update if new fitness is the highest so far.
            if fitness > max_fitness:
                max_fitness = fitness
                max_genome = individual

        return max_genome, max_fitness

    def niches_filled(self):
        """Get the number of niches that have been explored in the map."""
        return self.fitnesses.niches_filled

    def max_fitness(self):
        """Get the maximum fitness value in the map."""
        return self.fitnesses.max_finite

    def mean_fitness(self):
        """Get the mean fitness value in the map."""
        return self.fitnesses.mean

    def min_fitness(self):
        """Get the minimum fitness value in the map."""
        return self.fitnesses.min_finite

    def qd_score(self):
        """
        Get the quality-diversity score of the map.

        The quality-diversity score is the sum of the performance of all solutions
        in the map.
        """
        return self.fitnesses.qd_score


    def save_results(self, step: int):
        # create folder for dumping results and metadata
        output_folder = Path(self.config.output_dir + f"/data-{self.data_id}") / f"step_{step}"
        os.makedirs(output_folder, exist_ok=True)

        maps = {
            "fitnesses": self.fitnesses.array,
            "genomes": self.genomes.array,
            "nonzero": self.nonzero.array,
        }
        # Save maps as pickle file
        try:
            with open((output_folder / "maps.pkl"), "wb") as f:
                pickle.dump(maps, f)
        except Exception:
            pass


        # Save maps as pickle file
        try:
            with open((output_folder / "pools.pkl"), "wb") as f:
                pickle.dump(self.pool, f)
        except Exception:
            pass
        if self.save_history:
            with open((output_folder / "history.pkl"), "wb") as f:
                pickle.dump(self.history, f)

        with open((output_folder / "fitness_history.pkl"), "wb") as f:
            pickle.dump(self.fitness_history, f)

        # save numpy rng state to load if resuming from deterministic snapshot
        if self.save_np_rng_state:
            rng_generators = {
                "env_rng": self.env.get_rng_state(),
                "qd_rng": self.rng,
            }
            with open((output_folder / "np_rng_state.pkl"), "wb") as f:
                pickle.dump(rng_generators, f)

        # save env_name to check later, for verifying correctness of environment to run with snapshot load
        tmp_config = dict()
        tmp_config["env_name"] = self.env.config.env_name

        with open((output_folder / "config.json"), "w") as f:
            json.dump(tmp_config, f)
        f.close()

    def plot_fitness(self):
        import matplotlib.pyplot as plt

        save_path: str = self.config.output_dir
        plt.figure()
        plt.plot(self.fitness_history["max"], label="Max fitness")
        plt.plot(self.fitness_history["mean"], label="Mean fitness")
        plt.plot(self.fitness_history["min"], label="Min fitness")
        plt.legend()
        plt.savefig(f"{save_path}/MAPElites_fitness_history.png")
        plt.close("all")

        plt.figure()
        plt.plot(self.fitness_history["qd_score"], label="QD score")
        plt.legend()
        plt.savefig(f"{save_path}/MAPElites_qd_score.png")
        plt.close("all")

        plt.figure()
        plt.plot(self.fitness_history["niches_filled"], label="Niches filled")
        plt.legend()
        plt.savefig(f"{save_path}/MAPElites_niches_filled.png")
        plt.close("all")

        if len(self.map_dims) > 1:
            if len(self.fitnesses.dims) == 2:
                map2d = self.fitnesses.latest
                print(
                    "plotted genes:",
                    *[str(g) for g in self.genomes.latest.flatten().tolist()],
                )
            else:
                ix = tuple(np.zeros(max(1, len(self.fitnesses.dims) - 2), int))
                map2d = self.fitnesses.latest[ix]

                print(
                    "plotted genes:",
                    *[str(g) for g in self.genomes.latest[ix].flatten().tolist()],
                )

            plt.figure()
            plt.pcolor(map2d, cmap="inferno")
            plt.savefig(f"{save_path}/MAPElites_vis.png")
        plt.close("all")

    def visualize_individuals(self):
        """Visualize the genes of the best performing solution."""
        import matplotlib.pyplot as plt

        tmp = self.genomes.array.reshape(self.genomes.shape[0], -1)

        # if we're tracking history, rows will be the history dimension
        # otherwise, just the first dimension of the map
        plt.figure()
        _, axs = plt.subplots(nrows=tmp.shape[0], ncols=tmp.shape[1])
        for genome, ax in zip(tmp.flatten(), axs.flatten()):
            # keep the border but remove the ticks
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            try:
                genome.visualize(ax=ax)
            except AttributeError:
                pass
        save_path: str = self.config.output_dir
        plt.savefig(f"{save_path}/MAPElites_individuals.png")

class GA(GABase):
    """
    Class implementing MAP-Elites, a quality-diversity algorithm.

    MAP-Elites creates a map of high perfoming solutions at each point in a
    discretized behavior space. First, the algorithm generates some initial random
    solutions, and evaluates them in the environment. Then, it  repeatedly mutates
    the solutions in the map, and places the mutated solutions in the map if they
    outperform the solutions already in their niche.
    """

    def __init__(
        self,
        env,
        config: MAPElitesConfig,
        *args,
        **kwargs,
    ):
        """
        Class implementing MAP-Elites, a quality-diversity algorithm.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (MAPElitesConfig): The configuration for the algorithm.
        """
        self.map_grid_size = config.map_grid_size
        super().__init__(env=env, config=config, *args, **kwargs)

    def _init_discretization(self):
        """Set up the discrete behaviour space for the algorithm."""
        # TODO: make this work for any number of dimensions
        self.bins = np.linspace(*self.env.behavior_space, self.map_grid_size[0] + 1)[1:-1].T  # type: ignore

    def _get_map_dimensions(self):
        """Returns the dimensions of the map."""
        return self.map_grid_size * self.env.behavior_ndim

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        return (
            None
            if b is None
            else tuple(np.digitize(x, bins) for x, bins in zip(b, self.bins))
        )

    def visualize(self):
        """Visualize the map."""
        self.plot_fitness()


class CVTGA(GABase):
    """
    Class implementing CVT-MAP-Elites, a variant of MAP-Elites.

    This replaces the grid of niches in MAP-Elites with niches generated using a
    Centroidal Voronoi Tessellation. Unlike in MAP-Elites, we have a fixed number
    of total niches rather than a fixed number of subdivisions per dimension.
    """

    def __init__(
        self,
        env,
        config: CVTMAPElitesConfig,
        *args,
        **kwargs,
    ):
        """
        Class implementing CVT-MAP-Elites, a variant of MAP-Elites.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (CVTMAPElitesConfig): The configuration for the algorithm.
        """
        self.cvt_samples: int = config.cvt_samples
        self.n_niches: int = config.n_niches
        super().__init__(env=env, config=config, *args, **kwargs)

    def _init_discretization(self):
        """Discretize behaviour space using CVT."""
        # lower and upper bounds for each dimension
        low = self.env.behavior_space[0]
        high = self.env.behavior_space[1]

        points = np.zeros((self.cvt_samples, self.env.behavior_ndim))
        for i in range(self.env.behavior_ndim):
            points[:, i] = self.rng.uniform(low[i], high[i], size=self.cvt_samples)

        k_means = KMeans(init="k-means++", n_init="auto", n_clusters=self.n_niches)
        k_means.fit(points)
        self.centroids = k_means.cluster_centers_

        self.plot_centroids(points, k_means)

    def _get_map_dimensions(self):
        """Returns the dimensions of the map."""
        return (self.n_niches,)

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Maps a phenotype (position in behaviour space) to the index of the closest centroid."""
        return (
            None
            if b is None
            else (np.argmin(np.linalg.norm(b - self.centroids, axis=1)),)
        )

    def visualize(self):
        """Visualize the map."""
        self.plot_fitness()
        self.plot_behaviour_space()

    def plot_centroids(self, points, k_means):
        """
        Plot the CVT centroids and the points used to generate them.

        Args:
            points (np.ndarray, int): the points used to generate the centroids
            k_means (sklearn.cluster.KMeans): the k-means object used to generate the centroids
        """
        import matplotlib.pyplot as plt

        plt.figure()
        labels = k_means.labels_
        if self.env.behavior_ndim == 2:
            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(
                    i % 10
                )  # choose a color based on the cluster index
                plt.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    s=150,
                    marker="x",
                    color=color,
                    label=f"Niche {i}",
                )
                plt.scatter(
                    points[labels == i, 0],
                    points[labels == i, 1],
                    s=10,
                    marker=".",
                    color=color,
                )
        elif self.env.behavior_ndim >= 3:
            ax = plt.axes(projection="3d")

            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(
                    i % 10
                )  # choose a color based on the cluster index
                ax.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    self.centroids[i, 2],
                    s=150,
                    marker="x",
                    c=[color],
                    label=f"Niche {i}",
                )
                ax.scatter(
                    points[labels == i, 0],
                    points[labels == i, 1],
                    points[labels == i, 2],
                    s=10,
                    marker=".",
                    c=[color],
                )
        else:
            print("Not enough dimensions to plot centroids")
            return
        save_path: str = self.config.output_dir
        plt.savefig(f"{save_path}/MAPElites_centroids.png")

    def plot_behaviour_space(self):
        """Plot the first two dimensions (or three if available) of the behaviour space, along with the CVT centroids."""
        import matplotlib.pyplot as plt

        if self.env.behavior_ndim == 2:
            plt.figure()
            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(i % 10)
                plt.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    s=150,
                    marker="x",
                    color=color,
                    label=f"Niche {i}",
                )

                # get the first two dimensions for each behaviour in the history
                if self.genomes.history_length > 1:
                    phenotypes = [
                        g.to_phenotype()[:2]
                        for g in self.genomes.array[:, i]
                        if hasattr(g, "to_phenotype")
                    ]
                    if phenotypes:
                        hist = np.stack(phenotypes)
                        plt.scatter(
                            hist[:, 0], hist[:, 1], s=10, marker=".", color=color
                        )
                else:
                    g = self.genomes.array[i]
                    if hasattr(g, "to_phenotype"):
                        plt.scatter(
                            g.to_phenotype()[0],
                            g.to_phenotype()[1],
                            s=10,
                            marker=".",
                            color=color,
                        )

            plt.xlim([0, self.env.behavior_space[1, 0]])
            plt.ylim([0, self.env.behavior_space[1, 1]])

        elif self.env.behavior_ndim >= 3:
            plt.figure()
            ax = plt.axes(projection="3d")

            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(i % 10)
                ax.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    self.centroids[i, 2],
                    s=150,
                    marker="x",
                    c=[color],
                    label=f"Niche {i}",
                )

                # get the first three dimensions for each behaviour in the history
                if self.genomes.history_length > 1:
                    phenotypes = [
                        g.to_phenotype()[:3]
                        for g in self.genomes.array[:, i]
                        if hasattr(g, "to_phenotype")
                    ]
                    if phenotypes:
                        hist = np.stack(phenotypes)
                        ax.scatter(
                            hist[:, 0],
                            hist[:, 1],
                            hist[:, 2],
                            s=10,
                            marker=".",
                            c=[color],
                        )
                else:
                    g = self.genomes.array[i]
                    if hasattr(g, "to_phenotype"):
                        ax.scatter(
                            g.to_phenotype()[0],
                            g.to_phenotype()[1],
                            g.to_phenotype()[2],
                            s=10,
                            marker=".",
                            c=[color],
                        )

            ax.set_xlim([0, self.env.behavior_space[1, 0]])
            ax.set_ylim([0, self.env.behavior_space[1, 1]])
            ax.set_zlim([0, self.env.behavior_space[1, 2]])

        else:
            print("Not enough dimensions to plot behaviour space history")
            return
        save_path: str = self.config.output_dir
        plt.savefig(f"{save_path}/MAPElites_behaviour_history.png")