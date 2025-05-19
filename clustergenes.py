import sklearn
import numpy as np
from tqdm.auto import tqdm


def ParabolicElbow(data, p, k):
    def f1(n):
        return n**2 / 2 + n / 2

    def f2(n):
        return n**3 / 3 + n**2 / 2 + n / 6

    def f4(n):
        return n**5 / 5 + n**4 / 2 + n**3 / 3 - n / 30

    lsigma = f1(k) / k
    lv = f2(k) - k * lsigma**2
    fsigma = f2(k - 1) / k
    fv = f4(k - 1) - k * fsigma**2

    k0 = k - 1

    while k0 < len(data):
        fsum = 0
        lsum = 0

        for i in range(k):
            fsum += (i**2 - fsigma) * (data[k0 - k + 1 + i] + p * i)
            lsum += (i + 1 - lsigma) * (data[k0 - k + 1 + i] + p * i)

        delta = fsum**2 / fv - lsum**2 / lv

        if delta > 0:
            return k0
        else:
            k0 += 1

    return k0


def CountClusters(dist, p, k):
    return len(dist) + 1 - ParabolicElbow(dist, p, k)


class GeneCluster:
    def __init__(
        self,
        patient_data,
        k=4,
        min_clusters=5,
        max_clusters=50,
        p_points_per_unit=50,
        use_tqdm=False,
    ):
        # --- initialization ---

        self.k_value = k
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.p_points_per_unit = p_points_per_unit
        self.clustering = sklearn.cluster.AgglomerativeClustering(
            n_clusters=1,
            compute_distances=True,
            linkage="ward",
            memory=".cache.clustergenes",
        )
        self.patient_data = patient_data

        # --- clustering ---

        self.clustering = self.clustering.fit(patient_data)

        p_max = 1

        while (
            CountClusters(self.clustering.distances_, p_max, self.k_value)
            >= self.min_clusters
        ):
            p_max *= 2

        p_space = np.linspace(0, p_max, p_max * self.p_points_per_unit)
        count_space = np.zeros_like(p_space)

        def range_wrapper(x):
            return x

        if use_tqdm:
            range_wrapper = tqdm

        for i in range_wrapper(range(len(p_space))):
            count_space[i] = CountClusters(
                self.clustering.distances_, p_space[i], self.k_value
            )

        last_idx = 0
        intervals = []

        for i in range(1, len(p_space)):
            count_prev = count_space[i - 1]

            if count_space[i] != count_prev:
                p_start = p_space[last_idx]
                p_end = p_space[i - 1]
                nclusters = int(count_prev)

                if nclusters <= max_clusters:
                    intervals.append(
                        dict(
                            nclusters=nclusters,
                            p_interval=[p_start, p_end],
                            p_length=p_end - p_start,
                        )
                    )
                last_idx = i

        intervals.sort(key=lambda x: x["p_length"], reverse=True)

        self.intervals = intervals

    def get_clusters(self, nclusters):
        self.clustering.n_clusters = nclusters
        self.clustering.fit(self.patient_data)

        clusters = [[] for _ in range(nclusters)]
        for i in range(len(self.patient_data)):
            clusters[self.clustering.labels_[i]].append(i)

        return clusters
