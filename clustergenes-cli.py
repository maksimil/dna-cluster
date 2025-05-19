import clustergenes
import csv
import sys
import numpy as np
from matplotlib import pyplot as plt
import os


def SaveCluster(filename, data, cluster_idx):
    cluster_data = []
    for j in cluster_idx:
        cluster_data.append(data[j])
    cluster_data = np.array(cluster_data)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cluster_data.T, aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    fig.savefig(filename)

    plt.close()

if len(sys.argv) != 4:
    print(
        "Usage python3 clustergenes-cli.py [genes csv file] [n results] [output folder]"
    )
    exit(1)

datafilename = sys.argv[1]
ntopresults = int(sys.argv[2])
outfolder = sys.argv[3]

gene_names = []
patient_data = []
patient_names = []

with open(datafilename, "r") as datafile:
    datacsv = csv.reader(datafile)
    header = next(datacsv)
    patient_names = header[1:]
    patient_data = [[] for _ in range(len(patient_names))]
    npatients = len(patient_data)

    for line in datacsv:
        gene_names.append(line[0])
        assert npatients + 1 == len(line)

        for i in range(npatients):
            patient_data[i].append(float(line[i + 1]))
patient_data = np.array(patient_data)

clustering = clustergenes.GeneCluster(patient_data=patient_data, use_tqdm=True)

for i, interval in enumerate(clustering.intervals):
    n = interval["nclusters"]
    p0, p1 = interval["p_interval"]
    plen = interval["p_length"]
    clusters = clustering.get_clusters(n)

    print("--- Begin ---")
    print(f"n={n:4}, plen={plen:6.2f}, interval=[{p0:6.2f}, {p0:6.2f}]\n")
    for j, cluster in enumerate(clusters):
        print(f"Group {j + 1:4}, size={len(cluster):4}")
        names = [patient_names[k] for k in cluster]
        print(" ".join(names))
        print()
    print("End")
    print()

    if i < ntopresults:
        for j, cluster in enumerate(clusters):
            filename = os.path.join(outfolder, f"genes-{i + 1}-{j + 1}.eps")
            SaveCluster(filename, patient_data, cluster)
