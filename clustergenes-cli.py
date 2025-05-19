import clustergenes
import csv
import sys
import numpy as np

if len(sys.argv) != 2:
    print("Usage python3 clustergenes-cli.py [genes csv file]")

datafilename = sys.argv[1]

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

for interval in clustering.intervals:
    n = interval["nclusters"]
    p0, p1 = interval["p_interval"]
    plen = interval["p_length"]
    clusters = clustering.get_clusters(n)

    print("--- Begin ---")
    print(f"n={n:4}, plen={plen:6.2f}, interval=[{p0:6.2f}, {p0:6.2f}]\n")
    for i, cluster in enumerate(clusters):
        print(f"Group {i + 1:4}, size={len(cluster):4}")
        names = [patient_names[i] for i in cluster]
        print(" ".join(names))
        print()
    print("End")
    print()
