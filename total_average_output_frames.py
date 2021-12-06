import json

import sys

import os

import glob

import pprint

from statistics import mean


def main():

    if len(sys.argv) != 2:

        printf("Usage", sys.argv[0], "result_dir")

        exit(1)


    result_dir = sys.argv[1]


    path = os.path.join(result_dir,  "**", "*.json")

    json_files = glob.glob(path, recursive=True)


    file_diffs = []
    just_means = []

    for filename in json_files:

        with open(filename) as f:

            diffs = json.load(f)

        diffs = [float(diff) for diff in diffs]

        if len(diffs) == 0:

            continue

        mean_diff = mean(diffs)

        file_diffs.append((filename, mean_diff))
        just_means.append(mean_diff)

    file_diffs.sort(key=lambda x: x[1])
    #total average
    total_average = mean(just_means)
    print("Total average is:", total_average)

    # top 5 values
    
    file_diffs.reverse()

    pprint.pprint(file_diffs[:5])


if __name__ == "__main__":

    main()
