import json

import sys

import os

import glob

import pprint



def main():

    if len(sys.argv) != 2:

        printf("Usage", sys.argv[0], "result_dir")

        exit(1)


    result_dir = sys.argv[1]


    path = os.path.join(result_dir,  "**", "*.json")

    json_files = glob.glob(path, recursive=True)


    file_diffs = []


    for filename in json_files:

        with open(filename) as f:

            diffs = json.load(f)

        diffs = [float(diff) for diff in diffs]

        if len(diffs) == 0:

            continue

        #max_diff = max(diffs)
        max_diff = min(diffs)
        file_diffs.append((filename, max_diff))

    file_diffs.sort(key=lambda x: x[1])

    # top 5 values

    #file_diffs.reverse()

    pprint.pprint(file_diffs[:15])


if __name__ == "__main__":

    main()
