import subprocess

import sys

import os

import pathlib





def main():

    if len(sys.argv) != 3:

        print("Usage", sys.argv[0], "input_dir", "output_dir")

        exit(1)


    input_dir = sys.argv[1]

    output_dir = sys.argv[2]



    dir_list = []

    for root, dirs, files in os.walk(input_dir):

        if len(dirs) > 0:

            for dir_name in dirs:

                jpg = os.path.join(root, dir_name, "1.jpg")

                if os.path.exists(jpg):

                    dir_list.append(os.path.join(root, dir_name))


    for input_dirname in dir_list:

        check_output_dir(input_dirname, output_dir)

        output_video(input_dirname, output_dir)



def check_output_dir(input_dir, output_dir):

    paths = os.path.normpath(input_dir)

    paths = input_dir.split(os.sep)[1:]

    output_path = os.path.join(output_dir, *paths)

    if not os.path.exists(output_path):

        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)



def output_video(input_dirname, output_dir):

    output_dirname = os.path.normpath(input_dirname)

    base_dirnames = output_dirname.split(os.sep)[1:]

    output_filename = os.path.join(output_dir, *base_dirnames, "vid.mp4")

    output_filename = os.path.abspath(output_filename)

    command = "ffmpeg -f image2 -r 5 -i %d.jpg -vcodec libx264 -crf 18  -pix_fmt yuv420p {0} -y".format(output_filename)

    subprocess.check_call(command.split(), cwd=input_dirname)

if __name__ == "__main__":

    main()
