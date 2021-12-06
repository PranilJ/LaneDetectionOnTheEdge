

import sys

import cv2

import numpy as np

import os

import glob

import pathlib

import json

def wrap_image(image, flow):

    result = np.zeros(image.shape)

    for y in range(image.shape[0]):

        for x in range(image.shape[1]):

            dy, dx = flow[y][x]

            xx = int(x + dx)

            yy = int(y + dy)

            if xx < image.shape[1] and yy < image.shape[0] and xx >= 0 and yy >= 0:

                result[yy][xx] = image[y][x]

    return result

def _main():

    if len(sys.argv) != 4:

        print("Usage", sys.argv[0], "image_dir", "result_dir", "output_dir")

        exit(1)

    image_dir = sys.argv[1]

    result_dir = sys.argv[2]

    output_dir = sys.argv[3]

    # walkl through the directories

    dir_list = []

    for root, dirs, files in os.walk(image_dir):

        if len(dirs) > 0:

            for dir_name in dirs:

                dir_list.append(os.path.join(root, dir_name))

    # filter out each dir to make sure they have 20 images

    finished_dir_list = []

    for dirname in dir_list:

        image_files = glob.glob(os.path.join(dirname, "*.jpg"))

        # find corresponding folder in the result folder

        p = os.path.normpath(dirname)

        p = p.split(os.sep)[1:]

        p = os.path.join(result_dir, *p)

        result_files = glob.glob(os.path.join(p, "*.jpg"))

        if len(image_files) == len(result_files):

            finished_dir_list.append(dirname)

    for input_dirname in finished_dir_list:

        check_output_dir(input_dirname, output_dir)

        output_difference(input_dirname, result_dir, output_dir)

def check_output_dir(input_dir, output_dir):

    paths = os.path.normpath(input_dir)

    paths = input_dir.split(os.sep)[1:]

    output_path = os.path.join(output_dir, *paths)

    if not os.path.exists(output_path):

        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

def output_difference(image_dirname,  result_dir, output_dir):

    output_dirname = os.path.normpath(image_dirname)

    base_dirnames = output_dirname.split(os.sep)[1:]

    result_dirname = os.path.join(result_dir, *base_dirnames)

    output_dirname = os.path.join(output_dir, *base_dirnames)

    files = glob.glob(os.path.join(image_dirname, "*.jpg"))

    if len(files) == 0:

        return

    # use numeric naming scheme

    diffs = []

    for i in range(1, len(files)):

        input_image1 = os.path.join(image_dirname, str(i) + ".jpg")

        assert os.path.exists(input_image1)

        input_image2 = os.path.join(image_dirname, str(i + 1) + ".jpg")

        assert os.path.exists(input_image2)

        pred_image1 = os.path.join(result_dirname, str(i) + ".jpg")

        #print(pred_image1)

        assert os.path.exists(pred_image1)

        pred_image2 = os.path.join(result_dirname, str(i + 1) + ".jpg")

        assert os.path.exists(pred_image2)

        input_image1 = cv2.imread(input_image1)

        input_image2 = cv2.imread(input_image2)

        pred_image1 = cv2.imread(pred_image1)

        pred_image2 = cv2.imread(pred_image2)

        diff = compute_difference(input_image1, input_image2, pred_image1, pred_image2)

        diffs.append(str(diff))

    # dump the result as a JSON file

    result_filename = os.path.join(output_dirname, "diff.json")

    with open(result_filename, "w+") as f:

        print("Saving diff result to", result_filename)

        json.dump(diffs, f)

def compute_difference(image1, image2, pred1, pred2):

    image1 = cv2.resize(image1, (512, 256), interpolation=cv2.INTER_LINEAR)
    image2 = cv2.resize(image2, (512, 256), interpolation=cv2.INTER_LINEAR)

    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    pred1_gray = cv2.cvtColor(pred1, cv2.COLOR_BGR2GRAY)
    pred2_gray = cv2.cvtColor(pred2, cv2.COLOR_BGR2GRAY)

    image_flow = cv2.calcOpticalFlowFarneback(image1_gray, image2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    pred_image_flow = wrap_image(pred1_gray, image_flow)
    pred_flow = cv2.calcOpticalFlowFarneback(pred_image_flow, pred2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mean = np.mean(np.abs(pred_flow))
    print("Mean distance", mean)
    normalized_flow = mean



    #image_flow = cv2.calcOpticalFlowFarneback(image1_gray, image2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    #pred_image_flow = wrap_image(pred1_gray, image_flow)

    #pred_flow = cv2.calcOpticalFlowFarneback(pred_image_flow, pred2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    #mean = np.mean(np.abs(pred_flow))

    #print("Mean distance", mean)

    #input_flow = cv2.calcOpticalFlowFarneback(image1_gray, image2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    #input_difference = np.mean(abs(input_flow))

    #output_flow = cv2.calcOpticalFlowFarneback(pred1_gray, pred2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    #output_difference = np.mean(abs(output_flow))

    #print("Input difference:", input_difference)

    #print("Output difference:", output_difference)

    #normalized_flow = output_difference / input_difference

    #print("Normalized output difference:", normalized_flow)

    #print(len(input_flow))

    #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    #hsv[...,0] = ang*180/np.pi/2

    #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

    #bgr = cv.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    #mag, ang = cv2.cartToPolar(pred1_gray, pred2_gray)

    #hsv1 = ang*180/np.pi/2

    #hsv2 = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

    #bgr = cv.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    # outputting predicated and actual flow

    #cv2.imwrite("pred_flow.jpg", pred_image_flow)

    #cv2.imwrite("gold_flow.jpg", pred2_gray)

    #cv2.imwrite("input_flow.jpg", input_difference)

    #cv2.imwrite("output_flow.jpg", output_difference)

    return normalized_flow

if __name__ == "__main__":

    _main()


