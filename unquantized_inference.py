import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/home/ioannavav/Desktop/workspace/lanenet-lane-detection')
#import lanenet_model
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
CFG = parse_config_utils.lanenet_cfg

def minmax_scale(input_arr):
    """
    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr



def TfLiteInference(tflite_model_path, input_image_path):

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output1_details = interpreter.get_output_details()[0] # Binary Image Output
    output2_details = interpreter.get_output_details()[1] # Instance Image Output

    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    image_vis = image
    #print("**************")
    #print(len(image))
    #print(len(image[0]))
    #print("**************")
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    print("**************")
    print(len(image))
    print(len(image[0]))
    print("**************")

    test = np.expand_dims(image, axis=0)
    test = test.astype('float32')

    # If required, quantize the input layer (from float to integer)
    input_scale, input_zero_point = input_details["quantization"]
    if (input_scale, input_zero_point) != (0.0, 0):
        test = test / input_scale + input_zero_point
        test = test.astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], test)
    interpreter.invoke()
    binary_seg_ret = interpreter.get_tensor(output1_details["index"])[0]
    instance_seg_ret = interpreter.get_tensor(output2_details["index"])[0]

    # If required, dequantized the output layer (from integer to float)
    output_scale, output_zero_point = output1_details["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        binary_seg_ret = binary_seg_ret.astype(np.float32)
        binary_seg_ret = (binary_seg_ret - output_zero_point) * output_scale
    output_scale, output_zero_point = output2_details["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        instance_seg_ret = instance_seg_ret.astype(np.float32)
        instance_seg_ret = (instance_seg_ret - output_zero_point) * output_scale

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)
    postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_ret,
            instance_seg_result=instance_seg_ret,
            source_image=image_vis
        )
    mask_image = postprocess_result['mask_image']

    for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
        instance_seg_ret[:, :, i] = minmax_scale(instance_seg_ret[:, :, i])
    embedding_image = np.array(instance_seg_ret, np.uint8)

    #plt.figure('mask_image')
    #plt.imshow(mask_image[:, :, (2, 1, 0)])
    #plt.figure('src_image')
    #plt.imshow(image_vis[:, :, (2, 1, 0)])
    #plt.figure('instance_image')
    #plt.imshow(embedding_image[:, :, (2, 1, 0)])
    #plt.figure('binary_image')
    #plt.imshow(binary_seg_ret * 255, cmap='gray')
    #plt.show()
    print("------------------")
    print(len(binary_seg_ret))
    print(len(binary_seg_ret[0]))
    print("------------------")

    path_parts = input_image_path.split("/")
    #print(path_parts[-1])
    #print(path_parts[-2])
    dirname = './unquantized_results/0601/{}'.format(path_parts[-2])
    isExist = os.path.exists(dirname)
    if not isExist:
        os.mkdir(dirname)
    cv2.imwrite('./unquantized_results/0601/{}/{}'.format(path_parts[-2], path_parts[-1]), binary_seg_ret * 255)


tflite_model_path = '/home/ioannavav/Desktop/workspace/unquantized_model.tflite'
list_subfolders_with_paths = [f.name for f in os.scandir('/home/ioannavav/Desktop/workspace/train_set/clips/0601') if f.is_dir()]
#print(list_subfolders_with_paths)
for j in list_subfolders_with_paths:
    for i in range(1,21):
        input_image_path = '/home/ioannavav/Desktop/workspace/train_set/clips/0601/{}/{}.jpg'.format(j,i)
        TfLiteInference(tflite_model_path, input_image_path)
