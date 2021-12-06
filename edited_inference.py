import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from lanenet_model import lanenet_postprocess
#from local_utils.config_utils import parse_config_utils



def TfLiteInference(tflite_model_path, input_image_path):

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output1_details = interpreter.get_output_details()[0] # Binary Image Output
    #output2_details = interpreter.get_output_details()[1] # Instance Image Output

    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    plt.figure('input image')
    plt.imshow(image)
    plt.show()
    cv2.imwrite('/home/ioannavav/Desktop/workspace/used_input_image.jpg',image)

    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    test = np.expand_dims(image, axis=0)
    test = test.astype('float32')

    # If required, quantize the input layer (from float to integer)
    input_scale, input_zero_point = input_details["quantization"]
    if (input_scale, input_zero_point) != (0.0, 0):
        test = test / input_scale + input_zero_point
        test = test.astype(input_details["dtype"])
    #print("OK")
    
    interpreter.set_tensor(input_details["index"], test)
    interpreter.invoke()
    print("OK")
    
    binary_seg_ret = interpreter.get_tensor(output1_details["index"])[0]
    #instance_seg_ret = interpreter.get_tensor(output2_details["index"])[0]
    print("Until here all ok")

    
    # If required, dequantized the output layer (from integer to float)
    output_scale, output_zero_point = output1_details["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        binary_seg_ret = binary_seg_ret.astype(np.float32)
        binary_seg_ret = (binary_seg_ret - output_zero_point) * output_scale
    #output_scale, output_zero_point = output2_details["quantization"]
    #if (output_scale, output_zero_point) != (0.0, 0):
    #    instance_seg_ret = instance_seg_ret.astype(np.float32)
    #    instance_seg_ret = (instance_seg_ret - output_zero_point) * output_scale

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
    plt.figure('binary_image')
    new_binary = binary_seg_ret * 255
    plt.imshow(new_binary, cmap='gray')
    plt.show()
    cv2.imwrite('/workspace/binary1.jpg',new_binary)
    print("lalalalala")
    

tflite_model_path = '/home/ioannavav/Desktop/workspace/saved_model.tflite'
input_image_path = '/home/ioannavav/Desktop/workspace/train_set/clips/0601/1494452385593783358/1.jpg'
TfLiteInference(tflite_model_path, input_image_path)
    
print("--------------------------")
    #input_image_path = '/content/gdrive/MyDrive/train_set/clips/0601/1494452385593783358/2.jpg'
    #TfLiteInference(tflite_model_path, input_image_path)
