import sys
import cv2
import numpy as np


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
    if len(sys.argv) != 5:
        print("Usage", sys.argv[0], "image1", "image2", "pred1", "pred2")
        exit(1)
    image1 = cv2.imread(sys.argv[1])
    image2 = cv2.imread(sys.argv[2])
    pred1 = cv2.imread(sys.argv[3])
    pred2 = cv2.imread(sys.argv[4])

    #we first normalize the input images so that they have the same size (256,512) as the output images
    image1 = cv2.resize(image1, (512, 256), interpolation=cv2.INTER_LINEAR)
    image2 = cv2.resize(image2, (512, 256), interpolation=cv2.INTER_LINEAR)

    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    pred1_gray = cv2.cvtColor(pred1, cv2.COLOR_BGR2GRAY)
    pred2_gray = cv2.cvtColor(pred2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("my_image1.jpg", image1_gray)
    cv2.imwrite("my_image2.jpg", image2_gray)
    cv2.imwrite("my_pred1.jpg", pred1_gray)
    cv2.imwrite("my_pred2.jpg", pred2_gray)
    image_flow = cv2.calcOpticalFlowFarneback(image1_gray, image2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print(len(image_flow))
    print(len(image_flow[0]))
    print(len(image_flow[0][0]))
    print("Image flow is:", image_flow)
    pred_image_flow = wrap_image(pred1_gray, image_flow)
    wrap_input = wrap_image(image1, image_flow)
    #print(len(wrap_input))
    #print(len(wrap_input[0]))
    cv2.imwrite("WRAAAAP.jpg", wrap_input)
    cv2.imwrite("my_image_wrap.jpg", pred_image_flow)
    pred_flow = cv2.calcOpticalFlowFarneback(pred_image_flow, pred2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #print("Pred flow is:", pred_flow)
    mean = np.mean(np.abs(pred_flow))
    print("Mean distance", mean)
    normalized_flow = mean
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


if __name__ == "__main__":
    _main()
