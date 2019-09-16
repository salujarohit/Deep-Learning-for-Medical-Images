import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imread
from skimage.transform import resize
import cv2

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras import backend as K

except:
    import tensorflow as tf
    from tensorflow.python.keras.models import Sequential, load_model
    from tensorflow.python.keras import backend as K

tf.compat.v1.disable_eager_execution()


def show_visualization(model, hyperparameters):
    # model = load_model()
    print(model.summary)
    Sample = '/Lab1/Lab2/Bone/train/AFF/14.jpg'
    Img = imread(Sample)
    print(Img.shape)
#     Img = Img[:,:,0]
    Img = Img/255
    img_height, img_width = hyperparameters['input_shape'][0], hyperparameters['input_shape'][1]
    Img = resize(Img, (img_height, img_width), anti_aliasing = True).astype('float32')
#     Img = np.expand_dims(Img, axis = 2)
    print(Img.shape)
    Img = np.expand_dims(Img, axis = 0)
    print(Img.shape)
    preds = model.predict(Img)
    class_idx = np.argmax(preds[0])
    print(class_idx)
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer("conv2d_12")
    print(last_conv_layer.output)
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([Img])
    for i in range(hyperparameters['base']*8):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # For visualization
    img = cv2.imread(Sample)
    img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_AREA)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    fig = plt.figure()
    plt.imshow(img)
    result_path = os.path.join(os.path.join(os.getcwd(), 'results'), "Task 10_img.png")
    fig.savefig(result_path, dpi=fig.dpi)
    fig = plt.figure()
    plt.imshow(superimposed_img)
    result_path = os.path.join(os.path.join(os.getcwd(), 'results'), "Task 10_superimposed_img.png")
    fig.savefig(result_path, dpi=fig.dpi)

