import numpy as np
import matplotlib.pyplot as plt

# Visualize the learned weights for each class
w = best_softmax.W[:-1,:] # CHANGE TO FC layer pickle
w = w.reshape(32, 32, 3, 10)

w_min, w_max = np.min(w), np.max(w)

classes = ['low','med','high']
for i in range(len(classes)):
    plt.subplot(1, 3, i + 1)
    
    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])

plt.show()
