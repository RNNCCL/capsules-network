from tensorflow.examples.tutorials.mnist import input_data
import CapsuleNetwork as cn
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_mnist_train = int(mnist.train.num_examples)
n_mnist_test = int(mnist.test.num_examples)

n_batch_samples = 50
n_epochs = 10

# only train on digits '0' and '1'
train_labels = mnist.train.labels
zero_one_indices = []
for i, label in enumerate(train_labels):
    if label[0] == 1 or label[1] == 1:
        zero_one_indices.append(i)

n_train_zero_one = 100

capsule_network = cn.CapsuleNetwork(n_inputs=n_batch_samples, reconstruct=True)

for epoch in range(n_epochs):
    cost_batch = []
    n_batches = n_train_zero_one // n_batch_samples

    shuffle(zero_one_indices)
    train_zero_one_images = mnist.train.images[zero_one_indices[0:n_train_zero_one]]
    train_zero_one_labels = mnist.train.labels[zero_one_indices[0:n_train_zero_one]]

    batch_index = 0
    for batch in range(n_batches):
        X_batch = train_zero_one_images[batch_index:batch_index + n_batch_samples]
        Y_batch = train_zero_one_labels[batch_index:batch_index + n_batch_samples]
        batch_index += n_batch_samples

        cost_batch.append(capsule_network.fit(X_batch, Y_batch))
        print('batch {} ({} of {} training images) cost {:1.3f}'.format(
            batch, n_batch_samples * (batch+1), n_train_zero_one, cost_batch[-1]))

    cost_epoch = sum(cost_batch) / n_batches
    print('epoch {} avg cost {:1.3f}'.format(epoch, cost_epoch))

    # reconstruction
    reconstructed_samples = capsule_network.reconstruction(X_batch, Y_batch)
    print (reconstructed_samples.shape)
    recon_sample = reconstructed_samples[0]
    print (recon_sample.shape)
    recon_sample = np.reshape(recon_sample, [28, 28])
    recon_sample_uint8 = 255. * (recon_sample - np.min(recon_sample)) / (np.max(recon_sample) - np.min(recon_sample))
    plt.imshow(recon_sample_uint8.astype('uint8'), cmap='gray')
    plt.gca().axis('off')
    plt.show()
