from tensorflow.examples.tutorials.mnist import input_data
import CapsuleNetwork as cn

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_mnist_train = int(mnist.train.num_examples)
n_mnist_test = int(mnist.test.num_examples)

n_batch_samples = 100
n_epochs = 10
n_batches = n_mnist_train // n_batch_samples

capsule_network = cn.CapsuleNetwork(n_inputs=n_batch_samples)

for epoch in range(n_epochs):
    cost_batch = []
    for batch in range(n_batches):
        X_batch, Y_batch = mnist.train.next_batch(n_batch_samples)
        cost_batch.append(capsule_network.fit(X_batch, Y_batch))
        print('batch {} ({} of {} training images) cost {:1.3f}'.format(
            batch, n_batch_samples * (batch+1), n_mnist_train, cost_batch[-1]))

    cost_epoch = sum(cost_batch) / n_batches
    print('epoch {} avg cost {:1.3f}'.format(epoch, cost_epoch))
