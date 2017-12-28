import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.

training_file= 'train.p'
with open(training_file, mode='rb') as f:
	data=pickle.load(f)

nb_classes=43
EPOCHS=10
BATCH_SIZE=128



# TODO: Split data into training and validation sets.

X_train, X_valid, y_train, y_valid= train_test_split(data['fetaures'], data['labels'], test_size=0.30, ramdom_state=0)

# TODO: Define placeholders and resize operation.

features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
one_hot_y = tf.one_hot(labels, 43)
resized= tf.image.resize(features, (227), (227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape=(fc7.get_shape().as_list()[-1],nb_classes)
fc8W  = tf.Variable(tf.truncated_normal(shape, stddev=1e-02)
fc8_b = tf.Variable(tf.zeros(43))
logits= tf.matmul(fc7,fc8W) + fc8_b

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
loss_operation=tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation=optimizer.minimize(loss_operation,var_list=[fc8W, fc8_b])
init_op = tf.initialize_all_variables()

predictions = tf.arg_max(logits, 1)
accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))


# TODO: Train and evaluate the feature extraction model.

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss=0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        #accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:[1,1,1]})
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: [1,1,1]})
        #total_acc += (acc * len(batch_x))
        total_loss += (loss * len(batch_x))
        total_accuracy += (accuracy * len(batch_x))
        #total_acc += (acc * batch_x.shape[0])
        #total_loss += (loss * batch_x.shape[0])
    return total_loss/num_examples, total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(init_op)

    for i in range(EPOCHS):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
	for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y)


	validation_loss,validation_accuracy = evaluate(X_valid, y_valid)
        validation_loss_history.append(validation_loss)
        validation_accuracy_history.append(validation_accuracy)
        
        train_loss, train_accuracy = evaluate(X_train, y_train)
        training_loss_history.append(train_loss)
        training_accuracy_history.append(train_accuracy)

        
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy*100),"%")
        print("Validation Loss = {:.3f}".format(validation_loss))
        print()

    save_file='./Modeltrained20'
    saver.save(sess, save_file)
    print("Model saved")
	








