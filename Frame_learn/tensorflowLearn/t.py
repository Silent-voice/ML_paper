import sys
import os
import tensorflow as tf
import gc


def predict(image_path):
    # print(image_path)
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("./labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("./output.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    f.close()
    aResult = ''

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        highestScore = -1
        label = ''

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score > highestScore:
                highestScore = score
                label = human_string
                # print('%s (score = %.5f)' % (human_string, score))

        aResult = label

    sess.close()

    del image_data
    del label_lines
    del graph_def
    del _
    del softmax_tensor
    del predictions
    del top_k
    gc.collect()

    return aResult


def main():
    testSetPath = '/home/ivan1rufus/ML/tfClassifier/catvsdog/test/'
    result = open('/home/ivan1rufus/ML/tfClassifier/image_classification/result.txt', 'w')
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for each in range(1, 2001):
            # change this as you see fit
            # image_path = sys.argv[1]
            image_path = testSetPath + str(each) + '.jpg'
            aResult = str(each) + '\t' + predict(image_path) + '\n'
            print(aResult)
            result.write(aResult)

    result.close()


if __name__ == '__main__':
    main()