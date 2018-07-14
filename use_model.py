#!/usr/bin/env python

import tensorflow as tf
import random
import numpy
from numpy.random import RandomState

def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

#load graph
file_name = 'results/my_model.pb'
with tf.gfile.FastGFile(file_name,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

#run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    x = sess.graph.get_tensor_by_name("lstm_1_input_1:0")
    #print x
    #input_val = random_int_list(1000, 2000, 47)
    #input_val = [[input_val]]
    #input_val = numpy.transpose(input_val)

    input_val = numpy.arange(47).reshape(1, 47, 1)
    for i in range(47):
        input_val[0, i, 0] = random.randint(2000, 3000)
    #print input_val

    feed_dict ={x:input_val}
    y = sess.graph.get_tensor_by_name("activation_1_1/Identity:0")
        #print y
    summary_writer = tf.summary.FileWriter('log', sess.graph)
    logits = sess.run(y, feed_dict)
    print y
    print(y.eval())

