#references:
#https://github.com/NVlabs/ocroseg
#https://github.com/philipperemy/tensorflow-multi-dimensional-lstm
#https://github.com/tmbdev/ocropy
#paper:
#Multi-Dimensional Recurrent Neural Networks
#Robust_ Simple Page Segmentation Using Hybrid Convolutional MDLSTM Networks
#dataset:
#https://storage.googleapis.com/tmb-ocr/uw3-framed-lines-degraded-000.tgz

import os
import math
import tensorflow as tf
from PIL import Image
from functools import partial

from md_lstm import multi_dimensional_rnn_while_loop,horizontal_standard_lstm,snake_standard_lstm,horizontal_vertical_lstm_inorder,horizontal_vertical_lstm_together

import cv2
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
batch_size=1
batch_height=None
batch_width=None
batch_channel=1
save_steps=1000


def get_tf_dataset(dataset_text_file,batch_size=1, channels=1,shuffle_size=10):
    def _parse_function(filename, labelname):
        
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=channels)
        #image = tf.image.resize_images(image_decoded, size)
        #image=255-image
        image = tf.cast(image_decoded, tf.float32) * (1. / 255)
        
        label_string = tf.read_file(labelname)
        label_decoded = tf.image.decode_jpeg(label_string, channels=channels)
        label = tf.image.resize_images(label_decoded, [tf.round(tf.div(tf.shape(label_decoded)[0],4)),tf.round(tf.div(tf.shape(label_decoded)[1],4))])
        label = tf.cast(label, tf.float32) * (1. / 255)
        label =tf.cast(label>0.5, tf.float32)

        
        return image, label

    def read_labeled_image_list(dataset_text_file):
        base_dir="./uw3/"
        filenames=[]
        labels=[]
        with open(dataset_text_file,"r",encoding="utf-8") as f_l:
            filenames_lables=f_l.readlines()

        one_epoch_num=len(filenames_lables)

        for filename_lable in filenames_lables:
            filenames.append(base_dir+filename_lable.split(" ")[0])
            labels.append(base_dir+filename_lable.split(" ")[1].strip("\n"))
        return filenames,labels,one_epoch_num

    filenames, labels,one_epoch_num = read_labeled_image_list(dataset_text_file)

    filenames = tf.constant(filenames, name='filename_list')
    labels = tf.constant(labels, name='label_list')

    #tensorflow1.3:tf.contrib.data.Dataset.from_tensor_slices
    #tensorflow1.4+:tf.data.Dataset.from_tensor_slices
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat()

    return dataset,one_epoch_num


def network(is_training=False):
    network = {}
    network["inputs"] = tf.placeholder(tf.float32, [batch_size, batch_height,batch_width, batch_channel],
                                       name='inputs')
    network["conv1"] = tf.layers.conv2d(inputs=network["inputs"], filters=32, kernel_size=(3, 3), padding="same",
                                        activation=None, name="conv1")
    #with tf.variable_scope("BN"):
    network["batch_norm1"] = tf.contrib.layers.batch_norm(
            network["conv1"],
            decay=0.9,
            center=True,
            scale=True,
            epsilon=0.001,
            updates_collections=None,
            is_training=is_training,
            zero_debias_moving_mean=True,
            scope="BN1")
    network["batch_norm1"] = tf.nn.relu(network["batch_norm1"])
    network["pool1"] = tf.layers.max_pooling2d(inputs=network["batch_norm1"], pool_size=[2, 2], strides=2)
    network["conv2"] = tf.layers.conv2d(inputs=network["pool1"], filters=64, kernel_size=(3, 3), padding="same",
                                        activation=None, name="conv2")
    #with tf.variable_scope("BN"):
    network["batch_norm2"] = tf.contrib.layers.batch_norm(
            network["conv2"],
            decay=0.9,
            center=True,
            scale=True,
            epsilon=0.001,
            updates_collections=None,
            is_training=is_training,
            scope="BN2")
    network["batch_norm2"] = tf.nn.relu(network["batch_norm2"])
    network["pool2"] = tf.layers.max_pooling2d(inputs=network["batch_norm2"], pool_size=[2, 2], strides=2)
    network["conv3"] = tf.layers.conv2d(inputs=network["pool2"], filters=128, kernel_size=(3, 3), padding="same",
                                        activation=None, name="conv3")
    #with tf.variable_scope("BN"):
    network["batch_norm3"] = tf.contrib.layers.batch_norm(
            network["conv3"],
            decay=0.9,
            center=True,
            scale=True,
            epsilon=0.001,
            updates_collections=None,
            is_training=is_training,
            scope="BN3")
    network["batch_norm3"] = tf.nn.relu(network["batch_norm3"])

    network["LSTM2D1"] = horizontal_vertical_lstm_inorder(rnn_size=128, input_data=network["batch_norm3"], scope_n="LSTM2D1")
    #network["LSTM2D1"] = horizontal_vertical_lstm_together(rnn_size=128, input_data=network["batch_norm3"], scope_n="LSTM2D1")
    #network["LSTM2D1"] = horizontal_standard_lstm(rnn_size=128, input_data=network["batch_norm3"], scope_n="LSTM2D1")
    #network["LSTM2D1"] = snake_standard_lstm(rnn_size=128, input_data=network["batch_norm3"], scope_n="LSTM2D1")
    #network["LSTM2D1"], _ = multi_dimensional_rnn_while_loop(rnn_size=128, input_data=network["batch_norm3"],sh=[1, 1], dims=None, scope_n="LSTM2D1")

    network["conv4"] = tf.layers.conv2d(inputs=network["LSTM2D1"], filters=64, kernel_size=(3, 3), padding="same",
                                        activation=None, name="conv4")
    #with tf.variable_scope("BN"):
    network["batch_norm4"] = tf.contrib.layers.batch_norm(
            network["conv4"],
            decay=0.9,
            center=True,
            scale=True,
            epsilon=0.001,
            updates_collections=None,
            is_training=is_training,
            scope="BN4")
    network["batch_norm4"] = tf.nn.relu(network["batch_norm4"])
    network["LSTM2D2"] = horizontal_vertical_lstm_inorder(rnn_size=128, input_data=network["batch_norm4"], scope_n="LSTM2D2")
    #network["LSTM2D2"] = horizontal_vertical_lstm_together(rnn_size=128, input_data=network["batch_norm4"], scope_n="LSTM2D2")
    #network["LSTM2D2"] = horizontal_standard_lstm(rnn_size=128, input_data=network["batch_norm4"], scope_n="LSTM2D2")
    #network["LSTM2D2"] = snake_standard_lstm(rnn_size=128, input_data=network["batch_norm4"], scope_n="LSTM2D2")
    #network["LSTM2D2"], _ = multi_dimensional_rnn_while_loop(rnn_size=128, input_data=network["batch_norm4"],sh=[1, 1], dims=None, scope_n="LSTM2D2")

    network["conv5"] = tf.layers.conv2d(inputs=network["LSTM2D2"], filters=1, kernel_size=(3, 3), padding="same",
                                        activation=None, name="conv5")
    network["outputs"] = tf.nn.sigmoid(network["conv5"])
    return network






def train():
    #network
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.001,
                                               global_step=global_step,
                                               decay_steps=10000,
                                               decay_rate=0.1,
                                               staircase=True)
 
    model=network(is_training=True)
    y_ = tf.placeholder(tf.float32, [batch_size, batch_height, batch_width, batch_channel], name='labels')
    loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_,predictions=model["outputs"]))
    accuracy=tf.reduce_sum(tf.cast((tf.cast(model["outputs"]>0.5,tf.int32)+tf.cast(y_,tf.int32))>1,tf.float32))/tf.reduce_sum(y_)


    update_ops= tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        with tf.control_dependencies(update_ops):
            grad_update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    else:
        grad_update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    
    
    #tensorboard
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    for update_op in update_ops:
        tf.summary.histogram(update_op.name, update_op)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    merge_summary = tf.summary.merge_all()
    
    dataset,one_epoch_num = get_tf_dataset(dataset_text_file="./uw3/label.txt",batch_size=batch_size,channels=batch_channel)
    iterator = dataset.make_one_shot_iterator()
    img_batch, label_batch = iterator.get_next()

    init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        #saver.restore(save_path="./save/ocrseg.ckpt-2000", sess=session)

        #tensorboard
        summary_writer = tf.summary.FileWriter("./summary/", session.graph)

        epoch=0
        while True:
            try:
                img_batch_i, label_batch_i = session.run([img_batch,label_batch])
                
                if img_batch_i.shape[0]!=batch_size:
                    print("the last iter of one epoch")
                    continue
            except tf.errors.OutOfRangeError:
                print("one epoch over!")
                continue
            feed = {model["inputs"]: img_batch_i,y_: label_batch_i}
            learning_rate_train,loss_train,accuracy_train,step,summary,_=session.run([learning_rate,loss,accuracy,global_step,merge_summary,grad_update], feed_dict=feed)
            print("learning rate:%f epoch:%d iter:%d loss:%f accuracy:%f"%(learning_rate_train,epoch,step,loss_train,accuracy_train))

            #tensorboard
            summary_writer.add_summary(summary, step)

            if step > 0 and step % save_steps == 0:
                save_path = saver.save(session, "save/ocrseg.ckpt", global_step=step)
                print(save_path)
            if step > 0:
                epoch=step*batch_size//one_epoch_num

def test():
    model=network(is_training=False)
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(save_path="./save/ocrseg.ckpt-1000",sess=session)

        image=cv2.imread("./make_training_labels/W001.png",0)
        height,width=image.shape
        image=image.reshape((1,image.shape[0],image.shape[1],1))
        image=1-image/255
        feed = {model["inputs"]: image}
        output = session.run(model["outputs"], feed_dict=feed)
        output = output.reshape((output.shape[1],output.shape[2]))
        print("max:%f min:%f"%(np.max(np.max(output)),np.min(np.min(output))))
        output=(output>0.5)*255
        output=np.asarray(output,np.uint8)
        output=cv2.resize(output,(width,height))
        cv2.imwrite("out.png",output)

if __name__=="__main__":
    train()
    #test()
