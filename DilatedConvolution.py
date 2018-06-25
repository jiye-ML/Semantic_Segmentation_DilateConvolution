import tensorflow as tf
import os


from DataConfig import Config
from Tools import Tools
import pickle


class DilatedConvolution(object):

    def __init__(self, dataset, data, learning_rate, trainable=True):

        # data
        self.dataset = dataset
        self.data = data
        self.class_number = self.data.class_number
        self.image_height = self.data.image_height
        self.image_width = self.data.image_width
        self.image_channel = self.data.image_channel
        self.label_channel = self.data.label_channel
        self.label_channel = self.data.label_channel
        self.batch_size = self.data.batch_size

        # 创建checkpoint目录
        self.checkpoint = Tools.new_dir(os.path.join("checkpoint", 'dilated_' + self.dataset))
        self.checkpoint_file = os.path.join(self.checkpoint, "dilated")
        self.checkpoint_file_meta = self.checkpoint_file + ".meta"
        # model
        self.learning_rate = learning_rate
        self.w_pretrained = self.need_load_pretrained()
        self.trainable = trainable

        self.graph = tf.Graph()

        self.images, self.labels = None, None
        self.x, self.y, self.logits, self.loss, self.train_op = None, None, None, None, None
        self.prediction, self.is_correct, self.accuracy = None, None, None
        # 如果还没有图，绘制
        if self.w_pretrained is not None:
            with self.graph.as_default():
                # input
                self.x = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_channel],
                                        name="x")
                self.y = tf.placeholder(tf.uint8, [None, self.image_height, self.image_width, self.label_channel],
                                        name="y")
                # output
                self.logits = self.dilated_convolution_by_pretrained()
                self.prediction = tf.reshape(tf.cast(tf.argmax(self.logits, axis=3), tf.uint8),
                                                     shape=[-1, self.image_height, self.image_width,self.label_channel],
                                                     name="prediction")  # 预测 输出
                self.is_correct = tf.cast(tf.equal(self.prediction, tf.cast(self.y, tf.uint8)), tf.uint8, name="is_correct")  # 是否正确
                self.accuracy = tf.reduce_mean(tf.cast(self.is_correct, tf.float32), name='accuracy')  # 正确率

                # loss
                self.loss = self.loss_cross_entropy()
                # train
                self.train_op = self.get_train_op()

                # load batch
                self.images, self.labels = self.data.get_train_data()
                pass
        else : # 如果有图，加载节点
            with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
                # 从checkpoint中导入模型
                saver = tf.train.import_meta_graph(self.checkpoint_file_meta)
                saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint))

                self.graph = tf.get_default_graph()

                # input
                self.x = self.graph.get_tensor_by_name('x:0')
                self.y = self.graph.get_tensor_by_name('y:0')
                # output
                self.logits = self.graph.get_tensor_by_name('ResizeBilinear_1:0')
                self.prediction = self.graph.get_tensor_by_name('prediction:0')
                self.is_correct = self.graph.get_tensor_by_name('is_correct:0')
                self.accuracy = self.graph.get_tensor_by_name('accuracy:0')

                # loss
                self.loss = self.graph.get_tensor_by_name('loss/0:0')
                # train
                self.train_op = self.graph.get_tensor_by_name('train_op')

                # load batch
                self.images, self.labels = self.graph.get_tensor_by_name('shuffle_batch')
            pass
        pass

    # 卷积
    def _conv(self, name, input, strides=list([1, 1, 1, 1]), padding="VALID", add_bias=True, apply_relu=True, atrous_rate=None):
        with tf.variable_scope(name):
            w_kernel = tf.Variable(initial_value=self.w_pretrained[name + '/kernel:0'], trainable=self.trainable)

            if atrous_rate is None:
                conv_out = tf.nn.conv2d(input, w_kernel, strides, padding)
            else:
                conv_out = tf.nn.atrous_conv2d(input, w_kernel, atrous_rate, padding)

            if add_bias:
                w_bias = tf.Variable(initial_value=self.w_pretrained[name + '/bias:0'], trainable=self.trainable)
                conv_out = tf.nn.bias_add(conv_out, w_bias)

            if apply_relu:
                conv_out = tf.nn.relu(conv_out)

        return conv_out

    def dilated_convolution_by_pretrained(self):
        # Check on dataset name
        if self.dataset not in ['cityscapes', 'camvid']:
            raise ValueError('Dataset "{}" not supported.'.format(self.dataset))
        else:
            conv_margin = Config[self.dataset]["conv_margin"]
            h = tf.pad(self.x, [[0, 0], [conv_margin, conv_margin], [conv_margin, conv_margin], [0, 0]], mode="REFLECT")
            h = self._conv('conv1_1', h)  # [1394, 1394, 64]
            h = self._conv('conv1_2', h)  # [1392, 1392, 64]
            h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1')  # [696, 696, 64]

            h = self._conv('conv2_1', h)  # [694, 694, 128]
            h = self._conv('conv2_2', h)  # [692, 692, 128]
            h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2')  # [346, 346, 128]

            h = self._conv('conv3_1', h)  # [344, 344, 256]
            h = self._conv('conv3_2', h)  # [342, 342, 256]
            h = self._conv('conv3_3', h)  # [340, 340, 256]
            h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool3')  # [170, 170, 256]

            h = self._conv('conv4_1', h)  # [168, 168, 512]
            h = self._conv('conv4_2', h)  # [166, 166, 512]
            h = self._conv('conv4_3', h)  # [164, 164, 512]

            h = self._conv('conv5_1', h, atrous_rate=2)  # [160, 160, 512]
            h = self._conv('conv5_2', h, atrous_rate=2)  # [156, 156, 512]
            h = self._conv('conv5_3', h, atrous_rate=2)  # [152, 152, 512]
            h = self._conv('fc6', h, atrous_rate=4)  # [128, 128, 4096] (k=7)

            h = tf.layers.dropout(h, rate=0.5, name='drop6')
            h = self._conv('fc7', h)  # [128, 128, 4096]
            h = tf.layers.dropout(h, rate=0.5, name='drop7')
            h = self._conv('final', h)  # [128, 128, 19]
            # 上面属于Front-End

            # 下面开始Context network architecture
            h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad1_1')  # [130, 130, 19]
            h = self._conv('ctx_conv1_1', h)  # [128, 128, 19] receptive field=[3, 3]
            h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad1_2')  # [130, 130, 19]
            h = self._conv('ctx_conv1_2', h)  # [128, 128, 19] rf=[5, 5]

            h = tf.pad(h, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', name='ctx_pad2_1')  # [132, 132, 19]
            h = self._conv('ctx_conv2_1', h, atrous_rate=2)  # [128, 128, 19] rf=[9, 9]

            h = tf.pad(h, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='CONSTANT', name='ctx_pad3_1')  # [136, 136, 19]
            h = self._conv('ctx_conv3_1', h, atrous_rate=4)  # [128, 128, 19] rf=[17, 17]

            h = tf.pad(h, [[0, 0], [8, 8], [8, 8], [0, 0]], mode='CONSTANT', name='ctx_pad4_1')  # [144, 144, 19]
            h = self._conv('ctx_conv4_1', h, atrous_rate=8)  # [128, 128, 19] rf=[33, 33]

            h = tf.pad(h, [[0, 0], [16, 16], [16, 16], [0, 0]], mode='CONSTANT', name='ctx_pad5_1')  # [160, 160, 19]
            h = self._conv('ctx_conv5_1', h, atrous_rate=16)  # [128, 128, 19] rf=[65, 65]

            if self.dataset == 'cityscapes':
                h = tf.pad(h, [[0, 0], [32, 32], [32, 32], [0, 0]], mode='CONSTANT', name='ctx_pad6_1')  # [192, 192, 19]
                h = self._conv('ctx_conv6_1', h, atrous_rate=32)  # [128, 128, 19]  rf=[129, 129]

                h = tf.pad(h, [[0, 0], [64, 64], [64, 64], [0, 0]], mode='CONSTANT', name='ctx_pad7_1')  # [256, 256, 19]
                h = self._conv('ctx_conv7_1', h, atrous_rate=64)  # [128, 128, 19]  rf=[257, 257]
                pass

            h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad_fc1')  # [130, 130, 19]
            h = self._conv('ctx_fc1', h)  # [128, 128, 19]
            h = self._conv('ctx_final', h, padding='VALID', add_bias=True, apply_relu=False)  # [128, 128, 19]

            if self.dataset == 'cityscapes':
                h = tf.image.resize_bilinear(h, size=(1024, 1024))  # [1024, 1024, 19]
                logits = self._conv('ctx_upsample', h, padding='SAME', add_bias=False, apply_relu=True)  # [1024, 1024, 19]
            else:
                logits = h  # [128, 128, 19]

        return tf.image.resize_images(logits, size=[self.data.image_height, self.data.image_width])


    def loss_cross_entropy(self):
        with tf.name_scope('loss'):
            # reshape label
            labels = tf.squeeze(tf.cast(self.y, tf.int32), axis=3)
            # compute the cross entropy of logits vs labels
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.logits)
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name="0")
        return cross_entropy_mean

    def get_train_op(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, name="train_op")

    # 如果还没有模型，加载一份预训练的
    def need_load_pretrained(self):
        # 转换模型
        w_pretrained = None
        if not os.path.exists(self.checkpoint_file_meta):
            print('Loading pre-trained weights...')
            with open(Config[self.dataset]['weights_file'], 'rb') as f:
                w_pretrained = pickle.load(f)
            print('Loading pre-trained weights Done.')
        print("Model has existed ...")

        return w_pretrained

    # this function is the same as the one in the original repository
    # basically it performs upsampling for datasets having zoom > 1
    @staticmethod
    def interp_map(prob, zoom, width, height):
        channels = prob.shape[2]
        zoom_prob = tf.zeros((height, width, channels), dtype=tf.float32)
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    r0 = h // zoom
                    r1 = r0 + 1
                    c0 = w // zoom
                    c1 = c0 + 1
                    rt = float(h) / zoom - r0
                    ct = float(w) / zoom - c0
                    v0 = rt * prob[r1, c0, c] + (1 - rt) * prob[r0, c0, c]
                    v1 = rt * prob[r1, c1, c] + (1 - rt) * prob[r0, c1, c]
                    zoom_prob[h, w, c] = (1 - ct) * v0 + ct * v1
        return zoom_prob

    pass
