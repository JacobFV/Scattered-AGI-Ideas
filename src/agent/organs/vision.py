from ..nodes import SensoryNode, N_SAMPLES
from ..lca import DepthwiseConv2DLCA

import tensorflow as tf
import random

keras = tf.keras
tfkl = keras.layers
K = keras.backend


class LCAConvNet(keras.Model):

    def __init__(self, num_features, middle_depth, **kwargs):
        super(LCAConvNet, self).__init__(**kwargs)

        self.LCAConv1 = DepthwiseConv2DLCA(name='eye', filters=middle_depth)
        self.LCAConv2 = DepthwiseConv2DLCA(name='eye', filters=middle_depth)
        self.LCAConv3 = DepthwiseConv2DLCA(name='eye', filters=num_features)

    def call(self, inputs, training=None, mask=None):
        hid1 = self.LCAConv1(inputs, training=training)
        hid2 = self.LCAConv2(hid1, training=training)
        hid3 = self.LCAConv3(hid2, training=training)
        return hid3


class Vision2DNode(SensoryNode):
    """
    zc: y_accel: (? exponential coefficient bases),
        x_accel: (? exponential coefficient bases),
        pupil_zoom_rate: (1)
    zu: features: (feature_depth,)
    """

    def __init__(self,
                 feature_depth,
                 image_size,
                 pupil_size,
                 max_pupil_zoom,
                 max_speed=None,
                 enc=None,
                 name=None):
        """

        :param feature_depth: number
        :param image_size: (2-list or double tuple)
        :param pupil_size: (2-list or double tuple)
        :param max_pupil_zoom: number
        :param max_speed: (2-list or double tuple)
        :param enc: 4D->2D image model
        :param name: sensor name string
        """

        assert isinstance(feature_depth, int)
        image_size = tf.constant(image_size, dtype=K.floatx())
        pupil_size = tf.constant(pupil_size, dtype=K.floatx())
        max_pupil_zoom = tf.constant(max_pupil_zoom, dtype=K.floatx())
        self.image_size = tf.constant(image_size, dtype=K.floatx())
        self.pupil_size_after_scaling = pupil_size
        self.max_pupil_zoom = max_pupil_zoom
        self.pupil_zoom = (1 + self.max_pupil_zoom) / 2.
        self.pos = self.image_size / 2.
        self.vel = tf.ones((2,))
        if max_speed is None:
            max_speed = self.pupil_size_on_canvas / 2.
            if tf.rank(max_speed) == 1:
                max_speed = tf.reduce_mean(max_speed)
        else:
            max_speed = tf.constant(max_speed, shape=(),
                                    dtype=keras.backend.floatx())
        self.max_speed = max_speed
        self.num_accel_coefs = tf.round(tf.math.log(self.max_speed)+1)
        self.accel_coef = tf.exp(tf.range(self.num_accel_coefs))
        self.dual_accel_coef = tf.tile(self.accel_coef, multiples=[2])
        if enc is None:
            enc = LCAConvNet(num_features=64, middle_depth=128)
        self.enc = enc

        super(Vision2DNode, self).__init__(d_zc=2*self.num_accel_coefs+1,
                                           d_zu=feature_depth,
                                           name=name)

    @property
    def pupil_size_on_canvas(self):
        return tf.round(self.pupil_size_after_scaling*self.pupil_zoom)

    def update_state(self, image):
        if tf.rank(image) == 3:
            image = image[tf.newaxis, ...]
        elif tf.rank(image) == 4:
            pass
        else:
            raise NotImplementedError()

        image = tf.cast(image, K.floatx()) / 255.

        sample_index = random.randint(0, N_SAMPLES-1)
        self.zcs = self.child_targets[-1][-1].sample()
        self.child_targets.clear()

        pupil_zoom_rate = self.zcs[sample_index, -1]
        self.pupil_zoom = tf.clip_by_value(self.pupil_zoom + pupil_zoom_rate, 1., self.max_pupil_zoom)

        accel = tf.reduce_sum(tf.reshape(self.dual_accel_coef*self.zcs[sample_index, 0:-1],
                              shape=(2,-1)), axis=-1)
        self.vel = tf.clip_by_value(self.vel + accel, 0, self.max_speed)
        self.pos = tf.clip_by_value(self.pos + self.vel,
                                    self.pupil_size_on_canvas // 2,
                                    self.image_size - (self.pupil_size_on_canvas // 2))

        cropped_image = tf.image.crop_to_bounding_box(image,
            tf.cast(self.pos[0] - (self.pupil_size_on_canvas[0] // 2), tf.int64),
            tf.cast(self.pos[1] - (self.pupil_size_on_canvas[1] // 2), tf.int64),
            tf.cast(self.pupil_size_on_canvas[0], tf.int64),
            tf.cast(self.pupil_size_on_canvas[1], tf.int64))
        rescaled_image = tf.image.resize(cropped_image,
                                         tf.cast(self.pupil_size_after_scaling, tf.int32))
        self.zus = self.enc(rescaled_image, training=True)[0]
