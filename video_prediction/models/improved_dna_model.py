import itertools
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from video_prediction import ops
from video_prediction import tf_utils
from video_prediction.models import VideoPredictionModel
from video_prediction.models import pix2pix_model, mocogan_model
from video_prediction.ops import lrelu, dense, pad2d, conv2d, upsample_conv2d, conv_pool2d, flatten, tile_concat, pool2d
from video_prediction.rnn_ops import BasicConv2DLSTMCell

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12


def create_legacy_encoder(inputs,
                          nz=8,
                          nef=64,
                          norm_layer='instance'):
    norm_layer = ops.get_norm_layer(norm_layer)

    with tf.variable_scope('h1'):
        h1 = conv_pool2d(inputs, nef, kernel_size=5, strides=2)
        h1 = norm_layer(h1)
        h1 = tf.nn.relu(h1)

    with tf.variable_scope('h2'):
        h2 = conv_pool2d(h1, nef * 2, kernel_size=5, strides=2)
        h2 = norm_layer(h2)
        h2 = tf.nn.relu(h2)

    with tf.variable_scope('h3'):
        h3 = conv_pool2d(h2, nef * 4, kernel_size=5, strides=2)
        h3 = norm_layer(h3)
        h3 = tf.nn.relu(h3)
        h3_flatten = flatten(h3)

    with tf.variable_scope('z_mu'):
        z_mu = dense(h3_flatten, nz)
    with tf.variable_scope('z_log_sigma_sq'):
        z_log_sigma_sq = dense(h3_flatten, nz)
        z_log_sigma_sq = tf.clip_by_value(z_log_sigma_sq, -10, 10)
    outputs = {'enc_zs_mu': z_mu, 'enc_zs_log_sigma_sq': z_log_sigma_sq}
    return outputs


def create_n_layer_encoder(inputs,
                           nz=8,
                           nef=64,
                           n_layers=3,
                           norm_layer='instance'):
    norm_layer = ops.get_norm_layer(norm_layer)
    layers = []
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]

    with tf.variable_scope("layer_1"):
        convolved = conv2d(tf.pad(inputs, paddings), nef, kernel_size=4, strides=2, padding='VALID')
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    for i in range(1, n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = nef * min(2**i, 4)
            convolved = conv2d(tf.pad(layers[-1], paddings), out_channels, kernel_size=4, strides=2, padding='VALID')
            normalized = norm_layer(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    pooled = pool2d(rectified, rectified.shape[1:3].as_list(), padding='VALID', pool_mode='avg')
    squeezed = tf.squeeze(pooled, [1, 2])

    with tf.variable_scope('z_mu'):
        z_mu = dense(squeezed, nz)
    with tf.variable_scope('z_log_sigma_sq'):
        z_log_sigma_sq = dense(squeezed, nz)
        z_log_sigma_sq = tf.clip_by_value(z_log_sigma_sq, -10, 10)
    outputs = {'enc_zs_mu': z_mu, 'enc_zs_log_sigma_sq': z_log_sigma_sq}
    return outputs


def create_encoder(inputs, e_net='legacy', **kwargs):
    should_flatten = inputs.shape.ndims > 4
    if should_flatten:
        batch_shape = inputs.shape[:-3].as_list()
        inputs = flatten(inputs, 0, len(batch_shape) - 1)

    if e_net == 'legacy':
        kwargs.pop('n_layers', None)  # unused
        outputs = create_legacy_encoder(inputs, **kwargs)
    elif e_net == 'n_layer':
        outputs = create_n_layer_encoder(inputs, **kwargs)
    else:
        raise ValueError('Invalid encoder net %s' % e_net)

    if should_flatten:
        def unflatten(x):
            return tf.reshape(x, batch_shape + x.shape.as_list()[1:])
        outputs = nest.map_structure(unflatten, outputs)
    return outputs


def encoder_fn(inputs, hparams=None):
    images = inputs['images']
    image_pairs = tf.concat([images[:hparams.sequence_length - 1],
                             images[1:hparams.sequence_length]], axis=-1)
    if 'actions' in inputs:
        image_pairs = tile_concat([image_pairs,
                                   tf.expand_dims(tf.expand_dims(inputs['actions'], axis=-2), axis=-2)], axis=-1)
    outputs = create_encoder(image_pairs,
                             e_net=hparams.e_net,
                             nz=hparams.nz,
                             nef=hparams.nef,
                             n_layers=hparams.n_layers,
                             norm_layer=hparams.norm_layer)
    return outputs


def discriminator_fn(targets, inputs=None, hparams=None):
    outputs = {}
    if hparams.gan_weight or hparams.vae_gan_weight:
        _, pix2pix_outputs = pix2pix_model.discriminator_fn(targets, inputs=inputs, hparams=hparams)
        outputs.update(pix2pix_outputs)
    if hparams.image_gan_weight or hparams.image_vae_gan_weight or \
            hparams.video_gan_weight or hparams.video_vae_gan_weight:
        _, mocogan_outputs = mocogan_model.discriminator_fn(targets, inputs=inputs, hparams=hparams)
        outputs.update(mocogan_outputs)
    return None, outputs


class DNACell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, inputs, hparams, reuse=None):
        super(DNACell, self).__init__(_reuse=reuse)
        self.inputs = inputs
        self.hparams = hparams

        batch_size = inputs['images'].shape[1].value
        image_shape = inputs['images'].shape.as_list()[2:]
        height, width, _ = image_shape
        scale_size = min(height, width)
        if scale_size == 256:
            self.encoder_layer_specs = [
                (self.hparams.ngf, False),
                (self.hparams.ngf * 2, False),
                (self.hparams.ngf * 4, True),
                (self.hparams.ngf * 8, True),
                (self.hparams.ngf * 8, True),
            ]
            self.decoder_layer_specs = [
                (self.hparams.ngf * 8, True),
                (self.hparams.ngf * 4, True),
                (self.hparams.ngf * 2, False),
                (self.hparams.ngf, False),
                (self.hparams.ngf, False),
            ]
        elif scale_size == 64:
            self.encoder_layer_specs = [
                (self.hparams.ngf, True),
                (self.hparams.ngf * 2, True),
                (self.hparams.ngf * 4, True),
            ]
            self.decoder_layer_specs = [
                (self.hparams.ngf * 2, True),
                (self.hparams.ngf, True),
                (self.hparams.ngf, False),
            ]
        else:
            raise NotImplementedError

        # output_size
        gen_input_shape = list(image_shape)
        if 'actions' in inputs:
            gen_input_shape[-1] += inputs['actions'].shape[-1].value
        num_masks = int(bool(self.hparams.first_image_background)) + \
                    int(bool(self.hparams.prev_image_background)) + \
                    int(bool(self.hparams.generate_scratch_image)) + \
                    self.hparams.num_transformed_images
        output_size = {
            'gen_images': tf.TensorShape(image_shape),
            'gen_inputs': tf.TensorShape(gen_input_shape),
            'transformed_images': tf.TensorShape(image_shape + [num_masks]),
            'masks': tf.TensorShape([height, width, 1, num_masks]),
        }
        if 'pix_distribs' in inputs:
            output_size['gen_pix_distrib'] = tf.TensorShape([height, width, 1])
            output_size['transformed_pix_distribs'] = tf.TensorShape([height, width, 1, num_masks])
        if 'states' in inputs:
            output_size['gen_states'] = inputs['states'].shape[2:]
        self._output_size = output_size

        # state_size
        if self.hparams.lstm_skip_connection:
            lstm_filters_multiplier = 2
        else:
            lstm_filters_multiplier = 1
        lstm_cell_sizes = []
        lstm_state_sizes = []
        lstm_height, lstm_width = height, width
        for out_channels, use_lstm in self.encoder_layer_specs:
            lstm_height //= 2
            lstm_width //= 2
            if use_lstm:
                lstm_cell_sizes.append(tf.TensorShape([lstm_height, lstm_width, out_channels]))
                lstm_state_sizes.append(tf.TensorShape([lstm_height, lstm_width, lstm_filters_multiplier * out_channels]))
        for out_channels, use_lstm in self.decoder_layer_specs:
            lstm_height *= 2
            lstm_width *= 2
            if use_lstm:
                lstm_cell_sizes.append(tf.TensorShape([lstm_height, lstm_width, out_channels]))
                lstm_state_sizes.append(tf.TensorShape([lstm_height, lstm_width, lstm_filters_multiplier * out_channels]))
        lstm_states_size = [tf.nn.rnn_cell.LSTMStateTuple(lstm_cell_size, lstm_state_size)
                            for lstm_cell_size, lstm_state_size in zip(lstm_cell_sizes, lstm_state_sizes)]
        state_size = {'time': tf.TensorShape([]),
                      'gen_image': tf.TensorShape(image_shape),
                      'lstm_states': lstm_states_size}
        if 'zs' in inputs and self.hparams.use_lstm_z:
            state_size['lstm_z_state'] = tf.nn.rnn_cell.LSTMStateTuple(*[tf.TensorShape([self.hparams.nz])] * 2)
        if 'pix_distribs' in inputs:
            state_size['gen_pix_distrib'] = tf.TensorShape([height, width, 1])
        if 'states' in inputs:
            state_size['gen_state'] = inputs['states'].shape[2:]
        self._state_size = state_size

        ground_truth_sampling_shape = [self.hparams.sequence_length - 1 - self.hparams.context_frames, batch_size]
        if self.hparams.schedule_sampling == 'none':
            ground_truth_sampling = tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape)
        elif self.hparams.schedule_sampling in ('inverse_sigmoid', 'linear'):
            if self.hparams.schedule_sampling == 'inverse_sigmoid':
                k = self.hparams.schedule_sampling_k
                start_step = self.hparams.schedule_sampling_steps[0]
                iter_num = tf.to_float(tf.train.get_or_create_global_step())
                prob = (k / (k + tf.exp((iter_num - start_step) / k)))
                prob = tf.cond(tf.less(iter_num, start_step), lambda: 1.0, lambda: prob)
            elif self.hparams.schedule_sampling == 'linear':
                start_step, end_step = self.hparams.schedule_sampling_steps
                step = tf.clip_by_value(tf.train.get_or_create_global_step(), start_step, end_step)
                prob = 1.0 - tf.to_float(step - start_step) / tf.to_float(end_step - start_step)
            log_probs = tf.log([1 - prob, prob])
            ground_truth_sampling = tf.multinomial([log_probs] * batch_size, ground_truth_sampling_shape[0])
            ground_truth_sampling = tf.cast(tf.transpose(ground_truth_sampling, [1, 0]), dtype=tf.bool)
            # Ensure that eventually, the model is deterministically
            # autoregressive (as opposed to autoregressive with very high probability).
            ground_truth_sampling = tf.cond(tf.less(prob, 0.001),
                                            lambda: tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape),
                                            lambda: ground_truth_sampling)
        else:
            raise NotImplementedError
        ground_truth_context = tf.constant(True, dtype=tf.bool, shape=[self.hparams.context_frames, batch_size])
        self.ground_truth = tf.concat([ground_truth_context, ground_truth_sampling], axis=0)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def _lstm_func(self, inputs, state, num_units):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units, reuse=tf.get_variable_scope().reuse)
        return lstm_cell(inputs, state)

    def _conv_lstm_func(self, inputs, state, filters):
        inputs_shape = inputs.get_shape().as_list()
        input_shape = inputs_shape[1:]
        if self.hparams.norm_layer == 'none':
            normalizer_fn = None
        else:
            normalizer_fn = ops.get_norm_layer(self.hparams.norm_layer)
        lstm_cell = BasicConv2DLSTMCell(input_shape, filters, kernel_size=(5, 5),
                                        normalizer_fn=normalizer_fn,
                                        separate_norms=self.hparams.norm_layer == 'layer',
                                        skip_connection=self.hparams.lstm_skip_connection,
                                        reuse=tf.get_variable_scope().reuse)
        return lstm_cell(inputs, state)

    def call(self, inputs, states):
        norm_layer = ops.get_norm_layer(self.hparams.norm_layer)
        image_shape = inputs['images'].get_shape().as_list()
        batch_size, height, width, color_channels = image_shape
        lstm_states = states['lstm_states']

        time = states['time']
        with tf.control_dependencies([tf.assert_equal(time[1:], time[0])]):
            t = tf.to_int32(tf.identity(time[0]))

        image = tf.where(self.ground_truth[t], inputs['images'], states['gen_image'])  # schedule sampling (if any)
        if 'pix_distribs' in inputs:
            pix_distrib = tf.where(self.ground_truth[t], inputs['pix_distribs'], states['gen_pix_distribs'])

        state_action = []
        if 'actions' in inputs:
            state_action.append(inputs['actions'])
        if 'states' in inputs:
            # feed in ground_truth state only for first time step
            state = tf.cond(tf.equal(t, 0), lambda: inputs['states'], lambda: states['gen_state'])
            state_action.append(state)

        state_action_z = list(state_action)
        if 'zs' in inputs:
            if self.hparams.use_lstm_z:
                with tf.variable_scope('lstm_z'):
                    lstm_z, lstm_z_state = self._lstm_func(inputs['zs'], states['lstm_z_state'], self.hparams.nz)
                state_action_z.append(lstm_z)
            else:
                state_action_z.append(inputs['zs'])

        state_action = tf.concat(state_action, axis=-1)
        state_action_z = tf.concat(state_action_z, axis=-1)
        if 'actions' in inputs:
            gen_input = tile_concat([image, inputs['actions'][:, None, None, :]], axis=-1)
        else:
            gen_input = image

        layers = []
        new_lstm_states = []
        for i, (out_channels, use_lstm) in enumerate(self.encoder_layer_specs):
            with tf.variable_scope('h%d' % i):
                if i == 0:
                    h = tf.concat([image, self.inputs['images'][0]], axis=-1)
                    kernel_size = (5, 5)
                else:
                    h = layers[-1][-1]
                    kernel_size = (3, 3)
                h = conv_pool2d(tile_concat([h, state_action_z[:, None, None, :]], axis=-1),
                                out_channels, kernel_size=kernel_size, strides=(2, 2))
                h = norm_layer(h)
                h = tf.nn.relu(h)
            if use_lstm:
                lstm_state = lstm_states[len(new_lstm_states)]
                with tf.variable_scope('lstm_h%d' % i):
                    lstm_h, lstm_state = self._conv_lstm_func(tile_concat([h, state_action_z[:, None, None, :]], axis=-1),
                                                              lstm_state, out_channels)
                new_lstm_states.append(lstm_state)
            layers.append((h, lstm_h) if use_lstm else (h,))

        num_encoder_layers = len(layers)
        for i, (out_channels, use_lstm) in enumerate(self.decoder_layer_specs):
            with tf.variable_scope('h%d' % len(layers)):
                if i == 0:
                    h = layers[-1][-1]
                else:
                    h = tf.concat([layers[-1][-1], layers[num_encoder_layers - i - 1][-1]], axis=-1)
                h = upsample_conv2d(tile_concat([h, state_action_z[:, None, None, :]], axis=-1),
                                    out_channels, kernel_size=(3, 3), strides=(2, 2))
                h = norm_layer(h)
                h = tf.nn.relu(h)
            if use_lstm:
                lstm_state = lstm_states[len(new_lstm_states)]
                with tf.variable_scope('lstm_h%d' % len(layers)):
                    lstm_h, lstm_state = self._conv_lstm_func(tile_concat([h, state_action_z[:, None, None, :]], axis=-1),
                                                              lstm_state, out_channels)
                new_lstm_states.append(lstm_state)
            layers.append((h, lstm_h) if use_lstm else (h,))
        assert len(new_lstm_states) == len(lstm_states)

        if self.hparams.num_transformed_images:
            assert len(self.hparams.kernel_size) == 2
            kernel_shape = list(self.hparams.kernel_size) + [self.hparams.num_transformed_images]
            if self.hparams.transformation == 'dna':
                with tf.variable_scope('h%d_dna_kernel' % len(layers)):
                    h_dna_kernel = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                    h_dna_kernel = norm_layer(h_dna_kernel)
                    h_dna_kernel = tf.nn.relu(h_dna_kernel)

                # Using largest hidden state for predicting untied conv kernels.
                with tf.variable_scope('dna_kernels'):
                    kernels = conv2d(h_dna_kernel, np.prod(kernel_shape), kernel_size=(3, 3), strides=(1, 1))
                    kernels = tf.reshape(kernels, [batch_size, height, width] + kernel_shape)
                kernel_spatial_axes = [3, 4]
            elif self.hparams.transformation == 'cdna':
                with tf.variable_scope('cdna_kernels'):
                    smallest_layer = layers[num_encoder_layers - 1][-1]
                    kernels = dense(flatten(smallest_layer), np.prod(kernel_shape))
                    kernels = tf.reshape(kernels, [batch_size] + kernel_shape)
                kernel_spatial_axes = [1, 2]
            else:
                raise ValueError('Invalid transformation %s' % self.hparams.transformation)

            with tf.name_scope('kernel_normalization'):
                kernels = tf.nn.relu(kernels - RELU_SHIFT) + RELU_SHIFT
                kernels /= tf.reduce_sum(kernels, axis=kernel_spatial_axes, keep_dims=True)

        if self.hparams.generate_scratch_image:
            with tf.variable_scope('h%d_scratch' % len(layers)):
                h_scratch = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                h_scratch = norm_layer(h_scratch)
                h_scratch = tf.nn.relu(h_scratch)

            # Using largest hidden state for predicting a new image layer.
            # This allows the network to also generate one image from scratch,
            # which is useful when regions of the image become unoccluded.
            with tf.variable_scope('scratch_image'):
                scratch_image = conv2d(h_scratch, color_channels, kernel_size=(3, 3), strides=(1, 1))
                scratch_image = tf.nn.sigmoid(scratch_image)

        with tf.name_scope('transformed_images'):
            transformed_images = []
            if self.hparams.num_transformed_images:
                transformed_images += apply_kernels(image, kernels, self.hparams.dilation_rate)
            if self.hparams.prev_image_background:
                transformed_images.append(image)
            if self.hparams.first_image_background:
                transformed_images.append(self.inputs['images'][0])
            if self.hparams.generate_scratch_image:
                transformed_images.append(scratch_image)

        if 'pix_distribs' in inputs:
            with tf.name_scope('transformed_pix_distribs'):
                transformed_pix_distribs = []
                if self.hparams.num_transformed_images:
                    transformed_pix_distribs += apply_kernels(pix_distrib, kernels, self.hparams.dilation_rate)
                if self.hparams.prev_image_background:
                    transformed_pix_distribs.append(pix_distrib)
                if self.hparams.first_image_background:
                    transformed_pix_distribs.append(self.inputs['pix_distribs'][0])
                if self.hparams.generate_scratch_image:
                    transformed_pix_distribs.append(pix_distrib)

        with tf.name_scope('masks'):
            if len(transformed_images) > 1:
                with tf.variable_scope('h%d_masks' % len(layers)):
                    h_masks = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                    h_masks = norm_layer(h_masks)
                    h_masks = tf.nn.relu(h_masks)

                with tf.variable_scope('masks'):
                    if self.hparams.dependent_mask:
                        h_masks = tf.concat([h_masks] + transformed_images, axis=-1)
                    masks = conv2d(h_masks, len(transformed_images), kernel_size=(3, 3), strides=(1, 1))
                    masks = tf.nn.softmax(masks)
                    masks = tf.split(masks, len(transformed_images), axis=-1)
            elif len(transformed_images) == 1:
                masks = [tf.ones([batch_size, height, width, 1])]
            else:
                raise ValueError("Either one of the following should be true: "
                                 "num_transformed_images, first_image_background, "
                                 "prev_image_background, generate_scratch_image")

        with tf.name_scope('gen_images'):
            assert len(transformed_images) == len(masks)
            gen_image = tf.add_n([transformed_image * mask
                                  for transformed_image, mask in zip(transformed_images, masks)])

        if 'pix_distribs' in inputs:
            with tf.name_scope('gen_pix_distribs'):
                assert len(transformed_pix_distribs) == len(masks)
                gen_pix_distrib = tf.add_n([transformed_pix_distrib * mask
                                            for transformed_pix_distrib, mask in zip(transformed_pix_distribs, masks)])
                gen_pix_distrib /= tf.reduce_sum(gen_pix_distrib, axis=(1, 2), keep_dims=True)

        if 'states' in inputs:
            with tf.name_scope('gen_states'):
                with tf.variable_scope('state_pred'):
                    gen_state = dense(state_action, inputs['states'].shape[-1].value)

        outputs = {'gen_images': gen_image,
                   'gen_inputs': gen_input,
                   'transformed_images': tf.stack(transformed_images, axis=-1),
                   'masks': tf.stack(masks, axis=-1)}
        if 'pix_distribs' in inputs:
            outputs['gen_pix_distrib'] = gen_pix_distrib
            outputs['transformed_pix_distribs'] = tf.stack(transformed_pix_distribs, axis=-1)
        if 'states' in inputs:
            outputs['gen_states'] = gen_state

        new_states = {'time': time + 1,
                      'gen_image': gen_image,
                      'lstm_states': new_lstm_states}
        if 'zs' in inputs and self.hparams.use_lstm_z:
            new_states['lstm_z_state'] = lstm_z_state
        if 'pix_distribs' in inputs:
            new_states['gen_pix_distrib'] = gen_pix_distrib
        if 'states' in inputs:
            new_states['gen_state'] = gen_state
        return outputs, new_states


def generator_fn(inputs, outputs_enc=None, hparams=None):
    batch_size = inputs['images'].shape[1].value
    inputs = OrderedDict([(name, tf_utils.maybe_pad_or_slice(input, hparams.sequence_length - 1))
                          for name, input in inputs.items()])
    if hparams.nz:
        if outputs_enc is None:
            zs = tf.random_normal([hparams.sequence_length - 1, batch_size, hparams.nz], 0, 1)
        else:
            enc_zs_mu = outputs_enc['enc_zs_mu']
            enc_zs_log_sigma_sq = outputs_enc['enc_zs_log_sigma_sq']
            eps = tf.random_normal([hparams.sequence_length - 1, batch_size, hparams.nz], 0, 1)
            zs = enc_zs_mu + tf.sqrt(tf.exp(enc_zs_log_sigma_sq)) * eps
        inputs['zs'] = zs
    else:
        if outputs_enc is not None:
            raise ValueError('outputs_enc has to be None when nz is 0.')
    cell = DNACell(inputs, hparams)
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=[hparams.sequence_length - 1] * batch_size,
                                   dtype=tf.float32, swap_memory=False, time_major=True)
    # the RNN outputs generated images from time step 1 to sequence_length,
    # but generator_fn should only return images past context_frames
    gen_images = outputs['gen_images'][hparams.context_frames - 1:]
    outputs['ground_truth_sampling_mean'] = tf.reduce_mean(tf.to_float(cell.ground_truth[hparams.context_frames:]))
    return gen_images, outputs


class ImprovedDNAVideoPredictionModel(VideoPredictionModel):
    def __init__(self, *args, **kwargs):
        super(ImprovedDNAVideoPredictionModel, self).__init__(
            generator_fn, discriminator_fn, encoder_fn, *args, **kwargs)
        if self.hparams.e_net == 'none' or self.hparams.nz == 0:
            self.encoder_fn = None

    def get_default_hparams_dict(self):
        default_hparams = super(ImprovedDNAVideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            l1_weight=1.0,
            l2_weight=0.0,
            d_net='legacy',
            n_layers=3,
            ndf=32,
            norm_layer='instance',
            d_downsample_layer='conv_pool2d',
            ngf=32,
            transformation='cdna',
            kernel_size=(5, 5),
            dilation_rate=(1, 1),
            lstm_skip_connection=False,
            num_transformed_images=4,
            first_image_background=True,
            prev_image_background=True,
            generate_scratch_image=True,
            dependent_mask=True,
            schedule_sampling='inverse_sigmoid',
            schedule_sampling_k=900.0,
            schedule_sampling_steps=(0, 100000),
            e_net='legacy',
            nz=8,
            nef=32,
            use_lstm_z=True,
            d_context_frames=1,
            d_stop_gradient_inputs=False,
            d_use_gt_inputs=False,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))


def apply_dna_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 6-D of shape
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    dilation_rate = list(dilation_rate) if isinstance(dilation_rate, (tuple, list)) else [dilation_rate] * 2
    batch_size, height, width, color_channels = image.get_shape().as_list()
    batch_size, height, width, kernel_height, kernel_width, num_transformed_images = kernels.get_shape().as_list()
    kernel_size = [kernel_height, kernel_width]

    # Flatten the spatial dimensions.
    kernels_reshaped = tf.reshape(kernels, [batch_size, height, width,
                                            kernel_size[0] * kernel_size[1], num_transformed_images])
    image_padded = pad2d(image, kernel_size, rate=dilation_rate, padding='SAME', mode='SYMMETRIC')
    # Combine channel and batch dimensions into the first dimension.
    image_transposed = tf.transpose(image_padded, [3, 0, 1, 2])
    image_reshaped = flatten(image_transposed, 0, 1)[..., None]
    patches_reshaped = tf.extract_image_patches(image_reshaped, ksizes=[1] + kernel_size + [1],
                                                strides=[1] * 4, rates=[1] + dilation_rate + [1], padding='VALID')
    # Separate channel and batch dimensions, and move channel dimension.
    patches_transposed = tf.reshape(patches_reshaped, [color_channels, batch_size, height, width, kernel_size[0] * kernel_size[1]])
    patches = tf.transpose(patches_transposed, [1, 2, 3, 0, 4])
    # Reduce along the spatial dimensions of the kernel.
    outputs = tf.matmul(patches, kernels_reshaped)
    outputs = tf.unstack(outputs, axis=-1)
    return outputs


def apply_cdna_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 4-D of shape
            `[batch, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    batch_size, height, width, color_channels = image.get_shape().as_list()
    batch_size, kernel_height, kernel_width, num_transformed_images = kernels.get_shape().as_list()
    kernel_size = [kernel_height, kernel_width]
    image_padded = pad2d(image, kernel_size, rate=dilation_rate, padding='SAME', mode='SYMMETRIC')
    # Treat the color channel dimension as the batch dimension since the same
    # transformation is applied to each color channel.
    # Treat the batch dimension as the channel dimension so that
    # depthwise_conv2d can apply a different transformation to each sample.
    kernels = tf.transpose(kernels, [1, 2, 0, 3])
    kernels = tf.reshape(kernels, [kernel_size[0], kernel_size[1], batch_size, num_transformed_images])
    # Swap the batch and channel dimensions.
    image_transposed = tf.transpose(image_padded, [3, 1, 2, 0])
    # Transform image.
    outputs = tf.nn.depthwise_conv2d(image_transposed, kernels, [1, 1, 1, 1], padding='VALID', rate=dilation_rate)
    # Transpose the dimensions to where they belong.
    outputs = tf.reshape(outputs, [color_channels, height, width, batch_size, num_transformed_images])
    outputs = tf.transpose(outputs, [4, 3, 1, 2, 0])
    outputs = tf.unstack(outputs, axis=0)
    return outputs


def apply_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 4-D or 6-D tensor of shape
            `[batch, kernel_size[0], kernel_size[1], num_transformed_images]` or
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    if len(kernels.get_shape()) == 4:
        outputs = apply_cdna_kernels(image, kernels, dilation_rate=dilation_rate)
    elif len(kernels.get_shape()) == 6:
        outputs = apply_dna_kernels(image, kernels, dilation_rate=dilation_rate)
    else:
        raise ValueError
    return outputs
