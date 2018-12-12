import itertools
import os
import random
from collections import OrderedDict
import tensorflow as tf
import numpy as np

from .base_dataset import VideoDataset


class SweeperVideoDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        super(SweeperVideoDataset, self).__init__(*args, **kwargs)
        self.dataset_name = os.path.basename(os.path.split(self.input_dir)[0])
        self.state_like_names_and_shapes['images'] = 'move/%d/image/encoded', (64, 64, 3)
        self.state_like_names_and_shapes['states'] = 'move/%d/state/position', (5,)
        self.action_like_names_and_shapes['actions'] = 'move/%d/action/velocity', (5,)
        self._check_or_infer_shapes()

        self.sess = tf.Session()

    def get_default_hparams_dict(self):
        default_hparams = super(SweeperVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=15,
            use_state=True,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    @property
    def jpeg_encoding(self):
        return True

    def num_examples_per_epoch(self):
        #if os.path.basename(self.input_dir) == 'random_tf_records':
        #    count = 900
        #else:
        #    raise NotImplementedError
        return 1000

    def parser(self, serialized_example):
        """
        Parses a single tf.train.Example into images, states, actions, etc tensors.
        """
        features = dict()
        for i in range(self._max_sequence_length):
            for example_name, (name, shape) in self.state_like_names_and_shapes.items():
                if example_name == 'images':  # special handling for image
                    features[name % i] = tf.FixedLenFeature([64*64*3], tf.string)
                else:
                    features[name % i] = tf.FixedLenFeature(shape, tf.float32)
        for i in range(self._max_sequence_length - 1):
            for example_name, (name, shape) in self.action_like_names_and_shapes.items():
                features[name % i] = tf.FixedLenFeature(shape, tf.float32)

        # add boolean feature for action
        features['action_exist'] = tf.FixedLenFeature([1], tf.int64)

        # check that the features are in the tfrecord
        for name in features.keys():
            if name not in self._dict_message['features']['feature']:
                raise ValueError('Feature with name %s not found in tfrecord. Possible feature names are:\n%s' %
                                 (name, '\n'.join(sorted(self._dict_message['features']['feature'].keys()))))

        # parse all the features of all time steps together
        features = tf.parse_single_example(serialized_example, features=features)

        state_like_seqs = OrderedDict([(example_name, []) for example_name in self.state_like_names_and_shapes])
        action_like_seqs = OrderedDict([(example_name, []) for example_name in self.action_like_names_and_shapes])
        for i in range(self._max_sequence_length):
            for example_name, (name, shape) in self.state_like_names_and_shapes.items():
                state_like_seqs[example_name].append(features[name % i])
        for i in range(self._max_sequence_length - 1):
            for example_name, (name, shape) in self.action_like_names_and_shapes.items():
                action_like_seqs[example_name].append(features[name % i])

        # set infer action variable
        # infer_action = 1 - features['action_exist']

        # for this class, it's much faster to decode and preprocess the entire sequence before sampling a slice
        _, image_shape = self.state_like_names_and_shapes['images']
        state_like_seqs['images'] = self.decode_and_preprocess_images(state_like_seqs['images'], image_shape)

        state_like_seqs, action_like_seqs = \
            self.slice_sequences(state_like_seqs, action_like_seqs, self._max_sequence_length)
        return state_like_seqs, action_like_seqs #, infer_action

    # def make_batch(self, batch_size):
    #     filenames = self.filenames
    #     if self.mode == 'train':
    #         random.shuffle(filenames)

    #     dataset = tf.data.TFRecordDataset(filenames)
    #     dataset = dataset.map(self.parser, num_parallel_calls=batch_size)
    #     dataset.prefetch(2 * batch_size)

    #     # Could shuffle individual samples but it becomes too slow. Just shuffle filenames instead.
    #     # if self.mode == 'train':
    #     #     min_queue_examples = int(
    #     #         self.num_examples_per_epoch() * 0.4)
    #     #     # Ensure that the capacity is sufficiently large to provide good random
    #     #     # shuffling.
    #     #     dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    #     dataset = dataset.repeat(self.num_epochs)
    #     dataset = dataset.batch(batch_size)
    #     iterator = dataset.make_one_shot_iterator()
    #     state_like_batches, action_like_batches, infer_action = iterator.get_next()
    #     infer_action = self.sess.run(infer_action)
    #     # import pdb; pdb.set_trace()

    #     infer_action_batches = {'infer_action': infer_action}
    #     input_batches = OrderedDict(list(state_like_batches.items()) + list(action_like_batches.items()) + list(infer_action_batches.items()))
    #     for input_batch in input_batches.values():
    #         if isinstance(input_batch, np.ndarray):
    #             input_batch = np.reshape(input_batch, [batch_size, 1])
    #         else:
    #             input_batch.set_shape([batch_size] + [None] * (input_batch.shape.ndims - 1))
    #     target_batches = state_like_batches['images'][:, self.hparams.context_frames:]
    #     return input_batches, target_batches
