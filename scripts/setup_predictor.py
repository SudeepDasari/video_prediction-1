from video_prediction.models import SAVPVideoPredictionModel

import tensorflow as tf

import json
import os


DEFAULT_CONFIGS = {
    'num_samples': 50,
    'num_context_frames': 2,
    'sequence_length': 15,
    'action_dim': 5,
    'state_dim': 5,
    'im_width': 64,
    'im_height': 64,
    'num_channels': 3,
    'json_dir': '',
    # 'override_json': '',
    'checkpoint': '',
    'log_dir': '',
}


class VideoPredictor(object):
    def __init__(self, configs, images=None):
        with open(os.path.join(configs['json_dir'], "model_hparams.json")) as f:
            model_hparams_dict = json.loads(f.read())
            model_hparams_dict.pop('num_gpus', None)  # backwards-compatibility
            if 'override_json' in configs:
                model_hparams_dict.update(configs['override_json'])
        with open(os.path.join(configs['json_dir'], "dataset_hparams.json")) as f:
            datatset_hparams_dict = json.loads(f.read())

        self.action_dim = configs['action_dim']
        self.state_dim = configs['state_dim']
        self.sequence_length = model_hparams_dict['sequence_length']
        self.context_frames = model_hparams_dict['context_frames']
        self.im_width = configs['im_width']
        self.im_height = configs['im_height']
        self.num_channels = configs['num_channels']
        self.batch_size = configs['num_samples']

        if images is None:
            self.actions_pl = actions = tf.placeholder(tf.float32, name='actions',
                shape=(self.batch_size, self.sequence_length, self.action_dim))
            self.states_pl = states = tf.placeholder(tf.float32, name='states',
                shape=(self.batch_size, self.sequence_length, self.state_dim))
            self.images_pl = images = tf.placeholder(tf.float32, name='images',
                shape=(self.batch_size, self.sequence_length, self.im_height, self.im_width, self.num_channels))
            self.pix_distrib_pl = pix_distrib_pl = tf.placeholder(tf.float32, name='pix_distrib',
                shape=(self.batch_size, self.sequence_length, self.im_height, self.im_width, 1))
            targets = images[:, self.context_frames:]  # this is a hack to make Alexmodel compute losses. It will take take second image automatically.
        else:
            targets = None

        self.m = SAVPVideoPredictionModel(mode='test', hparams_dict=model_hparams_dict, num_gpus=1)

        inputs = {'actions': actions, 'images': images}
        if datatset_hparams_dict['use_state']:
            inputs['states'] = states

        self.m.build_graph(inputs, targets)

        self.gen_images = [self.m.outputs['gen_images']]

        if datatset_hparams_dict['use_state']:
            self.gen_states = self.m.outputs['gen_states']
        else:
            self.gen_states = None


def setup_predictor(configs=DEFAULT_CONFIGS, gpu_id=1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print('using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"])

    model = VideoPredictor(configs=configs)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer()),
    model.m.restore(sess, configs['checkpoint'])

    num_samples = configs['num_samples']
    sequence_length = configs['sequence_length']
    im_height = configs['im_height']
    im_width = configs['im_width']
    num_channels = configs['num_channels']

    num_context_frames = configs['num_context_frames']
    action_dim = configs['action_dim']

    def predictor_func(input_images=None, input_pix_distrib=None, input_actions=None, input_states=None):
        images_vec = np.zeros((num_samples, sequence_length, im_height, im_width, num_channels))
        images_vec[:, :num_context_frames] = input_images

        pix_distrib_vec = np.zeros((num_samples, sequence_length, im_height, im_width, 1))
        pix_distrib_vec[:, :num_context_frames] = input_pix_distrib

        actions_vec = np.zeros((num_samples, sequence_length, action_dim))
        actions_vec[:, :(sequence_length - num_context_frames)] = input_actions

        states_vec = np.zeros((num_samples, sequence_length, state_dim))
        states_vec[:, :num_context_frames] = input_states

        feed_dict = {
            # model.iter_num: np.float32(0),
            # model.lr: conf['learning_rate'],
            model.images_pl: images_vec,
            model.actions_pl: actions_vec,
            model.states_pl: states_vec,
            model.pix_distrib_pl: pix_distrib_vec,
        }
        gen_images, gen_pix_distrb, gen_states = sess.run([model.gen_images, model.pix_distrib_pl, model.gen_states], feed_dict)
        return gen_images, gen_pix_distrb, gen_states
    return predictor_func


if __name__ == '__main__':
    pred_fn = setup_predictor()
