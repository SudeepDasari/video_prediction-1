from sim.agent.general_agent import GeneralAgent
from sim.policy.pushing_policy import PushingPolicy
from sim.mujoco_env.cartgripper_env.cartgripper_xz_grasp import CartgripperXZGrasp
k
BASE_DIR='/home/sudeep/Desktop/video_prediction-1/data/sim_data'
env_params = {
        # resolution sufficient for 16x anti-aliasing
        'viewer_image_height': 96,
        'viewer_image_width': 128,
        'cube_objects': True
}
agent_params = {
    'env': (CartgripperXZGrasp, env_params),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height': 48,
    'image_width': 64,
    'gen_xml': 1,  # generate xml every nth trajecotry
    'record': BASE_DIR + '/record/',
    'rejection_sample': 5
}


def main():
    num_collect = 100
    agent = GeneralAgent(agent_params)
    policy = PushingPolicy({})

    for i in range(num_collect):
        agent.sample(policy, i)

if __name__ == '__main__':
    main()
