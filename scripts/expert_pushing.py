from sim.agent.general_agent import GeneralAgent
from sim.policy.pushing_policy import PushingPolicy
from sim.mujoco_env.cartgripper_env.cartgripper_xz_grasp import CartgripperXZGrasp


def main():
    env = CartgripperXZGrasp({})


if __name__ == '__main__':
    main()
