import numpy as np
import argparse
import imageio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dirs', nargs='+')
    parser.add_argument('--max_num', type=int, default=32)

    args = parser.parse_args()

    dirs = args.log_dirs
    big_numpy = None
    for area in dirs:
        file_name = '{}/prediction_eval_psnr_max/outputs/gen_image.npy'.format(area)
        loaded = np.load(file_name)[:args.max_num]
        print(loaded.shape)
        if big_numpy is None:
            big_numpy = loaded
        else:
            big_numpy = np.concatenate([big_numpy, loaded], axis=2)
    
    for i in range(args.max_num):
        imageio.mimsave('consolidated/{}.gif'.format(i), big_numpy[i])
