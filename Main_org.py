from VMEON_release import Predictor
import argparse
import time

def parse_config():
    """
    Customize some basic configurations
    Args:
        use_cuda: True: Use GPU and cuda to accelerate; False: CPU-compatible version
        num_workers: How many CPU threads will be used in reading video frames
        ckpt: saved model
        stride: spatial stride of crops. Default is 128. Videos with smaller resolutions require smaller strides
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action='store_true', help='Specify whether to use cuda or not')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt", default='./VMEON_release.pt',
                        type=str, help='name of the checkpoint to load')
    parser.add_argument("--stride", type=int, default=128)
    return parser.parse_args()


if __name__ == '__main__':
    """
    The test video could be in three kind of formats:
    1) decoded and saved in gray-scale png images in one folder;
    2) YUV files;
    3) any ffmpeg-supported files;
    """
    config = parse_config()
    vmeon = Predictor(config)

    # # a high quality video can be found in ./HighQuality
    test_vid = './samples/HighQuality'
    t1 = time.time()
    _, predicted_score = vmeon.predict_quality(test_vid, normalize=255)
    print(predicted_score, time.time()-t1)  # 1.4939142

    # # a low quality video can be found in ./LowQuality
    test_vid = './samples/LowQuality'
    t1 = time.time()
    _, predicted_score = vmeon.predict_quality(test_vid)
    print(predicted_score, time.time()-t1)  # -2.0662894

    # a high quality video can be found in V:/ExistingVQADatabases/EVVQ/reference/cartoon_0.yuv
    test_vid = './samples/cartoon_0.yuv'
    t1 = time.time()
    _, predicted_score = vmeon.predict_quality(test_vid, width=640, height=480, pix_fmt='yuv420p', fps = 25)
    print(predicted_score, time.time()-t1)  # 1.4939142

    # a high quality video can be found in V:/ExistingVQADatabases/EVVQ/distorted/cheerleaders_1.yuv
    test_vid = './samples/cheerleaders_4.yuv'
    t1 = time.time()
    _, predicted_score = vmeon.predict_quality(test_vid, width=640, height=480, pix_fmt='yuv420p', fps=25)
    print(predicted_score, time.time()-t1)  # -2.0662894