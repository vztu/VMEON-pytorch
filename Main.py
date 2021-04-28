from VMEON_release import Predictor
import argparse
import pandas
import os
import sys
import time
from PIL import Image
from scipy.io import savemat
from torchvision import datasets, models, transforms

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
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ckpt", default='./VMEON_release.pt',
                        type=str, help='name of the checkpoint to load')
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument('--model_name', type=str, default='VMEON',
                      help='Evaluated BVQA model name.')
    parser.add_argument('--dataset_name', type=str, default='YOUTUBE_UGC',
                      help='Evaluation dataset.')
    parser.add_argument('--vframes_path', type=str, default='video_frames/YOUTUBE_UGC', help='Path to decoded video frames.')
    parser.add_argument("--mos_file", type=str, default='mos_files/YOUTUBE_UGC_metadata.csv')
    parser.add_argument("--dataset_path", type=str, default='/media/ztu/Seagate-ztu-ugc/YT_UGC/original_videos')
    parser.add_argument('--out_file', type=str,
                      default='result/YOUTUBE_UGC_VMEON_feats.mat',
                      help='Output correlation results')
    parser.add_argument("--log_file", type=str, default='logs/logs_debug.log')
    return parser.parse_args()


# read YUV frame from file
def read_YUVframe(f_stream, width, height, idx):
  fr_offset = 1.5
  uv_width = width // 2
  uv_height = height // 2

  f_stream.seek(idx*fr_offset*width*height)

  # Read Y plane
  Y = np.fromfile(f_stream, dtype=np.uint8, count=width*height)
  if len(Y) < width * height:
    Y = U = V = None
    return Y, U, V
  Y = Y.reshape((height, width, 1)).astype(np.float)

  # Read U plane 
  U = np.fromfile(f_stream, dtype=np.uint8, count=uv_width*uv_height)
  if len(U) < uv_width * uv_height:
    Y = U = V = None
    return Y, U, V
  U = U.reshape((uv_height, uv_width, 1)).repeat(2, axis=0).repeat(2, axis=1).astype(np.float)
  # U = cv2.resize(U, (width, height), interpolation=cv2.INTER_CUBIC)

  # Read V plane
  V = np.fromfile(f_stream, dtype=np.uint8, count=uv_width*uv_height)
  if len(V) < uv_width * uv_height:
    Y = U = V = None
    return Y, U, V
  V = V.reshape((uv_height, uv_width, 1)).repeat(2, axis=0).repeat(2, axis=1).astype(np.float)
  # V = cv2.resize(V, (width, height), interpolation=cv2.INTER_CUBIC)
  YUV = np.concatenate((Y, U, V), axis=2)
  return YUV

# ref: https://gist.github.com/chenhu66/41126063f114410a6c8ce5c3994a3ce2
import numpy as np
#input is a RGB numpy array with shape (height,width,3), can be uint,int, float or double, values expected in the range 0..255
#output is a double YUV numpy array with shape (height,width,3), values in the range 0..255
def RGB2YUV( rgb ):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv

#input is an YUV numpy array with shape (height,width,3) can be uint,int, float or double,  values expected in the range 0..255
#output is a double RGB numpy array with shape (height,width,3), values in the range 0..255
def YUV2RGB( yuv ):
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
    rgb = np.dot(yuv,m)
    rgb[:,:,0]-=179.45477266423404
    rgb[:,:,1]+=135.45870971679688
    rgb[:,:,2]-=226.8183044444304
    return rgb.clip(0, 255).astype(np.uint8)

def YUV2RGB_OpenCV(YUV):
    YVU = YUV[:, :, [0, 2, 1]]  # swap UV 
    return cv2.cvtColor(YVU.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

data_transforms = {
    'train': transforms.Compose([
#         transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),
    'val': transforms.Compose([
        transforms.Resize((384,512), interpolation=Image.BILINEAR),
#         transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

# # ----------------------- Set System logger ------------- #
# class Logger:
#   def __init__(self, log_file):
#     self.terminal = sys.stdout
#     self.log = open(log_file, "a")

#   def write(self, message):
#     self.terminal.write(message)
#     self.log.write(message)  

#   def flush(self):
#     #this flush method is needed for python 3 compatibility.
#     #this handles the flush command by doing nothing.
#     #you might want to specify some extra behavior here.
#     pass

if __name__ == '__main__':
    """
    The test video could be in three kind of formats:
    1) decoded and saved in gray-scale png images in one folder;
    2) YUV files;
    3) any ffmpeg-supported files;
    """
    args = parse_config()
    log_dir = os.path.dirname(args.log_file)  # create out file parent dir if not exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(args.vframes_path):
        os.makedirs(args.vframes_path)
    # sys.stdout = Logger(args.log_file)

    video_tmp = '/media/ztu/Data/tmp'  # store tmp decoded .yuv file
    if not os.path.exists(video_tmp):
        os.makedirs(video_tmp)

    out_dir = os.path.dirname(args.out_file)  # create out file parent dir if not exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    mos_mat = pandas.read_csv(args.mos_file)
    num_videos = mos_mat.shape[0]

    #init vmeon model
    vmeon = Predictor(args)
    feats_mat = []
    time_cnts_all = []

    for i in range(230,num_videos):
        if args.dataset_name == 'KONVID_1K':
            video_name = os.path.join(args.dataset_path, str(mos_mat.loc[i, 'flickr_id'])+'.mp4')
        elif args.dataset_name == 'LIVE_VQC':
            video_name = os.path.join(args.dataset_path, mos_mat.loc[i, 'File'])
        elif args.dataset_name == 'YOUTUBE_UGC':
            video_name = os.path.join(args.dataset_path, mos_mat.loc[i, 'category'],
                                str(mos_mat.loc[i, 'resolution'])+'P',
                                mos_mat.loc[i, 'vid']+'.mkv')
        yuv_name = os.path.join(video_tmp, os.path.basename(video_name)+'.yuv')

        print(f"Computing features for {i}th sequence: {video_name}")

        # if mos_mat.loc[i, 'width'] > 1920:
        #     cmd_resize = 'ffmpeg -i ' + video_name + ' -s 1920x1080 -c:v libx264 -crf 1 -preset slow -pix_fmt yuv420p -filter:v fps=fps=30 ' + \
        #         os.path.join(video_tmp, os.path.basename(video_name))
        #     os.system(cmd_resize)
        #     # decode video and store in tmp
        #     cmd = 'ffmpeg -loglevel error -y -i ' + os.path.join(video_tmp, os.path.basename(video_name)) \
        #         + ' -pix_fmt yuv420p -vsync 0 ' + yuv_name
        #     os.system(cmd)
        #     # calculate number of frames 
        #     width = 1920
        #     height = 1080
        #     framerate = 30
        # else:
        # decode video and store in tmp
        cmd = 'ffmpeg -loglevel error -y -i ' + video_name + ' -pix_fmt yuv420p -vsync 0 ' + yuv_name
        os.system(cmd)

        # calculate number of frames 
        width = mos_mat.loc[i, 'width']
        height = mos_mat.loc[i, 'height']
        framerate = int(round(mos_mat.loc[i, 'framerate']))
        test_stream = open(yuv_name, 'r')
        test_stream.seek(0, os.SEEK_END)
        filesize = test_stream.tell()
        num_frames = int(filesize/(height*width*1.5))  # for 8-bit videos
        print(num_frames)
        t_start = time.time()
        try:
            _, predicted_score = vmeon.predict_quality(
                yuv_name, width=width, height=height, pix_fmt='yuv420p', fps=framerate)
        except:
            predicted_score = np.nan
        t_each = time.time() - t_start
        print(predicted_score)  # 1.4939142
        time_cnts_all.append(t_each)
        print(f"Elapsed {t_each} seconds")
        feats_mat.append(predicted_score)
        os.remove(yuv_name)
        if os.path.exists(os.path.join(video_tmp, os.path.basename(video_name))):
            os.remove(os.path.join(video_tmp, os.path.basename(video_name)))
        test_stream.close()
    
    savemat(args.out_file, {"feats_mat": feats_mat})
    # # # a high quality video can be found in ./HighQuality
    # test_vid = './samples/HighQuality'
    # _, predicted_score = vmeon.predict_quality(test_vid, normalize=255)
    # print(predicted_score)  # 1.4939142

    # # # a low quality video can be found in ./LowQuality
    # test_vid = './samples/LowQuality'
    # _, predicted_score = vmeon.predict_quality(test_vid)
    # print(predicted_score)  # -2.0662894

    # # a high quality video can be found in V:/ExistingVQADatabases/EVVQ/reference/cartoon_0.yuv
    # test_vid = './samples/cartoon_0.yuv'
    # _, predicted_score = vmeon.predict_quality(test_vid, width=640, height=480, pix_fmt='yuv420p', fps = 25)
    # print(predicted_score)  # 1.4939142

    # # a high quality video can be found in V:/ExistingVQADatabases/EVVQ/distorted/cheerleaders_1.yuv
    # test_vid = './samples/cheerleaders_4.yuv'
    # _, predicted_score = vmeon.predict_quality(test_vid, width=640, height=480, pix_fmt='yuv420p', fps=25)
    # print(predicted_score)  # -2.0662894