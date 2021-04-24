from PIL import Image
import os
import time
import glob


class FrameCropper(object):
    def __init__(self, root_dir, csv_file, out_dir, crop_size, stride):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.out_dir = out_dir
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

    def execute(self):
        status = 0
        if not os.path.isdir(self.root_dir):
            raise ValueError('Root directory not exists.')

        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        with open(self.csv_file, 'r') as train_set:
            for video in train_set:
                start_time = time.time()
                video_folder, score, disttype = video.split(' ')
                print('Processing {}'.format(video_folder))
                out_video_folder = os.path.join(self.out_dir, video_folder)
                if not os.path.isdir(out_video_folder):
                    os.makedirs(out_video_folder)
                orig_video_folder = os.path.join(self.root_dir, video_folder)
                if not os.path.isdir(orig_video_folder):
                    print('Video {} not exists.'.format(video_folder))
                    status = 1

                frame_list = [img_name for img_name in os.listdir(orig_video_folder)
                              if img_name.endswith(('.png', '.bmp', '.jpg'))]
                for img_name in frame_list:
                    img_fullname = os.path.join(orig_video_folder, img_name)
                    patches = self._crop_patches_(img_fullname)

                    img_basename, ext = os.path.splitext(img_name)
                    for iii, patch in enumerate(patches):
                        if not os.path.isdir(os.path.join(out_video_folder, str(img_basename))):
                            os.makedirs(os.path.join(out_video_folder, str(img_basename)))
                        patch_name = '{}_{:d}{}'.format(img_basename, iii+1, ext)
                        patch_fullname = os.path.join(out_video_folder, str(img_basename), patch_name)
                        patch.save(patch_fullname)

                duration = time.time()-start_time
                print('Processing time {:f}s'.format(duration))

        return status

    def _crop_patches_(self, img_name):
        img = Image.open(img_name)
        w, h = img.size
        img = img.crop((160, 160, w - 160, h - 160))
        w, h = img.size

        w_crop, h_crop = self.crop_size
        w_stride, h_stride = self.stride

        w_starts = range(0, w - w_crop + 1, w_stride)
        h_starts = range(0, h - h_crop + 1, h_stride)
        patches = [img.crop((w_start, h_start, w_start + w_crop, h_start + h_crop))
                   for w_start in w_starts for h_start in h_starts]
        return patches


def reorganize(patch_root_path, csv_file):
    with open(csv_file, 'r') as train_set:
        for video in train_set:
            video_folder, score, disttype = video.split(' ')
            print('Processing {}'.format(video_folder))
            full_video_folder = os.path.join(patch_root_path, video_folder)
            if not os.path.isdir(full_video_folder):
                raise ValueError('Video {} not exists'.format(video_folder))
            png_list = glob.glob(os.path.join(full_video_folder, '*.png'))
            frame_indices = list({get_frame_idx(png_name) for png_name in png_list})
            if not frame_indices:
                continue
            patches_per_frame = int(len(png_list)/len(frame_indices))
            for frm_id in frame_indices:
                frm_folder = os.path.join(full_video_folder, str(frm_id))
                if not os.path.isdir(frm_folder):
                    os.makedirs(frm_folder)
                for patch_id in range(patches_per_frame):
                    orig_patch_name = '{:d}_{:d}.png'.format(frm_id, patch_id)
                    orig_patch = os.path.join(full_video_folder, orig_patch_name)
                    new_patch_name = '{:d}_{:d}.png'.format(frm_id, patch_id + 1)
                    new_patch = os.path.join(frm_folder, new_patch_name)
                    if os.path.exists(orig_patch):
                        os.rename(orig_patch, new_patch)


def get_frame_idx(patch_name):
    basename = os.path.basename(patch_name)
    frame_idx = int(basename.split('_')[0])
    return frame_idx


if __name__ == '__main__':
    cropper = FrameCropper('/home/zduanmu/Videos/BVQA/ImagesForWentao/preData/train', '/filesystem2/preData/train/waterloo_tmp2.txt',
                           '/filesystem2/preData/train/patches', 235, 128)
    cropper.execute()
    # reorganize('/filesystem2/preData/train/patches', '/filesystem2/preData/train/waterloo_tmp2.txt')
