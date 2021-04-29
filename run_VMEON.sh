#!/bin/bash


MODELS=('VMEON')

DATASETS=(
  'LIVE_VQC'
  #'YOUTUBE_UGC'
  #'KONVID_1K'
)

for m in "${MODELS[@]}"
do
for DS in "${DATASETS[@]}"
do
  mos_file=mos_files/${DS}_metadata.csv
  out_file=result/${DS}_${m}_feats.mat
  log_file=logs/${DS}_${m}.log
  vframes_path=video_frames/${DS}
  if [ ${DS} = 'LIVE_VQC' ]
  then
    dataset_path=/media/ztu/Seagate-ztu-ugc/LIVE_VQC/VideoDatabase
  elif [ ${DS} = 'KONVID_1K' ]
  then
    dataset_path=/media/ztu/Seagate-ztu-ugc/KONVID_1K/KoNViD_1k_videos
  elif [ ${DS} = 'YOUTUBE_UGC' ]
  then
    dataset_path=/media/ztu/Seagate-ztu-ugc/YT_UGC/original_videos
  else
    echo "Unknown dataset name"
  fi

  # cmd="CUDA_VISIBLE_DEVICES=\"\" python3 demo_extract_paq2piq_scores_ugcvqa.py"
  cmd="python3 Main.py --use_cuda"
  cmd+=" --model_name ${m}"
  cmd+=" --dataset_name ${DS}"
  cmd+=" --dataset_path ${dataset_path}"
  cmd+=" --vframes_path ${vframes_path}"
  cmd+=" --mos_file ${mos_file}"
  cmd+=" --out_file ${out_file}"
  cmd+=" --log_file ${log_file}"

  echo "${cmd}"

  eval ${cmd}

done
done