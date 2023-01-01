upload:
  rsync -avh --exclude='*.mp4' --exclude='multirun' --exclude='outputs'  --exclude='.git' ${WT_LOCAL_DIR} ${WT_REMOTE_DIR}/..

download:
  rsync -avh --exclude='*.mp4' --exclude='multirun' --exclude='outputs' --exclude='cache' --exclude='justfile' --exclude='.git' ${WT_REMOTE_DIR} ${WT_LOCAL_DIR}/..

convert:
  for f in *.mp4; do ffmpeg -i "$f" "${f%.mp4}.wav"; done

run_cuda_:
  whisper crediwire.mp3 --model=large --language=English --device=cuda --model_dir=./cache/ --temperature=0.5 --best_of=10

pip_install:
  pip install -r requirements.txt