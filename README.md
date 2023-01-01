# Simple toolbox for transcription using Whisper

This script gathers the code presented in [this tutorial](https://www.youtube.com/watch?v=MVW746z8y_I) to transcribe audio/video interviews with speaker identififaction (SPEAKER 1: ...). Credits to [Dwarkesh Patel](https://twitter.com/dwarkesh_sp/status/1579672641887408129).

## Usage

Installation:

```bash
# install ffmpeg
brew install ffmpeg
# install the python dependencies
pip install -r requirements.txt
```

Transcribe a .wav file:
```bash
# run whisper/brainspeech to transcribe all the .wav files (see args in `conf/config.yaml`)
python run.py device=cuda input_file=interwiev.wav model=tiny
# or multiple files using
python run.py -m device=cuda input_file=file1.wav,file2.wav
```

## Utilities

A few utility scripts are gathered in the `justfile`. Usage:

```bash
# (optional) install `just` : a simple command line runner: https://github.com/casey/just
brew install just
# (optional) convert the videos into .wav files
just convert
# (optional) push/pull your repo to a remote server
just upload
just download
```
