import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import whisper
import datetime
import rich
import subprocess

import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

from pyannote.audio import Audio
from pyannote.core import Segment

import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np

import logging

speechbrain_logger = logging.getLogger("speechbrain")
speechbrain_logger.setLevel(logging.ERROR)

from loguru import logger


@torch.inference_mode()
@hydra.main(version_base=None, config_path="conf", config_name="config")
def transcribe(cfg: DictConfig) -> None:
    """
    Copied from: https://colab.research.google.com/drive/1V-Bt5Hm2kjaDb4P1RyMSswsDKyrzc2-3?usp=sharing
    """
    print(OmegaConf.to_yaml(cfg))

    # Load the audio embedding model
    embedding_model = PretrainedSpeakerEmbedding(
        cfg.embedding_model,
        device=torch.device(cfg.device),
    )

    # load Whisper model
    model = whisper.load_model(
        cfg.model,
        device=torch.device(cfg.device),
        download_root=cfg.download_root,
    )

    # transcribe the file
    input_file = Path(cfg.input_file)
    logger.info(f"Transcribing {input_file}")
    transcription = model.transcribe(str(input_file), **cfg.decode_options)
    segments = transcription["segments"]

    # compute the embedding for each segment
    audio = Audio()
    rate, duration = get_rate_an_duration(str(input_file))
    params = {
        "rate": rate,
        "duration": duration,
        "audio": audio,
        "embedding_model": embedding_model,
        "input_file": str(input_file),
    }
    embeddings = []
    for i, segment in enumerate(segments):
        embd = segment_embedding(segment, **params)
        embeddings.append(embd)
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = np.nan_to_num(embeddings)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # cluster the embeddings
    clustering = AgglomerativeClustering(cfg.num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = f"SPEAKER {labels[i] + 1}"

    # save the transcription
    output_file = input_file.with_suffix(".txt")
    logger.info(f"Saving transcription to {output_file}")
    with open(output_file, "w") as f:
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write(
                    "\n" + segment["speaker"] + " " + str(time(segment["start"])) + "\n"
                )
            f.write(segment["text"][1:] + " ")


def time(secs):
    return datetime.timedelta(seconds=round(secs))


def get_rate_an_duration(input_file):
    with wave.open(input_file, "r") as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return rate, duration


def segment_embedding(
    segment,
    *,
    audio: Audio,
    rate: float,
    duration: float,
    embedding_model: PretrainedSpeakerEmbedding,
    input_file: Path,
):
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(input_file, clip)
    return embedding_model(waveform[None])


if __name__ == "__main__":
    transcribe()
