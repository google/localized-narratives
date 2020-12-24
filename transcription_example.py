# python3
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example call to Google's speech-to-text API to transcribe Localized Narrative recordings.

Pre-requisites:
- Set up Google's API authentication:
https://cloud.google.com/docs/authentication/getting-started
- Install dependencies:
  + pip install ffmpeg
  + pip install pydub
  + pip install google-cloud-speech

Comments:
- Google's speech-to-text API does not support the Vorbis encoding in which the
Localized Narrative recordings were released. We therefore need to transcode
them Opus, which is supported. We do this in`convert_recording`.
- Transcription is limited to 60 seconds if loaded from a local file. For audio
longer than 1 minute, we need to upload the file to a GCS bucket and load the
audio using its URI: `audio = speech.RecognitionAudio(uri=gcs_uri)`.
"""
import io
import os

from google.cloud import speech
import pydub


def convert_recording(input_file, output_file):
  with open(input_file, 'rb') as f:
    recording = pydub.AudioSegment.from_file(f, codec='libvorbis')

  with open(output_file, 'wb') as f:
    recording.export(f, format='ogg', codec='libopus')


def speech_to_text(recording_file):
  # Loads from local file. If longer than 60 seconds, upload to GCS and use
  # `audio = speech.RecognitionAudio(uri=gcs_uri)`
  with io.open(recording_file, 'rb') as audio_file:
    content = audio_file.read()
  audio = speech.RecognitionAudio(content=content)

  config = speech.RecognitionConfig(
      encoding=speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
      sample_rate_hertz=48000,
      audio_channel_count=2,
      max_alternatives=10,
      enable_word_time_offsets=True,
      language_code='en-IN')

  client = speech.SpeechClient()
  operation = client.long_running_recognize(config=config, audio=audio)
  return operation.result(timeout=90)


if __name__ == '__main__':

  # Input encoded in Vorbis in an OGG container.
  input_recording = '/Users/jponttuset/Downloads/coco_val_137576_93.ogg'
  basename, extension = os.path.splitext(input_recording)
  output_recording = f'{basename}_opus{extension}'

  # Re-encodes in Opus and saves to file.
  convert_recording(input_recording, output_recording)

  # Actual call to Google's speech-to-text API.
  result = speech_to_text(output_recording)
  print(result)
