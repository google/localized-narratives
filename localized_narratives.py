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
"""Data Loader for Localized Narratives."""

import json
import os
import re
from typing import Dict, Generator, List, NamedTuple
import wget  # type: ignore


_ROOT_URL = 'https://storage.googleapis.com/localized-narratives'
_ANNOTATIONS_ROOT_URL = f'{_ROOT_URL}/annotations'
_RECORDINGS_ROOT_URL = f'{_ROOT_URL}/voice-recordings'

_ANNOTATION_FILES = {
    'open_images_train': [
        f'open_images_train_v6_localized_narratives-{i:05d}-of-00010.jsonl'
        for i in range(10)
    ],
    'open_images_val': ['open_images_validation_localized_narratives.jsonl'],
    'open_images_test': ['open_images_test_localized_narratives.jsonl'],
    'coco_train': [
        f'coco_train_localized_narratives-{i:05d}-of-00004.jsonl'
        for i in range(4)
    ],
    'coco_val': ['coco_val_localized_narratives.jsonl'],
    'flickr30k_train': ['flickr30k_train_localized_narratives.jsonl'],
    'flickr30k_val': ['flickr30k_val_localized_narratives.jsonl'],
    'flickr30k_test': ['flickr30k_test_localized_narratives.jsonl'],
    'ade20k_train': ['ade20k_train_localized_narratives.jsonl'],
    'ade20k_val': ['ade20k_validation_localized_narratives.jsonl']
}  # type: Dict[str, List[str, ...]]]


class TimedPoint(NamedTuple):
  x: float
  y: float
  t: float


class TimedUtterance(NamedTuple):
  utterance: str
  start_time: float
  end_time: float


class LocalizedNarrative(NamedTuple):
  """Represents a Localized Narrative annotation.

  Visit https://google.github.io/localized-narratives/index.html?file-formats=1
  for the documentation of each field.
  """
  dataset_id: str
  image_id: str
  annotator_id: int
  caption: str
  timed_caption: List[TimedUtterance]
  traces: List[List[TimedPoint]]
  voice_recording: str

  @property
  def voice_recording_url(self) -> str:
    """Returns the absolute path where to find the voice recording file."""
    # Fixes the voice recording path for Flickr30K and ADE20k
    if 'Flic' in self.dataset_id or 'ADE' in self.dataset_id:
      split_id, image_id = re.search(r'(\w+)/\w+_([0-9]+)_[0-9]+\.',
                                     self.voice_recording).groups()
      image_id = image_id.zfill(16)
      voice_recording = (f'{split_id}/'
                         f'{split_id}_{image_id}_{self.annotator_id}.ogg')
    else:
      voice_recording = self.voice_recording

    return f'{_RECORDINGS_ROOT_URL}/{voice_recording}'

  def __repr__(self):
    truncated_caption = self.caption[:60] + '...' if len(
        self.caption) > 63 else self.caption
    truncated_timed_caption = self.timed_caption[0].__str__()
    truncated_traces = self.traces[0][0].__str__()
    return (f'{{\n'
            f' dataset_id: {self.dataset_id},\n'
            f' image_id: {self.image_id},\n'
            f' annotator_id: {self.annotator_id},\n'
            f' caption: {truncated_caption},\n'
            f' timed_caption: [{truncated_timed_caption}, ...],\n'
            f' traces: [[{truncated_traces}, ...], ...],\n'
            f' voice_recording: {self.voice_recording}\n'
            f'}}')


def _expected_files(dataset_and_split: str) -> Generator[str, None, None]:
  try:
    yield from _ANNOTATION_FILES[dataset_and_split]
  except KeyError:
    raise ValueError(
        f'Unknown value for `dataset_and_split`: {dataset_and_split}')


class DataLoader:
  """Data Loader for Localized Narratives."""

  def __init__(self, local_root_dir: str):
    """DataLoader constructor.

    Args:
      local_root_dir: Local directory where the annotation files can be
        downloaded to and read from.
    """
    self._local_root_dir = local_root_dir
    self._current_open_file = None

  def download_annotations(self, dataset_and_split: str):
    """Downloads the Localized Narratives annotations.

    Args:
      dataset_and_split: Name of the dataset and split to download.
        Possible values are the keys in _ANNOTATION_FILES.
    """
    os.makedirs(self._local_root_dir, exist_ok=True)

    for filename in _expected_files(dataset_and_split):
      self._download_one_file(filename)

  def load_annotations(
      self, dataset_and_split: str, max_num_annotations: int = int(1e30)
  ) -> Generator[LocalizedNarrative, None, None]:
    """Loads the Localized Narratives annotations from local files.

    Args:
      dataset_and_split: Name of the dataset and split to load. Possible values
        are the keys in _ANNOTATION_FILES.
      max_num_annotations: Maximum number of annotations to load.

    Yields:
      One Localized Narrative at a time.
    """
    num_loaded = 0
    for local_file in self._find_files(dataset_and_split):
      self._current_open_file = open(local_file, 'rb')
      for line in self._current_open_file:
        yield LocalizedNarrative(**json.loads(line))
        num_loaded += 1
        if num_loaded == max_num_annotations:
          self._current_open_file.close()
          return
      self._current_open_file.close()

  def _local_file(self, filename: str) -> str:
    return os.path.join(self._local_root_dir, filename)

  def _find_files(self, dataset_and_split: str) -> Generator[str, None, None]:
    for filename in _expected_files(dataset_and_split):
      if os.path.exists(self._local_file(filename)):
        yield self._local_file(filename)

  def _download_one_file(self, filename: str):
    if not os.path.exists(self._local_file(filename)):
      print(f'Downloading: {filename}')
      wget.download(f'{_ANNOTATIONS_ROOT_URL}/{filename}',
                    self._local_file(filename))
      print()
    else:
      print(f'Already downloaded: {filename}')
