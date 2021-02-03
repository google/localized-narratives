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
"""Demo usage of the Localized Narratives data loader."""
import localized_narratives

# This folder is where you would like to download the annotation files to and
# where to read them from.
local_dir = '/usr/local/google/home/jponttuset/datasets/LocalizedNarratives'

# The DataLoader class allows us to download the data and read it from file.
data_loader = localized_narratives.DataLoader(local_dir)

# Downloads the annotation files (it first checks if they are not downloaded).
data_loader.download_annotations('flickr30k_test')

# Iterates through all or a limited number of (e.g. 1 in this case) annotations
# for all files found in the local folder for a given dataset and split. E.g.
# for `open_images_train` it will read only one shard if only one file was
# downloaded manually.
loc_narr = next(data_loader.load_annotations('flickr30k_test', 1))

print(f'\nLocalized Narrative sample:\n{loc_narr}')

print(f'\nVoice recording URL:\n {loc_narr.voice_recording_url}\n')
