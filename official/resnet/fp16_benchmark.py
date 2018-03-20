# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""
Hacky code used to run imagenet a bunch of times to get runtime info.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as it
from timeit import default_timer
import os
import shutil
import sys

from official.resnet.imagenet_main import main
import tensorflow as tf

class RecordKeeper:
  def __init__(self):
    self.records = []
    self.st = None

  def new_record(self, summary):
    self.records.append([summary, []])

  def start(self):
    self.st = default_timer()

  def end(self):
    run_time = default_timer() - self.st
    self.records[-1][1].append(run_time)

  def print(self):
    for summary, [train_time, eval_time] in self.records:
      print("{}  {:.2f}  {:.2f}".format(summary, train_time, eval_time))



def bm_main():
  temp_dir = "/tmp/imagenet"

  steps = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
  # steps = [1, 2]
  batch_per_gpu = [32, 64]
  use_fp16 = [True, False]
  multi_gpu = [True, False]
  # multi_gpu = [False]

  records = RecordKeeper()

  for step, batch, fp16, multi in it.product(*[steps, batch_per_gpu, use_fp16, multi_gpu]):
    if os.path.exists(temp_dir):
      shutil.rmtree(temp_dir)

    n_gpu = 8 if multi else 1
    bs = batch * n_gpu
    args = [sys.argv[0], "-md", temp_dir, "-synth", "-te", "1", "-mts", str(step),
            "-bs", str(bs), "-hk", ""]
    if fp16:
      args.append("-fp16")
    if multi:
      args.append("--multi_gpu")

    summary = "{}  {}  {}     {}".format(str(step).ljust(6), str(bs).ljust(6), "fp16" if fp16 else "fp32", n_gpu)
    print(summary)
    records.new_record(summary)
    main(args, records)
    records.print()





if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  bm_main()
