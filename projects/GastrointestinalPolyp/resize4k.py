# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2023/06/12 antillia.com


import os
import glob
from PIL import Image
import shutil
import traceback

def resize_to_4kimage(input_dir, output_dir, image_format=".jpg"):
  files  = glob.glob(input_dir + "/*.jpg")
  files += glob.glob(input_dir + "/*.png")
  files += glob.glob(input_dir + "/*.tif")
  files += glob.glob(input_dir + "/*.bmp")
  WIDTH  = 4096
  HEIGHT = 2160

  for file in files:
     basename = os.path.basename(file)
     name     = basename.split(".")[0]
     image = Image.open(file)
     w, h  = image.size
     ratio = WIDTH/w
     HEIGHT = int(h * ratio)

     image_4k = image.resize((WIDTH, HEIGHT))
     output_file = name + image_format
     output_filepath = os.path.join(output_dir, output_file)
     if image_format == ".jpg":
       image_4k.save(output_filepath, quality=95)
     else:
       image_4k.save(output_filepath)

     print("Saved {} as WIDTH {} HEIGHT {}".format(output_filepath, WIDTH, HEIGHT))

if __name__ == "__main__":
  try:
    input_dir = "./mini_test"
    output_dir = "./4k_mini_test"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    resize_to_4kimage(input_dir, output_dir)

  except:
    traceback.print_exc()


