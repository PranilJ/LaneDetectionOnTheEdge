
"""A demo which runs lane detection and streams video to the browser.

export MODEL_PATH=/home/mendel/lanenet-lane-detection


Run lanenet model:
python3 -m edgetpuvision.segment_server \
  --model ${MODEL_PATH}/saved_model_quantized_edgetpu.tflite

"""

from .apps import run_server
from .segment import add_render_gen_args, render_gen

def main():
    run_server(add_render_gen_args, render_gen)

if __name__ == '__main__':
    main()
