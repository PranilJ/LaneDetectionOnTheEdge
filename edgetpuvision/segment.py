import argparse
import colorsys
import itertools
import time
import numpy as np
from pycoral.adapters import common
from pycoral.utils import edgetpu

from . import svg
from . import utils
from .apps import run_app

CSS_STYLES = str(svg.CssStyle({'.back': svg.Style(fill='black',
                                                  stroke='black',
                                                  stroke_width='0.5em'),
                               '.bbox': svg.Style(fill_opacity=100.0,
                                                  stroke_width='0.1em',
                                                  fill='white')}))
def size_em(length):
    return '%sem' % str(0.6 * (length + 1))

def overlay(title, output, inference_time, inference_rate, layout):
    x0, y0, width, height = layout.window
    font_size = 0.05 * height
    # print(50*"#")
    # print('In render_gen overlay')
    # print(f'x0: {x0}')
    # print(f'y0: {y0}')
    # print(f'width: {width}')
    # print(f'height: {height}')

    defs = svg.Defs()
    defs += CSS_STYLES

    doc = svg.Svg(width=width, height=height,
                  viewBox='%s %s %s %s' % layout.window,
                  font_size=font_size, font_family='monospace', font_weight=500)
    doc += defs
    # test_output = np.load('/home/mendel/lanenet-lane-detection/img0003_bin.npy')
    # rows_test, cols_test = np.where(test_output == 1)
    rows, cols = np.where(output == 1)
    # print(f'Rows.shape: {rows.shape}')
    # print(f'Cols.shape: {cols.shape}')
    # doc += svg.Circle(cx=x0, cy=y0, r=100, style='stroke:%s' % svg.rgb((255, 255, 255)), _class='bbox')
    inference_width, inference_height = layout.inference_size
    img_height, img_width = output.shape

    print(f'inference_width: {inference_width}')
    print(f'inference_height: {inference_height}')
    print(f'img_width: {img_width}')
    print(f'img_height: {img_height}')
    for i in range(rows.shape[0]):
        doc += svg.Circle(cx=x0+cols[i] * (inference_width/img_width), cy=y0+rows[i] * (inference_height/img_height), r=5, style='stroke:%s' % svg.rgb((255, 255, 255)), _class='bbox')

    ox = x0 + 2
    oy1 = y0 + font_size #+ 20
    # oy2 = y0 + height - 5
    # # Title
    # if title:
    #     doc += svg.Rect(x=0, y=0, width=size_em(len(title)), height='1em',
    #                     transform='translate(%s, %s) scale(1,-1)' % (ox, oy1), _class='back')
    #     doc += svg.Text(title, x=ox, y=oy1, fill='white')

    # Info
    lines = [
        'Inference time: %.2f ms (%.2f fps)' % (inference_time * 1000, 1.0 / inference_time)
    ]

    for i, line in enumerate(reversed(lines)):
        y = oy1 - i * 1.7 * font_size
        doc += svg.Rect(x=0, y=0, width=size_em(len(line)), height='1em',
                       transform='translate(%s, %s) scale(1,-1)' % (ox, y), _class='back')
        doc += svg.Text(line, x=ox, y=y, fill='white')

    return str(doc)

def print_results(inference_rate, output):
    print('\nInference (rate=%.2f fps):' % inference_rate)
    plt.imshow(output[0] * 255, cmap='gray')
    plt.show()

def render_gen(args):
    fps_counter  = utils.avg_fps_counter(30)

    interpreters, titles = utils.make_interpreters(args.model)
    assert utils.same_input_image_sizes(interpreters)
    interpreters = itertools.cycle(interpreters)
    interpreter = next(interpreters)


    draw_overlay = True

    width, height = utils.input_image_size(interpreter)
    yield width, height

    output = None
    while True:
        tensor, layout, command = (yield output)

        inference_rate = next(fps_counter)
        if draw_overlay:
            start = time.monotonic()
            edgetpu.run_inference(interpreter, tensor)
            inference_time = time.monotonic() - start
            # print('\nInference (rate=%.2f fps):' % inference_rate)
            # print(50*'#')
            binary_seg_image = common.output_tensor(interpreter, 0)
            # print(f'Shape: {binary_seg_image[0].shape}')
            # print(50*'#')
            if args.print:
                print_results(inference_rate, output)

            title = titles[interpreter]
            output = overlay(title, binary_seg_image[0], inference_time, inference_rate, layout)
        else:
            print("NO DRAW OVERLAY")
            output = None

        if command == 'o':
            draw_overlay = not draw_overlay
        elif command == 'n':
            interpreter = next(interpreters)

def add_render_gen_args(parser):
    parser.add_argument('--model',
                        help='.tflite model path', required=True)
    # parser.add_argument('--color', default=None,
    #                     help='Lane display color'),
    parser.add_argument('--print', default=False, action='store_true',
                        help='Print inference results')

def main():
    run_app(add_render_gen_args, render_gen)

if __name__ == '__main__':
    main()