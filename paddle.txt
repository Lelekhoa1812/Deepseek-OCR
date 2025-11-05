import atexit
import functools
from queue import Queue
from threading import Event, Thread

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import gradio as gr


LANG_CONFIG = {
    "ch": {"num_workers": 2},
    "en": {"num_workers": 2},
    "fr": {"num_workers": 1},
    "german": {"num_workers": 1},
    "korean": {"num_workers": 1},
    "japan": {"num_workers": 1},
}
CONCURRENCY_LIMIT = 8


class PaddleOCRModelManager(object):
    def __init__(self,
                 num_workers,
                 model_factory):
        super().__init__()
        self._model_factory = model_factory
        self._queue = Queue()
        self._workers = []
        self._model_initialized_event = Event()
        for _ in range(num_workers):
            worker = Thread(target=self._worker, daemon=False)
            worker.start()
            self._model_initialized_event.wait()
            self._model_initialized_event.clear()
            self._workers.append(worker)

    def infer(self, *args, **kwargs):
        # XXX: Should I use a more lightweight data structure, say, a future?
        result_queue = Queue(maxsize=1)
        self._queue.put((args, kwargs, result_queue))
        success, payload = result_queue.get()
        if success:
            return payload
        else:
            raise payload

    def close(self):
        for _ in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join()

    def _worker(self):
        model = self._model_factory()
        self._model_initialized_event.set()
        while True:
            item = self._queue.get()
            if item is None:
                break
            args, kwargs, result_queue = item
            try:
                result = model.ocr(*args, **kwargs)
                result_queue.put((True, result))
            except Exception as e:
                result_queue.put((False, e))
            finally:
                self._queue.task_done()


def create_model(lang):
    return PaddleOCR(lang=lang, use_angle_cls=True, use_gpu=False)


model_managers = {}
for lang, config in LANG_CONFIG.items():
    model_manager = PaddleOCRModelManager(config["num_workers"], functools.partial(create_model, lang=lang))
    model_managers[lang] = model_manager


def close_model_managers():
    for manager in model_managers.values():
        manager.close()


# XXX: Not sure if gradio allows adding custom teardown logic
atexit.register(close_model_managers)


def inference(img, lang):
    ocr = model_managers[lang]
    result = ocr.infer(img, cls=True)[0]
    img_path = img
    image = Image.open(img_path).convert("RGB")
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores,
                    font_path="./simfang.ttf")
    return im_show


title = 'PaddleOCR'
description = '''
- Gradio demo for PaddleOCR. PaddleOCR demo supports Chinese, English, French, German, Korean and Japanese. 
- To use it, simply upload your image and choose a language from the dropdown menu, or click one of the examples to load them. Read more at the links below.
- [Docs](https://paddlepaddle.github.io/PaddleOCR/), [Github Repository](https://github.com/PaddlePaddle/PaddleOCR).
'''

examples = [
    ['en_example.jpg','en'],
    ['cn_example.jpg','ch'],
    ['jp_example.jpg','japan'],
]

css = ".output_image, .input_image {height: 40rem !important; width: 100% !important;}"
gr.Interface(
    inference,
    [
        gr.Image(type='filepath', label='Input'),
        gr.Dropdown(choices=list(LANG_CONFIG.keys()), value='en', label='language')
    ],
    gr.Image(type='pil', label='Output'),
    title=title,
    description=description,
    examples=examples,
    cache_examples=False,
    css=css,
    concurrency_limit=CONCURRENCY_LIMIT,
    ).launch(debug=False)
