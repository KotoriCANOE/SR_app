import os
import time
import queue
import threading
import numpy as np
import tensorflow as tf

# stderr print
def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)

# recursively list all the files' path under directory
def listdir_files(path, recursive=True, filter_ext=None, encoding=None):
    import os, locale
    if encoding is True: encoding = locale.getpreferredencoding()
    if filter_ext is not None: filter_ext = [e.lower() for e in filter_ext]
    files = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        for f in file_names:
            if not filter_ext or os.path.splitext(f)[1].lower() in filter_ext:
                file_path = os.path.join(dir_path, f)
                try:
                    if encoding: file_path = file_path.encode(encoding)
                    files.append(file_path)
                except UnicodeEncodeError as err:
                    eprint(file_path)
                    eprint(err)
        if not recursive: break
    files.sort()
    return files

# reset random seeds
def reset_random(seed=0):
    import random
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# setup tensorflow and return session
def create_session(graph=None, debug=False, memory_fraction=1.0):
    # create session
    gpu_options = tf.GPUOptions(allow_growth=True,
        per_process_gpu_memory_fraction=memory_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options,
        allow_soft_placement=True, log_device_placement=False)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(graph=graph, config=config)
    if debug:
        from tensorflow.python import debug as tfdbg
        sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    return sess

# API
class SRFilter:
    def __init__(self, data_format='NCHW', scaling=2,
                 sess_threads=1, memory_fraction=1.0, device='GPU:0', random_seed=None):
        # arXiv 1509.09308
        # a new class of fast algorithms for convolutional neural networks using Winograd's minimal filtering algorithms
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        
        self.data_format = data_format
        self.scaling = scaling
        self.memory_fraction = memory_fraction
        self.device = '/device:{}'.format(device)
        self.random_seed = random_seed

        self.semaphore = threading.Semaphore(value=sess_threads)

    def load_model(self, model_file):
        if not isinstance(model_file, str):
            raise TypeError('"model_file" should be the path to a model file')
        elif os.path.isdir(model_file):
            model_file = os.path.join(model_file, 'model.pb')
        if not os.path.exists(model_file):
            raise ValueError('model file not exists: "{}"'.format(model_file))
        # initialize
        if self.random_seed is not None:
            reset_random(self.random_seed)
        # build graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device(self.device):
                # build input
                self.infer_inputs = tf.placeholder(tf.float32, name='InferenceInput')
                inputs = self.infer_inputs
                # load model
                with open(model_file, 'rb') as fd:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(fd.read())
                outputs, = tf.import_graph_def(graph_def, name='',
                    input_map={'Input:0': inputs}, return_elements=['Output:0'])
                # build output
                self.infer_outputs = outputs
        # create session
        self.sess = create_session(self.graph, memory_fraction=self.memory_fraction)

    def inference(self, input):
        feed_dict = {self.infer_inputs: input}
        with self.semaphore:
            output = self.sess.run(self.infer_outputs, feed_dict)
        return output

    def process(self, src, max_patch_height=360, max_patch_width=360, patch_pad=8, patch_mod=8,
        data_format='NHWC', msg=None):
        assert isinstance(src, np.ndarray)
        if msg is True: msg = ''
        # shape standardization
        src_shape = src.shape
        if len(src_shape) == 2:
            src = np.expand_dims(src, 0)
            src = np.expand_dims(src, 0)
            data_format = 'NCHW'
        elif len(src_shape) == 3:
            src = np.expand_dims(src, 0)
        elif len(src_shape) != 4:
            raise ValueError('Not supported rank of \'src\', should be 4, 3 or 2.')
        if data_format != 'NCHW':
            src = src.transpose((0, 3, 1, 2))
        src_channels = src.shape[1]
        if src_channels == 1:
            src = np.concatenate([src, src, src], axis=1)
        elif src_channels != 3:
            raise ValueError('Only 3 channels or 1 channel is supported, but \'src\' has {} channels.\n'
                             'If it\'s an image with alpha channel, '
                             'you should process the alpha channel separately using other resizers.'
                             .format(src_channels))
        # convert to float32
        src_dtype = src.dtype
        if src_dtype != np.float32:
            src = src.astype(np.float32)
            if src_dtype == np.uint8:
                src *= 1 / 255
            elif src_dtype == np.uint16:
                src *= 1 / 65535
        # parameters
        shape = src.shape
        height = shape[2]
        width = shape[3]
        # split into (padded & overlapped) patches
        def pad_split_patch(dim, max_patch, patch_pad=0, patch_mod=1):
            split = 1
            patch = (dim + patch_pad * split * 2 + split - 1) // split
            patch = (patch + patch_mod - 1) // patch_mod * patch_mod
            while patch > max_patch:
                split += 1
                patch = (dim + patch_pad * split * 2 + split - 1) // split
                patch = (patch + patch_mod - 1) // patch_mod * patch_mod
            pad = patch * split - patch_pad * (split - 1) * 2 - dim
            return pad, split, patch
        pad_h, split_h, patch_h = pad_split_patch(height, max_patch_height, patch_pad, patch_mod)
        pad_w, split_w, patch_w = pad_split_patch(width, max_patch_width, patch_pad, patch_mod)
        # padding
        need_padding = pad_h > 0 or pad_w > 0
        pad_h = (patch_pad, pad_h - patch_pad)
        pad_w = (patch_pad, pad_w - patch_pad)
        if need_padding:
            src = np.pad(src, ((0, 0), (0, 0), pad_h, pad_w), mode='reflect')
        # splitting
        splits = split_h * split_w
        src_patches = []
        for s in range(splits):
            p_h = (s // split_w) * (patch_h - patch_pad * 2)
            p_w = (s % split_w) * (patch_w - patch_pad * 2)
            src_patches.append(src[:, :, p_h : p_h + patch_h, p_w : p_w + patch_w])
        # inference
        if msg or msg == '':
            print(msg + 'Inferencing using model...')
            _t = time.time()
        dst_patches = []
        for src_p in src_patches:
            if self.data_format != 'NCHW':
                src_p = src_p.transpose((0, 2, 3, 1))
            dst_p = self.inference(src_p)
            if self.data_format != 'NCHW':
                dst_p = dst_p.transpose((0, 3, 1, 2))
            dst_patches.append(dst_p)
        if msg or msg == '':
            _d = time.time() - _t
            print(msg + 'Inferencing finished. Duration: {} seconds.'.format(_d))
        # cropping
        for s in range(splits):
            crop_t = (pad_h[0] if s // split_w == 0 else patch_pad) * self.scaling
            crop_b = (pad_h[1] if s // split_w == split_h - 1 else patch_pad) * self.scaling
            crop_l = (pad_w[0] if s % split_w == 0 else patch_pad) * self.scaling
            crop_r = (pad_w[1] if s % split_w == split_w - 1 else patch_pad) * self.scaling
            if crop_t > 0 or crop_b > 0 or crop_l > 0 or crop_r > 0:
                crop_t = None if crop_t <= 0 else crop_t
                crop_b = None if crop_b <= 0 else -crop_b
                crop_l = None if crop_l <= 0 else crop_l
                crop_r = None if crop_r <= 0 else -crop_r
                dst_patches[s] = dst_patches[s][:, :, crop_t:crop_b, crop_l:crop_r]
        # stacking (concatenating)
        dst_patches_h = []
        for s_h in range(split_h):
            s = s_h * split_w
            dst_patches_h.append(np.concatenate(dst_patches[s : s + split_w], axis=-1))
        dst = np.concatenate(dst_patches_h, axis=-2)
        # clipping output value
        dst = np.clip(dst, 0, 1)
        # convert to src_type
        if src_dtype != np.float32:
            if src_dtype == np.uint8:
                dst *= 255
            elif src_dtype == np.uint16:
                dst *= 65535
            dst = dst.astype(src_dtype)
        # reshape to src_shape
        if src_channels == 1:
            dst = dst[:, :1, :, :]
        if len(src_shape) == 2:
            dst = np.squeeze(dst, (0, 1))
            data_format = 'NCHW'
        if data_format != 'NCHW':
            dst = dst.transpose((0, 2, 3, 1))
        if len(src_shape) == 3:
            dst = np.squeeze(dst, 0)
        # return
        return dst

# main
def process(args):
    from skimage import io, transform
    
    extensions = ['.jpeg', '.jpg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.jp2']
    dst_postfix = args.dst_postfix + '.png'
    channel_index = -1
    # number of threads
    if args.threads <= 0:
        num_threads = max(1, os.cpu_count() - args.threads)
    else:
        num_threads = args.threads
    # directories and files
    if not os.path.exists(args.dst_dir): os.makedirs(args.dst_dir)
    src_files = listdir_files(args.src_dir, args.recursive, extensions)
    # initialization
    filter = SRFilter(args.data_format, args.scaling,
        args.sess_threads, args.memory_fraction, args.device, args.random_seed)
    filter.load_model(args.model_file)
    # worker - read, process and save image files
    def worker(q, t):
        msg = '{}: '.format(t) if num_threads > 1 else ''
        while True:
            # dequeue
            item = q.get()
            if item is None:
                break
            else:
                file_path = item
                dir_path = os.path.dirname(file_path)
                file_name = os.path.splitext(os.path.basename(file_path))[0]
            # read
            src = io.imread(file_path)
            print(msg + 'Loaded {}'.format(file_path))
            # separate alpha channel
            channels = src.shape[channel_index]
            if channels == 2 or channels == 4:
                alpha = src[:, :, -1:]
                src = src[:, :, :-1]
            # process
            if num_threads == 1:
                tick = time.time()
            dst = filter.process(src, max_patch_height=args.patch_height, max_patch_width=args.patch_width,
                patch_pad=args.patch_pad, patch_mod=args.patch_mod,
                data_format='NHWC', msg=None if num_threads > 1 else True)
            if num_threads == 1:
                tock = time.time()
                print('Process time: {}'.format(tock - tick))
            # process and merge alpha channel
            if channels == 2 or channels == 4:
                dtype = alpha.dtype
                bits = dtype.itemsize * 8
                alpha = transform.rescale(alpha, args.scaling, mode='edge', preserve_range=True)
                if bits < 32:
                    alpha = np.clip(alpha, 0, (1 << bits) - 1)
                    alpha = alpha.astype(dtype)
                else:
                    alpha = np.clip(alpha, 0, 1)
                dst = np.concatenate([dst, alpha], axis=channel_index)
            # save
            save_dir = dir_path[len(args.src_dir):].strip('/').strip('\\')
            save_dir = os.path.join(args.dst_dir, save_dir)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            save_file = os.path.join(save_dir, file_name + dst_postfix)
            if num_threads == 1: print(msg + 'Saving... {}'.format(save_file))
            io.imsave(save_file, dst)
            print(msg + 'Result saved to {}'.format(save_file))
            # indicate enqueued task is complete
            q.task_done()
    # enqueue
    q = queue.Queue()
    for item in src_files:
        q.put(item)
    # start threads
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=lambda: worker(q, _))
        t.start()
        threads.append(t)
    # block until all tasks are done
    q.join()
    # stop workers
    for _ in range(num_threads):
        q.put(None)
    for t in threads:
        t.join()

def main(argv):
    import argparse
    argp = argparse.ArgumentParser(argv[0])
    # IO parameters
    argp.add_argument('--model-file', default='model.pb',
        help="""Path to the model file.""")
    argp.add_argument('--src-dir', default='./',
        help="""Directory where the image files are to be processed.""")
    argp.add_argument('--dst-dir', default='{src_dir}/results',
        help="""Directory where to write the processed images.""")
    argp.add_argument('--dst-postfix', default='.SR',
        help="""Postfix added to the processed filenames.""")
    argp.add_argument('--recursive', default=False, action='store_true',
        help="""Recursively search all the files in 'src_dir'.""")
    argp.add_argument('--threads', type=int, default=0,
        help="""Concurrent multi-threading Python execution.""")
    # model parameters
    argp.add_argument('--scaling', type=int, default=2,
        help="""Up-scaling factor of the model.""")
    argp.add_argument('--data-format', default='NCHW',
        help="""Data layout format.""")
    argp.add_argument('--sess-threads', type=int, default=1,
        help="""Maximum number of concurrent running TensorFlow sessions.""")
    argp.add_argument('--memory-fraction', type=float, default=1.0,
        help="""Maximum allowed fraction of memory to allocate.""")
    argp.add_argument('--device', default='GPU:0',
        help="""Preferred device to use.""")
    argp.add_argument('--random-seed', type=int,
        help="""Initialize with a specific random seed.""")
    # data parameters
    argp.add_argument('--patch-height', type=int, default=512,
        help="""Max patch height.""")
    argp.add_argument('--patch-width', type=int, default=512,
        help="""Max patch width.""")
    argp.add_argument('--patch-pad', type=int, default=8,
        help="""Padding around patches.""")
    argp.add_argument('--patch-mod', type=int, default=8,
        help="""Mod of patches.""")
    # parse
    args = argp.parse_args(argv[1:])
    args.dst_dir = args.dst_dir.format(src_dir=args.src_dir)
    # run
    process(args)

if __name__ == '__main__':
    import sys
    main(sys.argv)
