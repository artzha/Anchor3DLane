import argparse
import os
import os.path as osp
import shutil
import time
import warnings
import sys
import yaml
sys.path.append(os.getcwd())
import numpy as np
import cv2
from helpers.data import create_data
from helpers.visualization import extract_line
from queue import Queue

# PyTorch
import torch
from torch.nn.functional import interpolate
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

# mmcv
import mmcv
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

# mmseg
from mmseg import digit_version
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_lanedetector
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes
from mmseg.datasets.tools.vis_openlane import LaneVis

# ROS related imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# For import
class ImageQueue(object):
    def __init__(self, img_pre_topic, img_post_topic):
        self._img_pre_topic = img_pre_topic
        self._img_post_topic = img_post_topic

        rospy.init_node('img_queue')
        
        self._img_pre_queue = Queue(maxsize=20)
        self._img_post_queue = Queue(maxsize=20)

        self._img_sub = rospy.Subscriber(self._img_pre_topic, Image, self.img_callback, queue_size=20)
        self._img_pub = rospy.Publisher(self._img_post_topic, Image, queue_size=20)

        self.bridge = CvBridge()
        
        self.img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        self.input_size = (360, 480)

        self.transform_pre = transforms.Compose([
                            transforms.ToTensor(),
                            # resize to match Anchor3DLane
                            # transforms.Resize((360, 480), interpolation=InterpolationMode.BICUBIC),
                            transforms.Resize((360, 480), interpolation=InterpolationMode.BILINEAR),
                            transforms.Normalize(**self.img_norm_cfg)
                         ])

        self.transform_post = transforms.Compose([
                            # transforms.Resize(self.input_size, interpolation=InterpolationMode.BICUBIC),
                            transforms.Resize(self.input_size, interpolation=InterpolationMode.BILINEAR),
                            transforms.ToPILImage(),
                         ])

    def pre_process(self):
        imgmsg = self._img_pre_queue.get()
        # print(f'Img seq {imgmsg.header.seq}')
        img_pre = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding="passthrough") # OpenCV (h, w, nc)
        img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
        img_pre_after = self.transform_pre(img_pre) # PyTorch expects (B, nc, w, h)
        return img_pre_after

    def img_callback(self, imgmsg):
        try:
            self._img_pre_queue.put(imgmsg)
            # print(f'Img seq {imgmsg.header.seq}')
        except CvBridgeError as e:
            print(e)

    def get(self):
        return self.pre_process()

    def put(self, img_post):
        self._img_post_queue.put(img_post)
        self.publish() # --> make it paralle?
    
    def publish(self):
        img_post = self._img_post_queue.get()
        img_post = self.transform_post(img_post)
        cv_img_bgr = cv2.cvtColor(img_post, cv2.COLOR_RGB2BGR)
        ros_img = self.bridge.cv2_to_imgmsg(cv_img_bgr, "passthrough")
        ros_img.header = Header()
        ros_img.header.stamp = rospy.Time.now()
        ros_img.header.seq = fidx
        img_pub.publish(ros_img)
        self._img_pub.publish(ros_img)

    def empty(self):
        return self._img_pre_queue.empty()


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        default=False,
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--eval', action='store_true', help='show results', default=True)
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def postprocess(output, visualizer):
    line = extract_line(output)
    import pdb; pdb.set_trace()
    # TODO — modify visualize func to add lines on top of input image
    P_GND = visualizer.visualize(line)
    img_post = transform_gnd_to_gm(P_GND)

    return img_post

def load_proj_mat(yaml_path):
    with open(yaml_path) as stream:
        try:
            project_matrix = yaml.safe_load(stream)['extrinsic_matrix']['data'][:12]
            project_matrix = np.asarray(project_matrix).reshape(1, 1, 3, 4).tolist()
        except yaml.YAMLError as exc:
            print(exc)
    return project_matrix

def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    assert not distributed, "Only single GPU inference supported while in inference mode"

    rank, _ = get_dist_info()

    dataset = build_dataset(cfg.data.test)
    visualizer = LaneVis(dataset)

    # build the model and load checkpoint
    model = build_lanedetector(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    mmcv.mkdir_or_exist(args.show_dir)
    
    # To visualize results
    args.show = True

    cfg.device = get_device()
    if not distributed:
        # warnings.warn(
        #     'SyncBN is only supported with DDP. To be compatible with DP, '
        #     'we convert SyncBN to BN. Please use dist_train.sh which can '
        #     'avoid this error.')
        # if not torch.cuda.is_available():
        #     assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
        #         'Please use MMCV >= 1.4.4 for CPU training!'
        model = revert_sync_batchnorm(model)
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)


        #0 Initialize ROS subscribers and publishers
        img_pre_topic = "img_pre"
        img_post_topic = "img_post"
        img_queue = ImageQueue(img_pre_topic, img_post_topic)
        img_pre = None
        yaml_path = '/robodata/ecocar_logs/processed/CACCDataset/calibrations/44/calib_os1_to_cam0.yaml'
        project_matrix = load_proj_mat(yaml_path)                
        results = []
        
        rospy.loginfo("Waiting for raw images ...")
        r = rospy.Rate(10)
        while not rospy.is_shutdown():          
            #1 Check if images are in ROS Image Queue
            if not img_queue.empty():
                #2 Take last image in FIFO queue
                img_pre = img_queue.get()

            if img_pre is not None:
                #3 Construct data dictionary by loading static 3x4 projection matrix, image, and image metadata
                data = create_data(img_pre, project_matrix)

                #4 Perform Model Inference
                output = model(return_loss=False, **data)

                # 5 Convert lane detections to GM Format
                # result = postprocess(output, anchor_len=20)
                img_post = postprocess(output, visualizer)

                # 6 Publish lane detections over ROS
                img_queue.put(img_post)

                #7 (Optional) Publish BEV image with lane detection plotted on it

                rospy.loginfo("\n Finished processing inference ...")
                img_pre = None

            r.sleep()


if __name__ == '__main__':
    main()
    