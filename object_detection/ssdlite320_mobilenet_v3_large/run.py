import os
import time
import argparse
import warnings
import torchvision

from utils.cv.coco import COCODataset
from utils.pytorch import PyTorchRunner
from utils.benchmark import run_model

from utils.misc import UnsupportedPrecisionValueError, FrameworkUnsupportedError
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SSD VGG-16 model.")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--images_path",
                        type=str,
                        help="path to directory with COCO validation images")
    parser.add_argument("--anno_path",
                        type=str,
                        help="path to file with validation annotations")
    return parser.parse_args()


def run_pytorch_fp(batch_size, num_of_runs, timeout, images_path, anno_path):
    def run_single_pass(pytorch_runner, coco):
        shape = (320, 320)
        output = pytorch_runner.run(coco.get_input_array(shape))
        output = output[1]

        for i in range(batch_size):
            for d in range(output[i]['boxes'].shape[0]):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(output[i]['boxes'][d].tolist()),
                    output[i]['scores'][d].item(),
                    output[i]['labels'][d].item()
                )

    dataset = COCODataset(batch_size, "BGR", "COCO_val2014_000000000000", images_path, anno_path,
                          pre_processing="PyTorch_objdet", sort_ascending=True, order="NCHW")
    runner = PyTorchRunner(torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True))

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def run_pytorch_fp32(batch_size, num_of_runs, timeout, images_path, anno_path):
    return run_pytorch_fp(batch_size, num_of_runs, timeout, images_path, anno_path)


def main():
    args = parse_args()

    if args.precision == "fp32":
        run_pytorch_fp32(
            args.batch_size, args.num_runs, args.timeout, args.images_path, args.anno_path
        )
    else:
        raise UnsupportedPrecisionValueError(args.precision)

if __name__ == "__main__":
    main()
