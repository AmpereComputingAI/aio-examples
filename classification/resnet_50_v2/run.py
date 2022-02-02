import argparse
from utils.imagenet import ImageNet
from utils.onnx import OnnxRunner
from utils.benchmark import run_model

def parse_args():
    parser = argparse.ArgumentParser(description="Run ResNet-50 v1.5 model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "fp16", "int8"], required=True,
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
                        help="path to directory with ImageNet validation images")
    parser.add_argument("--labels_path",
                        type=str,
                        help="path to file with validation labels")
    return parser.parse_args()


def run_onnx_fp32(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):

    def run_single_pass(onnx_runner, imagenet):
        shape = (224, 224)
        onnx_runner.set_input_tensor("data", imagenet.get_input_array(shape))
        output = onnx_runner.run()
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output["resnetv24_dense0_fwd"][i]),
                imagenet.extract_top5(output["resnetv24_dense0_fwd"][i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing="VGG", is1001classes=True)
    runner = OnnxRunner(model_path, ["resnetv24_dense0_fwd"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def run_onnx_fp16(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):

    def run_single_pass(onnx_runner, imagenet):
        shape = (224, 224)
        onnx_runner.set_input_tensor("data", imagenet.get_input_array(shape))
        output = onnx_runner.run()
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output["resnetv24_dense0_fwd"][i]),
                imagenet.extract_top5(output["resnetv24_dense0_fwd"][i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing="VGG", is1001classes=True)
    runner = OnnxRunner(model_path, ["resnetv24_dense0_fwd"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)

def main():
    args = parse_args()
    if args.precision == "fp32":
        run_onnx_fp32(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
        )
    elif args.precision == "fp16":
        run_onnx_fp16(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
        )
    else:
        assert False


if __name__ == "__main__":
    main()
