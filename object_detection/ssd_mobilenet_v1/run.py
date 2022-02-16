import argparse
from utils.coco import COCODataset
from utils.onnx import OnnxRunner
from utils.benchmark import run_model

def parse_args():
    parser = argparse.ArgumentParser(description="Run SSD mobilenet v1 model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
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
    parser.add_argument("--anno_path",
                        type=str,
                        help="path to file with validation annotations")
    return parser.parse_args()


def run_onnx(model_path, batch_size, num_of_runs, timeout, images_path, anno_path):
    # Note: This ssd_mobilenet_v1 only support batch size of 1

    def run_single_pass(onnx_runner, coco):
        shape = (640, 640)
        onnx_runner.set_input_tensor("image_tensor:0", coco.get_input_array(shape))
        outputs = onnx_runner.run()
        #print("detection_classes shape: {}".format(outputs[0].shape))
        #print("detection_boxes shape: {}".format(outputs[1].shape))
        #print("detection_scores shape: {}".format(outputs[2].shape))
        #print("num_detections shape: {}".format(outputs[3].shape))
        for i in range(batch_size):
          for d in range(int(outputs[3][i])):
             coco.submit_bbox_prediction(
                i,
                coco.convert_bbox_to_coco_order(outputs[1][i][d] * shape[0], 1, 0, 3, 2),
                outputs[2][i][d],
                int(outputs[0][i][d])
             )

    dataset = COCODataset(batch_size, "BGR", "COCO_val2014_000000000000", images_path, anno_path, pre_processing=False, sort_ascending=True, transpose_input=False)
    runner = OnnxRunner(model_path, ["detection_classes:0", "detection_boxes:0", "detection_scores:0", "num_detections:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)

def main():
    args = parse_args()
    run_onnx(args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.anno_path)

if __name__ == "__main__":
    main()
