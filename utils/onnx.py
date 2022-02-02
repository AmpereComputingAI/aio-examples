import os
import torch
import utils.misc as utils
import time
import utils.benchmark as bench_utils
import numpy as np
import sys

try:
    import onnxruntime as ort
except ImportError:
    pass

class OnnxRunner:
    """
    A class providing facilities to run Onnx model (as pretrained torchvision model).
    """

    def __init__(self, model:str, output_names: list):
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = bench_utils.get_intra_op_parallelism_threads()
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.__sess = rt.InferenceSession(model, sess_options=sess_options)
        self.__warm_up_run_latency = 0.0
        self.__total_inference_time = 0.0
        self.__times_invoked = 0
        self.__output_names = output_names

        print("\nRunning with Onnx\n")

    def set_input_tensor(self, input_name: str, input_array):
        """
        A function assigning given numpy input array to the tensor under the provided input name.
        :param input_name: str, name of a input node in a model, eg. "image_tensor:0"
        :param input_array: numpy array with intended input
        """
        self.__feed_dict[input_name] = input_array

    def run(self):
        """
        A function assigning values to input tensor, executing single pass over the network, measuring the time needed
        and finally returning the output.
        :return: dict, output dictionary with tensor names and corresponding output
        """
        start = time.time()
        output = self.__sess.run(self.__output_names, self.__feed_dict)
        finish = time.time()
        self.__total_inference_time += finish - start
        if self.__times_invoked == 0:
            self.__warm_up_run_latency += finish - start
        self.__times_invoked += 1
        return output

    def print_performance_metrics(self, batch_size):
        perf = bench_utils.print_performance_metrics(
            self.__warm_up_run_latency, self.__total_inference_time, self.__times_invoked, batch_size)
        if os.getenv("AIO_PROFILER", "0") == "1":
            torch.AIO.print_profile_data()
        return perf
