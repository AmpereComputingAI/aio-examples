# densenet_169
wget -O classification/densenet_169/densenet_169_tf_fp32.pb https://ampereaimodelzoo.s3.amazonaws.com/densenet_169_tf_fp32.pb
wget -O classification/densenet_169/densenet_169_tf_fp16.pb https://ampereaimodelzoo.s3.amazonaws.com/densenet_169_tf_fp16.pb
wget -O classification/densenet_169/densenet_169_tflite_int8.tflite https://ampereaimodelzoo.s3.amazonaws.com/densenet_169_tflite_int8.tflite

# resnet_50_v1.5
wget -O classification/resnet_50_v15/resnet_50_v15_tf_fp32.pb https://ampereaimodelzoo.s3.amazonaws.com/resnet_50_v15_tf_fp32.pb
wget -O classification/resnet_50_v15/resnet_50_v15_tf_fp16.pb https://ampereaimodelzoo.s3.amazonaws.com/resnet_50_v15_tf_fp16.pb
wget -O classification/resnet_50_v15/resnet_50_v15_tflite_int8.tflite https://ampereaimodelzoo.s3.amazonaws.com/resnet_50_v15_tflite_int8.tflite

# mobilenet_v2
wget -O classification/mobilenet_v2/mobilenet_v2_tf_fp32.pb https://ampereaimodelzoo.s3.amazonaws.com/mobilenet_v2_tf_fp32.pb
wget -O classification/mobilenet_v2/mobilenet_v2_tf_fp16.pb https://ampereaimodelzoo.s3.amazonaws.com/mobilenet_v2_tf_fp16.pb
wget -O classification/mobilenet_v2/mobilenet_v2_tflite_int8.tflite https://ampereaimodelzoo.s3.amazonaws.com/mobilenet_v2_tflite_int8.tflite

# ssd_mobilenet_v2
wget -O object_detection/ssd_mobilenet_v2/ssd_mobilenet_v2_tf_fp32.pb https://ampereaimodelzoo.s3.amazonaws.com/ssd_mobilenet_v2_tf_fp32.pb
wget -O object_detection/ssd_mobilenet_v2/ssd_mobilenet_v2_tflite_int8.tflite https://ampereaimodelzoo.s3.amazonaws.com/ssd_mobilenet_v2_tflite_int8.tflite

# ssd_inception_v2
wget -O object_detection/ssd_inception_v2/ssd_inception_v2_tf_fp32.pb https://ampereaimodelzoo.s3.amazonaws.com/ssd_inception_v2_tf_fp32.pb
wget -O object_detection/ssd_inception_v2/ssd_inception_v2_tf_fp16.pb https://ampereaimodelzoo.s3.amazonaws.com/ssd_inception_v2_tf_fp16.pb

# yolo_v4_tiny
wget -O object_detection/yolo_v4_tiny/yolo_v4_tiny_tf_fp32.tar.gz https://ampereaimodelzoo.s3.amazonaws.com/yolo_v4_tiny_tf_fp32.tar.gz
tar -xvf object_detection/yolo_v4_tiny/yolo_v4_tiny_tf_fp32.tar.gz -C object_detection/yolo_v4_tiny/
rm object_detection/yolo_v4_tiny/yolo_v4_tiny_tf_fp32.tar.gz
