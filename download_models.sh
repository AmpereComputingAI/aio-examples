# densenet_169
wget -O classification/densenet_169/densenet_169_tf_fp32.pb https://www.dropbox.com/s/txqyl9tsrza0l55/densenet_169_tf_fp32.pb
wget -O classification/densenet_169/densenet_169_tf_fp16.pb https://www.dropbox.com/s/kaue3ualwq4qphp/densenet_169_tf_fp16.pb
wget -O classification/densenet_169/densenet_169_tflite_int8.tflite https://www.dropbox.com/s/1nd80f3eq3y5d83/densenet_169_tflite_int8.tflite

# resnet_50_v1.5
wget -O classification/resnet_50_v15/resnet_50_v15_tf_fp32.pb https://www.dropbox.com/s/pysdpptedp6py9b/resnet_50_v15_tf_fp32.pb
wget -O classification/resnet_50_v15/resnet_50_v15_tf_fp16.pb https://www.dropbox.com/s/s14l5kq607o1whi/resnet_50_v15_tf_fp16.pb
wget -O classification/resnet_50_v15/resnet_50_v15_tflite_int8.tflite https://www.dropbox.com/s/w7upqmkwaa02iz7/resnet_50_v15_tflite_int8.tflite

# mobilenet_v2
wget -O classification/mobilenet_v2/mobilenet_v2_tf_fp32.pb https://www.dropbox.com/s/thl4v2s6ngspkg3/mobilenet_v2_tf_fp32.pb
wget -O classification/mobilenet_v2/mobilenet_v2_tf_fp16.pb https://www.dropbox.com/s/iqo5xchr8tx8qjt/mobilenet_v2_tf_fp16.pb
wget -O classification/mobilenet_v2/mobilenet_v2_tflite_int8.tflite https://www.dropbox.com/s/euxgo5yficcif9i/mobilenet_v2_tflite_int8.tflite

# ssd_mobilenet_v2
wget -O object_detection/ssd_mobilenet_v2/ssd_mobilenet_v2_tf_fp32.pb https://www.dropbox.com/s/lnaqscsqydzlt1e/ssd_mobilenet_v2_tf_fp32.pb
wget -O object_detection/ssd_mobilenet_v2/ssd_mobilenet_v2_tflite_int8.tflite https://www.dropbox.com/s/hdi9a72uawshp2q/ssd_mobilenet_v2_tflite_int8.tflite

# ssd_inception_v2
wget -O object_detection/ssd_inception_v2/ssd_inception_v2_tf_fp32.pb https://www.dropbox.com/s/jbjgimlrctjgkik/ssd_inception_v2_tf_fp32.pb
wget -O object_detection/ssd_inception_v2/ssd_inception_v2_tf_fp16.pb https://www.dropbox.com/s/lib0gld5tpkudue/ssd_inception_v2_tf_fp16.pb

# yolo_v4_tiny
wget -O object_detection/yolo_v4_tiny/yolo_v4_tiny_tf_fp32.tar.gz https://www.dropbox.com/s/2ogna8d0wqa5war/yolo_v4_tiny_tf_fp32.tar.gz
tar -xvf object_detection/yolo_v4_tiny/yolo_v4_tiny_tf_fp32.tar.gz -C object_detection/yolo_v4_tiny/
rm object_detection/yolo_v4_tiny/yolo_v4_tiny_tf_fp32.tar.gz
