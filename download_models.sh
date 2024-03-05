set -eo pipefail

# densenet_169
wget -c https://ampereaimodelzoo.s3.amazonaws.com/densenet_169_op11.tar.gz -O - | tar -xz
cp densenet_169_op11/densenet_169_op11.onnx classification/densenet_169
rm -Rf densenet_169_op11

# resnet_50_v1.5 fp32
wget -c https://zenodo.org/records/4735647/files/resnet50_v1.onnx
mv resnet50_v1.onnx classification/resnet_50_v1.5/resnet_50_v1.5_fp32.onnx

# resnet_50_v1.5 fp16
wget -c https://ampereaimodelzoo.s3.amazonaws.com/resnet_50_v1.5_fp16.onnx
mv resnet_50_v1.5_fp16.onnx classification/resnet_50_v1.5

# ssd_mobilenet_v1
wget -c https://ampereaimodelzoo.s3.amazonaws.com/torch2onnx_ssd_mobilenet_v1.onnx
mv torch2onnx_ssd_mobilenet_v1.onnx object_detection/ssd_mobilenet_v1
