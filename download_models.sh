# densenet_169
wget -c https://www.dropbox.com/s/6gja02x160xvpg0/densenet_169_op11.tar.gz -O - | tar -xz
cp densenet_169_op11/densenet_169_op11.onnx classification/densenet_169
rm -Rf densenet_169_op11

# resnet_50_v1.5 fp32
wget -c https://zenodo.org/record/2592612/files/resnet50_v1.onnx
mv resnet50_v1.onnx classification/resnet_50_v1.5

# resnet_50_v1.5 fp16
wget -c https://www.dropbox.com/s/r80ndhbht7tixn5/resnet_50_v1.5_fp16.onnx
mv resnet_50_v1.5_fp16.onnx classification/resnet_50_v1.5
