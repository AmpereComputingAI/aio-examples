# densenet_169
rm -Rf classification/densenet_169/densenet_169_op11
wget -c https://www.dropbox.com/s/6gja02x160xvpg0/densenet_169_op11.tar.gz -O - | tar -xz
mv densenet_169_op11 classification/densenet_169

# resnet_50_v2
rm -Rf classification/resnet_50_v2/resnet50v2
wget -c https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz -O - | tar -xz
mv resnet50v2 classification/resnet_50_v2

# vgg_16
rm -Rf classification/vgg_16/vgg16
wget -c https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.tar.gz -O - | tar -xz
mv vgg16 classification/vgg_16
rm .*vgg*
