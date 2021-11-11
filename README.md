# aio-examples

You can try AIO by either running jupyter notebook examples or python scripts on CLI level. 

**Note: Before running the examples, please run download_models.sh script to pull down all models.**

## Running Jupyter Notebook QuickStart Examples

Use DLS_NUM_THREADS to specify the number of cores the AIO compute kernels will run on
```
export DLS_NUM_THREADS=16
cd /aio-examples/
bash start_notebook.sh
```

If you run it on a cloud instance, make sure your machine has port 8080 open and on your local device run:
```
ssh -N -L 8080:localhost:8080 -i <ssh key> your_user@xxx.xxx.xxx.xxx
```

Use a browser to point to the URL printed out by the Jupyter notebook launcher. You will find 
Jupyter Notebook examples, examples.ipynb, under /classification and /object_detection folders.
The examples run through several inference models, visualize results and present the performance
numbers.

## Running Examples With CLI
To use CLI-level scripts:

Use DLS_NUM_THREADS to specify the number of cores the AIO compute kernels will run on
```
export DLS_NUM_THREADS=16
cd /aio-examples/
```

Go to the directory of choice, eg.
```
cd classification/resnet_50_v15
```
Evaluate the model with run.py script

Optional arguments:

  -h, --help            show this help message and exit
  
  -m MODEL_PATH, --model_path MODEL_PATH
                        
  -p {fp32,fp16,int8}, --precision {fp32,fp16,int8}
                        
  -b BATCH_SIZE, --batch_size BATCH_SIZE

```
python run.py -m resnet_50_v15_tf_fp32.pb -p fp32
python run.py -m resnet_50_v15_tflite_int8.tflite -p int8
```
