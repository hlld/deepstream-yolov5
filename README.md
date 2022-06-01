## DeepStream YOLOv5

An implementation of YOLOv5 running on DeepStream 6

## Requirements

```
CUDA 11.6
TensorRT 8.2
DeepStream 6.1
```

### How to Use

Enter $ROOT of this repository.

* Step1: Prepare the wts file of YOLOv5s model follow instructions [tensorrtx/yolov5](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5), then put the wts file into $ROOT/data folder and rename as `yolov5s.wts`. Note that the model version of YOLOv5s is 6.1.
* Step2: Enter $ROOT/source folder, modify `EXFLAGS` and `EXLIBS` in `Makefile` corresponding to your installed TensorRT library path, run `make` command to compile the run-time library.
* Step3: Back to $ROOT folder, run `deepstream-app -c configs/deepstream_app_config_yolov5s.txt` command.
* Step4: rename the generated engine file `model_b1_gpu0_fp16.engine` as `yolov5s_b1_gpu0_fp16.engine` for reuse.

## Acknowledgements

* [https://github.com/wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)
