# Cat-Dog-Classification-Kaggle
Cat-Dog classification predictor using Deep Learning (CNNs) using Caffe Framework.

### Features Implemented/To be Implemented
* ~~Preprocess images. (Histogram leveling)~~
* ~~Train Caffe using AlexNet.~~
* ~~Calculate accuracy using validation set.~~

Change the Directory names accordingly.

### Generate Mean images
/home/mohit/caffe/build/tools/compute_image_mean -backend=lmdb /home/mohit/Cat-Dog-Classification-Kaggle/input/train_lmdb /home/mohit/Cat-Dog-Classification-Kaggle/input/mean.binaryproto

### Generate an image of the CNN architecture
python /home/mohit/caffe/python/draw_net.py /home/mohit/Cat-Dog-Classification-Kaggle/model_definition/caffenet_train_val_1.prototxt /home/mohit/Cat-Dog-Classification-Kaggle/model_definition/caffe_model_1.png


### Train the Model

/home/mohit/caffe/build/tools/caffe train --solver /home/mohit/Cat-Dog-Classification-Kaggle/model_definition/solver_1.prototxt 2>&1 | tee /home/mohit/Cat-Dog-Classification-Kaggle/model_definition/model_1_train.log
