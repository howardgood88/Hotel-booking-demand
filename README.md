# Hotel-booking-demand
Final project of Machine Learning Techniques of prof. Hsuan-Tien Lin - NTU.  
description: https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/project/project.pdf

# Training
### Steps
1. Predict `is_canceled`.
2. Predict `adr` of each request.
3. Sum up (adr * staying days) of each request to get daily revenue, and predict the `scale`.

### Model
1. SVM: Change the encoding_mode to `one_hot` in preprocessing.py and run it to generate input file, then run svm.py to train.
2. Random Forest: Change the encoding_mode to `label_encode` in preprocessing.py and run it to generate input file, then run ranforest.py to train.
3. Neural Network(NN): Run nn_is_cancel.py, nn_adr.py and nn_label.py respectively to train.

# File List
1. nn_*.py: Three main file and one model for NN.
2. ranforest.py: Main file for Random Forest.
3. svm.py: Main file for SVM.
4. preprocessing.py: Generate `Dataset/train_final.csv` and `Dataset/test_final.csv` using `train_day_of_week.csv` and `test_day_of_week.csv`.
5. util.py: Collections of utility function.