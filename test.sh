#!/bin/bash
#Replace the variables with your github repo url, repo name, test
video name, json named by your UIN
GIT_REPO_URL="https://github.com/jug971990/Hand_on_Chest_Action_Detection_using_Deep_Learning.git"
REPO="Hand_on_Chest_Action_Detection_using_Deep_Learning"
VIDEO="./test_videos/tm1.avi"
UIN_JSON="822007030.json"
UIN_JPG="822007030.jpg"
git clone $GIT_REPO_URL
cd $REPO
#Replace this line with commands for running your test python file.
echo $VIDEO
python hc_dnn_model_final.py VIDEO
#If your test file is ipython file, uncomment the following lines and replace IPYTHON_NAME with your test ipython file.
#IPYTHON_NAME="test.ipynb"
#echo $IPYTHON_NAME
#jupyter notebook
#rename the generated timeLabel.json and figure with your UIN.
cp timeLable.json $UIN_JSON
cp timeLable.jpg $UIN_JPG