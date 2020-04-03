# Hand_on_Chest_Action_Detection_using_Deep_Learning
 A Deep Learning model to detect the action of hand on chest

 # Install Dependencies

     pandas
     numpy
     scipy
     sklearn
     tensorflow-gpu
     keras
     matplotlib

 # Execution

 # Generate landmarks:

     Videos are submitted to the OpenPose portable execution file to generate JSON files of the body landmarks, and the left and right hand landmarks for each frame in the video.

 # Preprocess data:

     Read the JSON files (landmarks for each frame in the video) in every folder.
     Collect the full body (pose), and the left and right hand landmarks.
     Calculate the angle between upper arm and fore arm for each frame.
     Calculate the distance between fore arm and neck for each frame.
     Drop rows with label '0' which has at least one of the previous/next 6 rows with label '1' (Only for training data files)   
     Output the file for each video.

     The above operations can be executed by running 'parse_json_hc.ipynb'

 # Deep Learning model:

     Read all training data files and append them.
     Read the test data file.
     Fit the Deep Learning model on training data.
     Test the model on testing data.
     Save Time vs Label JSON file and graph.

     The above operations can be executed by running 'hc_dnn_model.ipynb'

 # Trained model:

     The trained models are available in the below path

     Hand_on_Chest_Action_Detection_using_Deep_Learning/Trained_model

 # Code Demo Video Link:

     https://youtu.be/KH3XNrsNYJ0    
