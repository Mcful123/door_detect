# door_detect
This code detects if the XRD and furnace doors are open or closed using the webcam and machine learning

for training the door detect models, I used https://teachablemachine.withgoogle.com/train/image. Each door gets its own separate model and the images used for training are in the folder. The images are resized to (224, 224) beforehand and also 60 images per door is added because with fewer than that, the model is unusable. 
