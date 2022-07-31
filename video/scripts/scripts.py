import deeplabcut

# path to YAML file, always need this
config_path='/scratch/sa9nc/Capstone/capstone-kmt2ns-2021-11-07/config.yaml'

# Run this before creating a dataset, will convert all files to correct format for deeplabcut
deeplabcut.convertannotationdata_fromwindows2unixstyle(config_path,userfeedback=False)

#this line of code is optional and is used if you want to manually take a video and put into the test set.
# number after trainindex is the video that will be in the test set and corresponds with the YAML file.
# deeplabcut will automatically split the video into trainig and test sets if you don't use this line
#trainIndices, testIndices=deeplabcut.mergeandsplit(config_path, trainindex=49, uniform=False)

#creates the training set and test set
deeplabcut.create_training_dataset(config_path, augmenter_type='imgaug',windows2linux=True)

# Trains the network, the increase in maxiter will result in a better trained model
deeplabcut.train_network(config_path,displayiters=100, saveiters=15000, maxiters=300000)

# Will provide you with test and training errors along with predicted test and training videos labeled
#deeplabcut.evaluate_network(config_path, plotting=True)

# Both lines below will create a video of deeplabcut predicting the points
deeplabcut.analyze_videos(config_path, ['/scratch/sa9nc/Capstone/capstone-kmt2ns-2021-11-07/videos/105.mp4'], save_as_csv=True)
deeplabcut.create_labeled_video(config_path,['/scratch/sa9nc/Capstone/capstone-kmt2ns-2021-11-07/videos/105.mp4'], videotype='.mp4',save_frames=True)


# Always refer back to deeplabcut documentation for more in depth explanation