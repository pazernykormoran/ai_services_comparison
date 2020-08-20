# ai-services-comparison

INSTALL: 
#TODO

WORK: 

1. Edit config.cfg - especially "user_parameters" section
2. Dataset must contain min. 25 photos per label
3. Folders inside dataset directory must look like: 

datasets_dir			the same folder name that in config.cfg
--- folder_with_dataset_name	the same folder name that in config.cfg
--- --- good			here good photos for train
--- --- bad			here bad fotos for train
--- --- test			folder with test photos (must have name 'test')
--- --- --- good		here good photos for test
--- --- --- bad			here bad photos for test

4. Make sure that you have internet connection 
5. Make sure you are going to pass all the service restrictions
 	- In IBM Watson Rekognition you can run script only twice in one account. The cause is that you can train only two models in one free account. You cannot make more predict operations than 1000 during one month. That means - your dataset cannot be bigger than 990 photos wher you run script at new account. Every New account must be created probably from new ip addres.
	- Nano nets max 1000 api calls per model.
	- amazon 2k requestow miesienicznie .
	- google 100 node hours
	- clarifai 10 custom models
	- nanoNets - 1000 api calls and 500 images per month
6. Run the script


