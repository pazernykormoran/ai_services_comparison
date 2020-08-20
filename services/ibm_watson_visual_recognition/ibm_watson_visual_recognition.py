import os
import sys
import time
import json

from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from distutils.dir_util import copy_tree
from zipfile import ZipFile 

class IBMWatsonVisualRecognition: 
    def __init__(self, parser, logs, output_path, model_name, datasets_dir,folder_with_dataset_name): 
        print("--- IBMWatsonVisualRecognition: init")

        print(parser.get('IBMService', 'description'))

        # global user variables from config file:
        self.model_name = model_name
        self.datasets_dir = datasets_dir
        self.folder_with_dataset_name = folder_with_dataset_name
        self.output_path = output_path

        # parse from config file specific config for this service: 
        url = parser.get('IBMService', 'url')
        api_key = parser.get('IBMService', 'api_key')
        version = parser.get('IBMService', 'version')

        authenticator = IAMAuthenticator(api_key)
        self.visual_recognition = VisualRecognitionV3(
            version = version,
            authenticator = authenticator
        )
        self.visual_recognition.set_service_url(url)

        self.logs = logs



    def configure(self,results,times):
        print("--- IBMWatsonVisualRecognition: configure")
        start = time.time()

        copy_tree(self.datasets_dir + "/" +self.folder_with_dataset_name, "./services/ibm_watson_visual_recognition/datasets/"+self.folder_with_dataset_name)

        directory1 = './services/ibm_watson_visual_recognition/datasets/'+self.folder_with_dataset_name+'/good'
        directory2 = './services/ibm_watson_visual_recognition/datasets/'+self.folder_with_dataset_name+'/bad'
        
        file_paths1 = [] 
        # crawling through directory and subdirectories 
        for root, directories, files in os.walk(directory1): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                file_paths1.append(filepath) 

        file_paths2 = [] 
        # crawling through directory and subdirectories 
        for root, directories, files in os.walk(directory2): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                file_paths2.append(filepath) 

        # writing files to a zipfile 
        with ZipFile('./services/ibm_watson_visual_recognition/datasets/'+self.folder_with_dataset_name+'/good.zip','w') as zip: 
            # writing each file one by one 
            for file in file_paths1: 
                zip.write(file) 
    
        # writing files to a zipfile 
        with ZipFile('./services/ibm_watson_visual_recognition/datasets/'+self.folder_with_dataset_name+'/bad.zip','w') as zip: 
            # writing each file one by one 
            for file in file_paths2: 
                zip.write(file) 
    
    
        print('All files zipped successfully!')  
        print("--- IBMWatsonVisualRecognition configured")  
        end = time.time()
        results['configured'] = True
        times['configure_time'] = round(end - start)
  
    def load_dataset(self,results, times):
        print("--- IBMWatsonVisualRecognition: load_dataset") 
        start = time.time()

        end = time.time()

        results['dataset_loaded'] = True
        times['dataset_load_time'] = round(end - start)

    def train_model(self,results, times):
        print("--- IBMWatsonVisualRecognition: train model")
        start = time.time()

        with open('./services/ibm_watson_visual_recognition/datasets/'+self.folder_with_dataset_name+'/good.zip', 'rb') as good, open('./services/ibm_watson_visual_recognition/datasets/'+self.folder_with_dataset_name+'/bad.zip', 'rb') as bad:
            model = self.visual_recognition.create_classifier(
            self.model_name,
            positive_examples={'good': good, 'bad': bad}).get_result()
        print(json.dumps(model, indent=2))  

        self.model_id = model['classifier_id']
        print("")
        print("your model id: ", self.model_id)

        self.check_training_status()

        results['model_trained'] = True
        end = time.time()
        print("train model time",round(end - start))
        times['model_train_time'] = round(end - start)

    def check_training_status(self):
        
        dobreak = False

        while True: 
            print("--- IBMWatsonVisualRecognition: check training status")
            time.sleep(20)
    
            classifiers = self.visual_recognition.list_classifiers(verbose=True).get_result()
            # print(json.dumps(classifiers, indent=2))

            
            for operation in classifiers['classifiers']:
                if operation['classifier_id'] == self.model_id and operation['status'] == "ready":
                    print("Training has been finished")
                    dobreak = True

            if dobreak == True: 
                break

    def download_model(self, results, times):
        print("--- IBMWatsonVisualRecognition: download model")
        start = time.time()

        # core_ml_model = self.visual_recognition.get_core_ml_model(
        #     classifier_id=self.model_id).get_result()
        # folder = './services/ibm_watson_visual_recognition/models/'
        # try:
        #     os.makedirs(folder)
        # except OSError as exc: # Guard against race condition
        #     print("error while creating directory, probably directory exists")
        # with open(folder+self.model_id+".mlmodel", 'wb') as fp:
        #     fp.write(core_ml_model.content)  
        # print("--- model has been downloaded") 


        end = time.time()

        results['model_downloaded'] = True
        times['model_download_time'] = round(end-start) 

    def test_model(self,results, times, output):
        print("--- IBMWatsonVisualRekognition: test model")
        start = time.time()

        ## uncomment this if u want to test only this function: 
        # classifiers = self.visual_recognition.list_classifiers(verbose=True).get_result()
        # print(json.dumps(classifiers, indent=2))
        # for operation in classifiers['classifiers']:
        #     if len(operation['classes'])>1:
        #         self.model_id = operation['classifier_id']


        ### prepare variables:
        good_precision = 0 # correct_good / (correct_good + incorrect_good)
        bad_precision = 0 # correct_bad / (correct_bad + incorrect_bad)
        good_recall = 0 # correct_good / good_number
        bad_recall = 0 # correct_bad / bad_number
        good_f1 = 0 # 2*good_precision*good_recall/(good_precision+good_recall)
        bad_f1 = 0 # 2*bad_precision*bad_recall/(bad_precision+bad_recall)

        good_number = 0
        bad_number = 0

        correct_good = 0
        incorrect_good = 0

        correct_bad = 0
        incorrect_bad = 0

        ### walk through directories
        directory1 = os.path.join(self.datasets_dir, self.folder_with_dataset_name, "test", "good")
        directory2 = os.path.join(self.datasets_dir, self.folder_with_dataset_name, "test", "bad")

        print("--- IBMWatsonVisualRekognition: testing data with good labels ...")
        for root, directories, files in os.walk(directory1): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                # print("precidtion for: ", filepath)
                with open(filepath, 'rb') as images_file:
                    classes = self.visual_recognition.classify(
                        images_file=images_file,
                        threshold='0',
                        classifier_ids = [self.model_id],
                        owners=["me"]).get_result()
                    # print(json.dumps(classes, indent=2))
                    print(json.dumps(classes['images'][0]['classifiers'][0]['classes'][0]), json.dumps(classes['images'][0]['classifiers'][0]['classes'][1]))

                    if classes['images'][0]['classifiers'][0]['classes'][0]['class'] == "good":
                        if float(json.dumps(classes['images'][0]['classifiers'][0]['classes'][0]['score'])) > float(json.dumps(classes['images'][0]['classifiers'][0]['classes'][1]['score'])):
                            predicted_class = "good"
                        else: 
                            predicted_class = "bad"
                    elif classes['images'][0]['classifiers'][0]['classes'][0]['class'] == "bad":
                        if float(json.dumps(classes['images'][0]['classifiers'][0]['classes'][0]['score'])) > float(json.dumps(classes['images'][0]['classifiers'][0]['classes'][1]['score'])):
                            predicted_class = "bad"
                        else: 
                            predicted_class = "good"

                #calc: 
        
                good_number +=1
                if predicted_class == "good":
                    correct_good += 1
                elif predicted_class == "bad":
                    incorrect_bad += 1

                



        print("--- IBMWatsonVisualRekognition: testing data with bad labels ...")
        for root, directories, files in os.walk(directory2): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                # print("precidtion for: ", filepath)
                with open(filepath, 'rb') as images_file:
                    classes = self.visual_recognition.classify(
                        images_file=images_file,
                        threshold='0',
                        classifier_ids = [self.model_id],
                        owners=["me"]).get_result()
                    # print(json.dumps(classes, indent=2))
                    print(json.dumps(classes['images'][0]['classifiers'][0]['classes'][0]), json.dumps(classes['images'][0]['classifiers'][0]['classes'][1]))
            
                    if classes['images'][0]['classifiers'][0]['classes'][0]['class'] == "good":
                        if float(json.dumps(classes['images'][0]['classifiers'][0]['classes'][0]['score'])) > float(json.dumps(classes['images'][0]['classifiers'][0]['classes'][1]['score'])):
                            predicted_class = "good"
                        else: 
                            predicted_class = "bad"
                    elif classes['images'][0]['classifiers'][0]['classes'][0]['class'] == "bad":
                        if float(json.dumps(classes['images'][0]['classifiers'][0]['classes'][0]['score'])) > float(json.dumps(classes['images'][0]['classifiers'][0]['classes'][1]['score'])):
                            predicted_class = "bad"
                        else: 
                            predicted_class = "good"


                #calc: 
        
                bad_number +=1
                if predicted_class == "bad":
                    correct_bad += 1
                elif predicted_class == "good":
                    incorrect_good += 1


        ### calculate: 
        if correct_good is not 0 or incorrect_good is not 0:
            good_precision =  correct_good / (correct_good + incorrect_good)
        if correct_bad is not 0 or incorrect_bad is not 0: 
            bad_precision = correct_bad / (correct_bad + incorrect_bad)
        if good_number is not 0:
            good_recall = correct_good / good_number
        if bad_number is not 0:
            bad_recall = correct_bad / bad_number
        if good_precision is not 0 and good_recall is not 0: 
            good_f1 = 2*good_precision*good_recall/(good_precision+good_recall)
        if bad_precision is not 0 and bad_recall is not 0:
            bad_f1 = 2*bad_precision*bad_recall/(bad_precision+bad_recall)

        output['good_precision'] = round(good_precision,2)
        output['good_recall'] = round(good_recall,2)
        output['good_f1'] = round(good_f1,2)
        output['bad_precision'] = round(bad_precision,2)
        output['bad_recall'] = round(bad_recall,2)
        output['bad_f1'] = round(bad_f1,2)

        print("--- GoogleCloudVision: model tested")

        print("print variables: ")
        print("good_number",good_number)
        print("bad_number",bad_number)
        print("correct_good",correct_good)
        print("incorrect_good",incorrect_good)
        print("correct_bad",correct_bad)
        print("incorrect_bad",incorrect_bad)

        end = time.time()
        results['model_tested'] = True
        times['model_test_time'] = round(end - start)



        end = time.time()
        results['model_tested'] = True
        times['model_test_time'] = round(end - start)

    def do_cleaning(self, results, times):
        print("--- IBMWatsonVisualRecognition: do cleaning")
        start = time.time()

        end = time.time()

        # visual_recognition.delete_classifier(self.model_id) 
        print("--- cleaned up")    

        results['done_cleaning'] = True
        times['do_cleaning_time'] = round(end - start)   
    
    def do_some_staff(self):
        print("--- IBMWatsonVisualRecognition: do_some_staff")


