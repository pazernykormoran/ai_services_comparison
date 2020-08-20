import os
import sys
import time
import json
from distutils.dir_util import copy_tree
from datetime import datetime

from clarifai.rest import ClarifaiApp

class Clarifai: 
    def __init__(self, parser, logs, output_path, model_name, datasets_dir,folder_with_dataset_name): 
        print("--- Clarifai: init")

        print(parser.get('clarifaiService', 'description'))

        # global user variables from config file:
        self.model_name = model_name
        self.datasets_dir = datasets_dir
        self.folder_with_dataset_name = folder_with_dataset_name
        self.output_path = output_path

        # parse from config file specific config for this service: 
        api_key = parser.get('clarifaiService', 'api_key')

        self.app = ClarifaiApp(api_key=api_key)

        self.logs = logs

    def configure(self,results,times):
        print("--- Clarifai: configure")
        start = time.time()

        self.model_name = self.model_name + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        copy_tree(self.datasets_dir + "/" +self.folder_with_dataset_name, "./services/clarifai/datasets/"+self.folder_with_dataset_name)
        
        print("--- Clarifai: configured")
        end = time.time()
        results['configured'] = True
        times['configure_time'] = round(end - start)

    def do_some_staff(self):
        print("--- Clarifai: do some staff")


    def load_dataset(self,results, times):
        print("--- Clarifai: load dataset")
        start = time.time()

        directory1 = './services/clarifai/datasets/'+self.folder_with_dataset_name+'/good'
        directory2 = './services/clarifai/datasets/'+self.folder_with_dataset_name+'/bad'
        
        # crawling through directory and subdirectories 
        for root, directories, files in os.walk(directory1): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename)  
                print("Adding image: "+filepath)
                self.app.inputs.create_image_from_filename(filename = filepath, concepts = ["good"])
                

        # crawling through directory and subdirectories 
        for root, directories, files in os.walk(directory2): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                print("Adding image: "+filepath)
                self.app.inputs.create_image_from_filename(filename = filepath, concepts = ["bad"])

        print("--- Clarifai: dataset loaded")
        end = time.time()
        results['dataset_loaded'] = True
        times['dataset_load_time'] = round(end - start)

    def train_model(self,results, times):
        print("--- Clarifai: train model")
        start = time.time()
        model = self.app.models.create(model_id=self.model_name, concepts=["good", "bad"])
        self.model = model.train()
        # print(model.get_info())
        print("--- Clarifai: model has been trained")
        end = time.time()
        results['model_trained'] = True
        print("train model time",round(end - start))
        times['model_train_time'] = round(end - start)


    
    def clear_staff(self):
        print("--- Clarifai: do cleaning")
        
        self.app.inputs.delete_all()
        print("--- Clarifai: cleaned up")


    def download_model(self,results, times):
        print("--- Clarifai: download model")

        start = time.time()

        end = time.time()

        results['model_downloaded'] = True
        times['model_download_time'] = round(end-start)


    def test_model(self,results, times, output):
        print("--- Clarifai: test model")
        start = time.time()


        # uncomment this if you ar using this function only
        # self.model = self.app.models.get('model107-07-2020_12-02-50')

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

        print("--- Clarifai: testing data with good labels ...")
        for root, directories, files in os.walk(directory1): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                # print("precidtion for: ", filepath)
                response = self.model.predict_by_filename(filename = filepath)
                val = round(float(response['outputs'][0]['data']['concepts'][0]['value']),2)
                val2 = round(float(response['outputs'][0]['data']['concepts'][1]['value']),2)
                print(response['outputs'][0]['data']['concepts'][0]['name'],val,response['outputs'][0]['data']['concepts'][1]['name'], val2)

                if response['outputs'][0]['data']['concepts'][0]['name'] == "good": 
                    if float(json.dumps(response['outputs'][0]['data']['concepts'][0]['value'])) > float(json.dumps(response['outputs'][0]['data']['concepts'][1]['value'])) :
                        predicted_class = "good"
                    else:
                        predicted_class = "bad"

                elif response['outputs'][0]['data']['concepts'][0]['name'] == "bad":
                    if float(json.dumps(response['outputs'][0]['data']['concepts'][0]['value'])) > float(json.dumps(response['outputs'][0]['data']['concepts'][1]['value'])) :
                        predicted_class = "bad"
                    else:
                        predicted_class = "good"

                #calc: 
        
                good_number +=1
                if predicted_class == "good":
                    correct_good += 1
                elif predicted_class == "bad":
                    incorrect_bad += 1

                



        print("--- Clarifai: testing data with bad labels ...")
        for root, directories, files in os.walk(directory2): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                # print("precidtion for: ", filepath)
                response = self.model.predict_by_filename(filename = filepath)
                val = round(float(response['outputs'][0]['data']['concepts'][0]['value']),2)
                val2 = round(float(response['outputs'][0]['data']['concepts'][1]['value']),2)
                print(response['outputs'][0]['data']['concepts'][0]['name'],val,response['outputs'][0]['data']['concepts'][1]['name'], val2)
                if response['outputs'][0]['data']['concepts'][0]['name'] == "good": 
                    if float(json.dumps(response['outputs'][0]['data']['concepts'][0]['value'])) > float(json.dumps(response['outputs'][0]['data']['concepts'][1]['value'])) :
                        predicted_class = "good"
                    else:
                        predicted_class = "bad"

                elif response['outputs'][0]['data']['concepts'][0]['name'] == "bad":
                    if float(json.dumps(response['outputs'][0]['data']['concepts'][0]['value'])) > float(json.dumps(response['outputs'][0]['data']['concepts'][1]['value'])) :
                        predicted_class = "bad"
                    else:
                        predicted_class = "good"

                # calc: 
        
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

        print("--- Clarifai: model tested")

        print("print variables: ")
        print("good_number",good_number)
        print("bad_number",bad_number)
        print("correct_good",correct_good)
        print("incorrect_good",incorrect_good)
        print("correct_bad",correct_bad)
        print("incorrect_bad",incorrect_bad)


        end = time.time()
        results['model_tested'] = True
        print("test model time",round(end - start))
        times['model_test_time'] = round(end - start)

    def do_cleaning(self, results, times):
        print("--- Clarifai: do cleaning")
        start = time.time()


                
        self.app.inputs.delete_all()
        print("--- Clarifai: cleaned up")

        end = time.time()

        results['done_cleaning'] = True
        times['do_cleaning_time'] = round(end - start)



