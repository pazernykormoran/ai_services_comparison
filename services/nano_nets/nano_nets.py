import os
import sys
import time
import json
from distutils.dir_util import copy_tree

import shutil

import requests

class NanoNets: 
    def __init__(self, parser, logs, output_path, model_name, datasets_dir,folder_with_dataset_name): 
        print("--- NanoNets: init")

        print(parser.get('nanoNetsService', 'description'))

        ##TODO n ie tworz zawsze nowego modelu

        # global user variables from config file:
        self.datasets_dir = datasets_dir
        self.folder_with_dataset_name = folder_with_dataset_name
        self.output_path = output_path
        
        # parse from config file specific config for this service: 
        api_key = parser.get('nanoNetsService', 'api_key')

        self.api_key = api_key
        self.headers = {
            'accept': 'application/x-www-form-urlencoded'
        }

        self.logs = logs


    def configure(self,results,times):
        print("--- NanoNets: configure")
        start = time.time()

        if os.path.isdir( "./services/nano_nets/datasets/"+self.folder_with_dataset_name):
            shutil.rmtree( "./services/nano_nets/datasets/"+self.folder_with_dataset_name)

        copy_tree(self.datasets_dir + "/" +self.folder_with_dataset_name, "./services/nano_nets/datasets/"+self.folder_with_dataset_name)

        url = 'https://app.nanonets.com/api/v2/ImageCategorization/Model/'



        data = {'categories' : ['good', 'bad']}

        response = requests.request("POST", url, headers=self.headers, auth=requests.auth.HTTPBasicAuth(self.api_key, ''), data=data)
        self.model = response.json()

        print(self.model)
        print(self.model['model_id'])

        print("--- NanoNets: configured")
        end = time.time()
        results['configured'] = True
        times['configure_time'] = round(end - start)

    def do_some_staff(self):
        print("--- NanoNets: do some staff")
        


    def load_dataset(self,results, times):
        print("--- NanoNets: load dataset")
        start = time.time()

        directory1 = './services/nano_nets/datasets/'+self.folder_with_dataset_name+'/good'
        directory2 = './services/nano_nets/datasets/'+self.folder_with_dataset_name+'/bad'
        url = 'https://app.nanonets.com/api/v2/ImageCategorization/UploadFile/'

        # crawling through directory and subdirectories 

        for root, directories, files in os.walk(directory1): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                os.rename(os.path.join(root, filename), os.path.join(root, 'good'+filename) )
                filepath = os.path.join(root, 'good'+filename)  
                print(filepath)
                data = {'file' :open(filepath, 'rb'),'category' :('', 'good'), 'modelId' :('', self.model['model_id'])}
                response = requests.post(url, auth= requests.auth.HTTPBasicAuth(self.api_key, ''), files=data)


                

        # crawling through directory and subdirectories 
        for root, directories, files in os.walk(directory2): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                os.rename(os.path.join(root, filename), os.path.join(root, 'bad'+filename) )
                filepath = os.path.join(root, 'bad'+filename)
                print(filepath)
                data = {'file' :open(filepath, 'rb'),'category' :('', 'bad'), 'modelId' :('', self.model['model_id'])}
                response = requests.post(url, auth= requests.auth.HTTPBasicAuth(self.api_key, ''), files=data)



        print("--- NanoNets: dataset loaded")
        end = time.time()
        results['dataset_loaded'] = True
        times['dataset_load_time'] = round(end - start)

    def train_model(self,results, times):
        print("--- NanoNets: train model")
        start = time.time()
        
        url = 'https://app.nanonets.com/api/v2/ImageCategorization/Train/'

        querystring = {'modelId': self.model['model_id']}

        response = requests.request('POST', url, headers=self.headers, auth=requests.auth.HTTPBasicAuth(self.api_key, ''), params=querystring)

        print(response.text)

        print("--- NanoNets: model is training")

        self.check_training_status()

        end = time.time()
        print("--- NanoNets: train model time",round(end - start))
        results['model_trained'] = True
        times['model_train_time'] = round(end - start)

    def check_training_status(self):
        while True: 
            print("--- NanoNets: check training status")
            time.sleep(20)

            # modelid = '9bf5b1b9-48d6-4925-a1b8-5aade86b7fe5'

            url = 'https://app.nanonets.com/api/v2/OCR/Model/' + self.model['model_id']

            response = requests.get( url, auth=requests.auth.HTTPBasicAuth(self.api_key,''))

            if response.json()["status"] != "Training in progress":
                break

        print("--- NanoNets: Training finished")

    def download_model(self, results, times):
        print("--- NanoNets: download model")

        start = time.time()

        end = time.time()

        results['model_downloaded'] = True
        times['model_download_time'] = round(end-start)


    def predict(self, imagepath):
        url = 'https://app.nanonets.com/api/v2/ImageCategorization/LabelFile/'

        data = {'file': open(imagepath, 'rb'), 'modelId': ('', self.model['model_id'])}

        response = requests.post(url, auth= requests.auth.HTTPBasicAuth(self.api_key, ''), files=data)

        print(response.text)
        return response.json()['result']

    def test_model(self,results, times, output):
        print("--- NanoNets: test model")
        start = time.time()

        # ## uncommnet this if u want to test only this fucntion: 
        # url = 'https://app.nanonets.com/api/v2/ImageCategorization/Model/'
        # querystring = {'modelId': '906a951d-e94f-4982-97bf-2ef3efc49bb9'}
        # response = requests.request('GET', url, auth=requests.auth.HTTPBasicAuth(self.api_key,''), params=querystring)
        # print(response.text)
        # self.model = response.json()



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

        print("--- NanoNets: testing data with good labels ...")
        for root, directories, files in os.walk(directory1): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                print("precidtion for: ", filepath)
                
                res = self.predict(filepath)

                if res[0]['prediction'][0]['probability'] > res[0]['prediction'][1]['probability']:
                    predicted_class = res[0]['prediction'][0]['label']
                else: 
                    predicted_class = res[0]['prediction'][1]['label']
                  

                #calc: 
        
                good_number +=1
                if predicted_class == "good":
                    correct_good += 1
                elif predicted_class == "bad":
                    incorrect_bad += 1

                

        print("--- NanoNets: testing data with bad labels ...")
        for root, directories, files in os.walk(directory2): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                # print("precidtion for: ", filepath)
                res = self.predict(filepath)

                if res[0]['prediction'][0]['probability'] > res[0]['prediction'][1]['probability']:
                    predicted_class = res[0]['prediction'][0]['label']
                else: 
                    predicted_class = res[0]['prediction'][1]['label']


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

        print("--- NanoNets: model tested")

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

    
    def do_cleaning(self, results, times):
        print("--- NanoNets: do cleaning")
        start = time.time()

        # self.client.remove_label(self.label_good.id)
        # self.client.remove_label(self.label_bad.id)
        # delete all labels
        # delete task
        print("--- NanoNets: cleaned up")
        end = time.time()

        results['done_cleaning'] = True
        times['do_cleaning_time'] = round(end - start)
