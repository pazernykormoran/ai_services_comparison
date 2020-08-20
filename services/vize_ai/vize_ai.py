import os
import sys
import time
import json
from distutils.dir_util import copy_tree
import base64

from ximilar.client import RecognitionClient, DetectionClient
from ximilar.client import DominantColorProductClient, DominantColorGenericClient
from ximilar.client import FashionTaggingClient, GenericTaggingClient

class VizeAi: 

    label_bad = None
    label_good = None
    def __init__(self, parser, logs, output_path, model_name, datasets_dir,folder_with_dataset_name): 
        print("--- VizeAi: init")
        print(parser.get('vizeAiService', 'description'))

        ##TODO nie tworz zawsze nowych categories

        # global user variables from config file:
        self.model_name = model_name
        self.datasets_dir = datasets_dir
        self.folder_with_dataset_name = folder_with_dataset_name
        self.output_path = output_path

        # parse from config file specific config for this service: 
        token = parser.get('vizeAiService', 'token')
        task_name = parser.get('vizeAiService', 'task_name')

        self.task_name = task_name
        self.client = RecognitionClient(token=token)

        self.logs = logs


    def configure(self, results, times):
        print("--- VizeAi: configure")
        start = time.time()
        
        copy_tree(self.datasets_dir + "/" +self.folder_with_dataset_name, "./services/vize_ai/datasets/"+self.folder_with_dataset_name)

        self.classification_task, status = self.client.create_task(str(time.time()))
        labels, status = self.client.get_all_labels()
        for label in labels: 
            if label.name == 'good':
                print("found good")
                self.label_good = label


            elif label.name == 'bad':
                print("found bad")
                self.label_bad = label

        if len(labels) == 0:
            self.label_good, status = self.client.create_label(name='good')
            self.label_bad, status = self.client.create_label(name='bad')
        elif len(labels) == 1: 
            if labels[0].name == 'bad':
                self.label_good, status = self.client.create_label(name='good')
            elif labels[0].name == 'good': 
                self.label_bad, status = self.client.create_label(name='bad')

        self.classification_task.add_label(self.label_good.id)
        self.classification_task.add_label(self.label_bad.id)

        print(self.classification_task)
        self.task_id = self.classification_task.id

        
        print("--- VizeAi: configured")
        end = time.time()
        results['configured'] = True
        times['configure_time'] = round(end - start)

    def do_some_staff(self):
        print("--- VizeAi: do some staff")


    def load_dataset(self,results, times):
        print("--- VizeAi: load dataset")
        start = time.time()


        directory1 = './services/vize_ai/datasets/'+self.folder_with_dataset_name+'/good'
        directory2 = './services/vize_ai/datasets/'+self.folder_with_dataset_name+'/bad'
        
        # crawling through directory and subdirectories 

        list1= []
        for root, directories, files in os.walk(directory1): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename)  
                d={'_file': filepath, 'labels': [self.label_good.id]}
                list1.append(d)

                

        # crawling through directory and subdirectories 
        list2= []
        for root, directories, files in os.walk(directory2): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                d={'_file': filepath, 'labels': [self.label_bad.id]}
                list2.append(d)

        print("--- adding good examples...")
        images, status = self.client.upload_images(list1)
        print(status)
        print("--- adding bad examples...")
        images, status = self.client.upload_images(list2)
        print(status)

        end = time.time()
        print("--- VizeAi: dataset loaded")
        results['dataset_loaded'] = True
        times['dataset_load_time'] = round(end - start)

    def train_model(self,results, times):
        print("--- VizeAi: train model")
        start = time.time()
        self.classification_task.train()

        

        self.check_training_status()

        end = time.time()
        print("train model time",round(end - start))
        print("--- VizeAi: model has been trained")
        results['model_trained'] = True
        times['model_train_time'] = round(end - start)

    def check_training_status(self):

        
        while True: 
            print("--- VizeAi: check training status... ")
            time.sleep(20)
            task, status = self.client.get_task(task_id = self.task_id)
            if task.last_train_status == 'TRAINED':
                self.task = task
                break


    def download_model(self,results, times):
        print("--- VizeAi: download model")

        start = time.time()

        end = time.time()

        results['model_downloaded'] = True
        times['model_download_time'] = round(end-start)

    def test_model(self,results, times, output):
        print("--- VizeAi: test model")
        start = time.time()

        # # TODO uncomment that shit if u want to use only that function: 
        # self.task_id = '1bce5aff-70f9-4f4e-a6ee-d499c6623cfc'
        # self.task, status = self.client.get_task(task_id  = self.task_id)


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

        print("--- VizeAi: testing data with good labels ...")
        for root, directories, files in os.walk(directory1): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                result  = self.task.classify([{'_file': filepath}])

                predicted_class = result['records'][0]['best_label']['name']
                prob = result['records'][0]['best_label']['prob']

                good_number +=1
                print('{: <{}}'.format(str(good_number), 4) ,'{: <{}}'.format('good', 6), '{: <{}}'.format(predicted_class, 6), '{: <{}}'.format(str(round(prob,4)), 6), filename)
        
                
                if predicted_class == "good":
                    correct_good += 1
                elif predicted_class == "bad":
                    incorrect_bad += 1

                



        print("--- VizeAi: testing data with bad labels ...")
        for root, directories, files in os.walk(directory2): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                result  = self.task.classify([{'_file': filepath}])

                predicted_class = result['records'][0]['best_label']['name']
                prob = result['records'][0]['best_label']['prob']

                # calc: 
        
                bad_number +=1

                print('{: <{}}'.format(str(bad_number), 4) ,'{: <{}}'.format('bad', 6), '{: <{}}'.format(predicted_class, 6), '{: <{}}'.format(str(round(prob,4)), 6), filename)

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

        print("--- VizeAi: model tested")

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

    
    def do_cleaning(self,results,times):
        print("--- VizeAi: do cleaning")
        start = time.time()
        

        # print("client")
        # print(self.client.__dict__)
        # print(dir(self.client))
        # print("")

        images = self.client.training_images_iter()
        for img in images:
            print('removing', img.id)
            res = self.client.remove_image(img.id)

        # TODO delete lastest task (3)
        tasks, status = self.client.get_all_tasks()
        
        if len(tasks) >2:
            num = time.time()
            taskid = ""
            for task in tasks: 
                print(dir(task))
                print("")
                print(task.__dict__)

                if float(task.name) < num: 
                    num = float(task.name)
                    taskid = task.id
            res = self.client.remove_task(taskid)
            print(res)


        end = time.time()
        print("--- VizeAi: cleaned up")
        results['done_cleaning'] = True
        times['do_cleaning_time'] = round(end - start)


