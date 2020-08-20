import os
import sys
import time
import json
from distutils.dir_util import copy_tree
from shutil import copyfile

import shutil

from imageai.Prediction.Custom import ModelTraining
from imageai.Prediction.Custom import CustomImagePrediction


class ImageAI: 
    def __init__(self, parser, logs, output_path, model_name, datasets_dir,folder_with_dataset_name): 
        print("--- ImageAI: init")

        print(parser.get('imageAIService', 'description'))

        # global user variables from config file:
        self.model_name = model_name
        self.datasets_dir = datasets_dir
        self.folder_with_dataset_name = folder_with_dataset_name
        self.output_path = output_path

        # parse from config file specific config for this service: 
        model_architecture = parser.get('imageAIService', 'model_architecture')
        self.enhance_data = False
        if parser.get('imageAIService', 'enhance_data') == 'True': 
            self.enhance_data = True
        self.batch_size = int(parser.get('imageAIService', 'batch_size'))
        self.num_experiments = int(parser.get('imageAIService', 'num_experiments'))

        self.model_architecture = model_architecture


        self.logs = logs


    def configure(self, results, times):
        print("--- ImageAI: configure")
        start = time.time()

        self.model_trainer = ModelTraining()

        if self.model_architecture == "SqueezeNet":
            self.model_trainer.setModelTypeAsSqueezeNet()
    
        if self.model_architecture == "ResNet":
            self.model_trainer.setModelTypeAsResNet()

        if self.model_architecture == "InceptionV3":
            self.model_trainer.setModelTypeAsInceptionV3()

        if self.model_architecture == "DenseNet":
            self.model_trainer.setModelTypeAsDenseNet()

        print("--- ImageAI: configured")
        end = time.time()
        results['configured'] = True
        times['configure_time'] = round(end - start)

    def do_some_staff(self):
        print("--- ImageAI: do some staff")

    def loop_dirs(self,src,dst):
        for root, directories, files in os.walk(src): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename)  
                copyfile(filepath, dst+'/'+filename)

    def load_dataset(self,results, times):
        print("--- ImageAI: load dataset")
        start = time.time()

        if os.path.isdir( "./services/image_ai/datasets/"+self.folder_with_dataset_name):
            shutil.rmtree( "./services/image_ai/datasets/"+self.folder_with_dataset_name)

        copy_tree(self.datasets_dir + "/" +self.folder_with_dataset_name, "./services/image_ai/datasets/"+self.folder_with_dataset_name)

        dst1 = './services/image_ai/datasets/'+self.folder_with_dataset_name+'/train/good'
        dst2 = './services/image_ai/datasets/'+self.folder_with_dataset_name+'/train/bad'
        dst3 = './services/image_ai/datasets/'+self.folder_with_dataset_name+'/test/good'
        dst4 = './services/image_ai/datasets/'+self.folder_with_dataset_name+'/test/bad'
        os.makedirs(dst1, exist_ok=True)
        os.makedirs(dst2, exist_ok=True)

        os.makedirs(dst3, exist_ok=True)
        os.makedirs(dst4, exist_ok=True)

        src1 = self.datasets_dir+'/'+self.folder_with_dataset_name+'/good'
        src2 = self.datasets_dir+'/'+self.folder_with_dataset_name+'/bad'

        src3 = self.datasets_dir+'/'+self.folder_with_dataset_name+'/test/good'
        src4 = self.datasets_dir+'/'+self.folder_with_dataset_name+'/test/bad'
        
        # crawling through directory and subdirectories 

        self.loop_dirs(src1,dst1)
        self.loop_dirs(src2,dst2)
        self.loop_dirs(src3,dst3)
        self.loop_dirs(src4,dst4)

        print("--- ImageAI: dataset loaded")
        end = time.time()
        results['dataset_loaded'] = True
        times['dataset_load_time'] = round(end - start)

    def train_model(self,results, times):
        print("--- ImageAI: train model")
        start = time.time()

        print("--- ImageAI: Wait 600 s before trainig starts")
        time.sleep(600)

        self.model_folder = str(round(time.time()))
        self.models_dir = os.path.join(self.datasets_dir, self.folder_with_dataset_name, self.logs.output_folder, "models", 'image_ai')
        self.model_dir = os.path.join(self.models_dir, self.model_folder)

        self.model_trainer.setDataDirectory(data_directory= './services/image_ai/datasets/'+self.folder_with_dataset_name)



        self.model_trainer.trainModel(num_objects=2, num_experiments=self.num_experiments, 
            enhance_data=self.enhance_data, batch_size=self.batch_size, show_network_summary=True)

        copy_tree(os.path.join('./services/image_ai/datasets',self.folder_with_dataset_name, 'models'),self.model_dir)
        copy_tree(os.path.join('./services/image_ai/datasets',self.folder_with_dataset_name, 'json'),self.model_dir)


        end = time.time()
        print("train model time",round(end - start))
        print("--- ImageAI: model has been trained")
        results['model_trained'] = True
        times['model_train_time'] = round(end - start)

    def download_model(self,results, times):
        print("--- ImageAI: download model")
        

        start = time.time()

        end = time.time()

        results['model_downloaded'] = True
        times['model_download_time'] = round(end-start)

    
    def do_cleaning(self,results, times):
        print("--- ImageAI: do cleaning")
        start = time.time()

        #TODO delete dataset folder in service

        end = time.time()

        print("--- ImageAI: cleaned up")

        results['done_cleaning'] = True
        times['do_cleaning_time'] = round(end - start)

    def test_model(self,results, times, output):
        print("--- ImageAi: test model")
        start = time.time()

        #set this under to use only this functin: 
        # self.model_dir = '/home/george/inzynierka/datasets/dataset2-leki/script_output_07-06-2020_11:49:17/models/image_ai/1591523357'
        # self.model_dir = '/home/george/inzynierka/datasets/hotdog-nothotdog/script_output_07-07-2020_12:02:44/models/image_ai/1594116949'

        prediction = CustomImagePrediction()

        if self.model_architecture == "SqueezeNet":
            prediction.setModelTypeAsSqueezeNet()
    
        if self.model_architecture == "ResNet":
            prediction.setModelTypeAsResNet()

        if self.model_architecture == "InceptionV3":
            prediction.setModelTypeAsInceptionV3()

        if self.model_architecture == "DenseNet":
            prediction.setModelTypeAsDenseNet()

        #find best model in model dir folder: 
        print("ImageAi: find best model")
        name_of_best_model = ""
        best_acc = 0
        for root, directories, files in os.walk(self.model_dir): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename)
                print("model filapath: ", filepath)
                index = filename.find('acc') 
                if index > 0:  
                    str_acc = filename[index+4 : index+10]
                    acc = float(str_acc)
                    print("acc", acc)
                    if acc > best_acc:
                        best_acc = acc
                        name_of_best_model = filename
        
        print("best model: ",name_of_best_model)

        prediction.setModelPath(os.path.join(self.model_dir, name_of_best_model))
        prediction.setJsonPath(os.path.join(self.model_dir, "model_class.json"))
        prediction.loadModel(num_objects=2)



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

        print("--- ImageAI: testing data with good labels ...")
        for root, directories, files in os.walk(directory1): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                print("precidtion for: ", filepath)
                
                predictions, probabilities = prediction.predictImage(filepath, result_count = 2)

                prob = 0
                predicted_class = ""

                for eachPrediction, eachProbability in zip(predictions, probabilities):
                    print(eachPrediction , " : " , eachProbability)
                    if eachProbability > prob:
                        prob = eachProbability
                        predicted_class = eachPrediction
                        
                #calc: 
        
                good_number +=1
                if predicted_class == "good":
                    correct_good += 1
                elif predicted_class == "bad":
                    incorrect_bad += 1

                

        print("--- ImageAI: testing data with bad labels ...")
        for root, directories, files in os.walk(directory2): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                print("precidtion for: ", filepath)

                predictions, probabilities = prediction.predictImage(filepath, result_count = 2)

                prob = 0
                predicted_class = ""

                for eachPrediction, eachProbability in zip(predictions, probabilities):
                    print(eachPrediction , " : " , eachProbability)
                    if eachProbability > prob:
                        prob = eachProbability
                        predicted_class = eachPrediction


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

        print("--- ImageAI: model tested")

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

    def check_precision(self):
        print("--- ImageAI: check precision")