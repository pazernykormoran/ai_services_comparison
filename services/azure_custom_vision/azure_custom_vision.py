import sys, socket, time, urllib.request, os.path, datetime, struct, zipfile
import uuid
import json
from distutils.dir_util import copy_tree

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient



import numpy as np
import cv2
import tensorflow as tf
from os import path, walk
from PIL import Image

class Predict:
    graph_def: tf.compat.v1.GraphDef = tf.compat.v1.GraphDef()
    labels = []
    tensor_size = 224

    def init(self, model_dir: str):
        self.__import_labels(path.join(model_dir, 'labels.txt'))
        self.__import_model(path.join(model_dir, 'model.pb'))
            


    def run(self, image_file: str):
        image = Image.open(image_file)
        image = self.update_orientation(image)
        image = self.__convert_to_opencv(image)
        image = self.resize_down_to_1600_max_dim(image)

        h, w = image.shape[:2]
        min_dim = min(w,h)
        max_square_image = self.__crop_center(image, min_dim, min_dim)

        augmented_image = self.resize_to_256_square(max_square_image)

        augmented_image = self.__crop_center(augmented_image, self.tensor_size, self.tensor_size)

        output_layer = 'loss:0'
        input_node = 'Placeholder:0'

        with tf.compat.v1.Session() as sess:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            predictions, = sess.run(prob_tensor, {input_node: [augmented_image]})

            # case 1 label:
            if len(predictions) == 1:
                if predictions[0] >= self.treshold_corner:
                    return predictions[0] , self.labels[0]
                else:
                    return predictions[0] , 'negative'

            # case more than 1 label:
            highest_probability_index = np.argmax(predictions)

            label_index = 0
            for p in predictions:
                truncated_probablity = np.float64(np.round(p, 8))
                # print(self.labels[label_index], truncated_probablity)
                label_index += 1
            if self.labels[highest_probability_index] == 'perfect':
                return True, self.labels[highest_probability_index]
            return predictions[highest_probability_index] , self.labels[highest_probability_index]


    def __import_model(self, model_file: str):
        with tf.compat.v1.gfile.FastGFile(model_file, 'rb') as f:
            self.graph_def.ParseFromString(f.read())
            tf.import_graph_def(self.graph_def, name='')

    def __import_labels(self, labels_file: str):
        with open(labels_file, 'rt') as lf:
            for l in lf:
                self.labels.append(l.strip())

    def load_labels(self, filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f.readlines()]


    def __convert_to_opencv(self, image):
        # RGB -> BGR conversion is performed as well.
        r, g, b = np.array(image).T
        opencv_image = np.array([b, g, r]).transpose()
        return opencv_image

    def __crop_center(self, img, cropx, cropy):
        h, w = img.shape[:2]
        x = w//2-(cropx//2)
        y = h//2-(cropy//2)
        return img[y:y+cropy, x:x+cropx]

    def resize_down_to_1600_max_dim(self,image):
        h, w = image.shape[:2]
        if (h < 1600 and w < 1600):
            return image

        new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
        return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

    def resize_to_256_square(self,image):
        h, w = image.shape[:2]
        return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

    def update_orientation(self,image):
        exif_orientation_tag = 0x0112
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if (exif != None and exif_orientation_tag in exif):
                orientation = exif.get(exif_orientation_tag, 1)
                # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
                orientation -= 1
                if orientation >= 4:
                    image = image.transpose(Image.TRANSPOSE)
                if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

class AzureCustomVision: 

    def __init__(self, parser, logs, output_path, model_name, datasets_dir,folder_with_dataset_name): 
        print("--- AzureCustomVision: init")

        print(parser.get('azureCustomVisionService', 'description'))

        # global user variables from config file:
        self.model_name = model_name
        self.datasets_dir = datasets_dir
        self.folder_with_dataset_name = folder_with_dataset_name
        self.output_path = output_path

        # parse from config file specific config for this service: 
        endpoint = parser.get('azureCustomVisionService', 'endpoint')
        training_key = parser.get('azureCustomVisionService', 'training_key')
        resource_id = parser.get('azureCustomVisionService', 'resource_id')
        general_compact_domain_id = parser.get('azureCustomVisionService', 'general_compact_domain_id')
        project_name = parser.get('azureCustomVisionService', 'project_name')

        self.project_name = project_name
        self.trainer = CustomVisionTrainingClient(training_key, endpoint=endpoint)
        self.generalCompactDomainId = general_compact_domain_id
        self.resource_id = resource_id

        self.logs = logs

        self.predictor = CustomVisionPredictionClient(training_key, endpoint=endpoint)

    def configure(self,results, times):
        print("--- AzureCustomVision: configure")
        start = time.time()

        copy_tree(self.datasets_dir + "/" +self.folder_with_dataset_name, "./services/azure_custom_vision/datasets/"+self.folder_with_dataset_name)

        print("Looking for the project...")

        for project in self.trainer.get_projects():
            print(project.name)
            if project.name == self.project_name:
                print("Project found!")
                self.project = project
                break

        project_photos = self.trainer.get_tagged_images(self.project.id)
        while len(project_photos)>0:
            print("Deleting old photos..." + str(len(project_photos))) 
            photos_to_delete = []
            for photo in project_photos:
                photos_to_delete.append(photo.id)
            # delete all photos from the project
            self.trainer.delete_images(self.project.id, photos_to_delete)
            project_photos = self.trainer.get_tagged_images(self.project.id)

        print("--- AzureCustomVision: configured")
        end = time.time()
        results['configured'] = True
        times['configure_time'] = round(end - start)

    def do_some_staff(self):
        print("--- AzureCustomVision: do some staff")


    def load_dataset(self,results, times):
        print("--- AzureCustomVision: load dataset")
        start = time.time()
        
        created_labels = self.trainer.get_tags(self.project.id)
        self.good_tag = None
        self.bad_tag = None
        for label in created_labels: 
            if label.name == 'good':
                print('--- good tag found')
                self.good_tag = label
            if label.name == 'bad':
                print('--- bad tag found')
                self.bad_tag = label

        if self.good_tag == None:
            print('--- create good tag')
            self.good_tag = self.trainer.create_tag(self.project.id, "good")
        if self.bad_tag == None:
            print('--- create bad tag')
            self.bad_tag = self.trainer.create_tag(self.project.id, "bad")

        directory1 = './services/azure_custom_vision/datasets/'+self.folder_with_dataset_name+'/good'
        directory2 = './services/azure_custom_vision/datasets/'+self.folder_with_dataset_name+'/bad'
        
        for root, directories, files in os.walk(directory1): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename)  
                print("Adding image: " + filename)
                with open(filepath, 'rb') as photo_file:
                    raw_photo = photo_file.read()
                    self.trainer.create_images_from_data(self.project.id, raw_photo, [self.good_tag.id])

        # crawling through directory and subdirectories 
        for root, directories, files in os.walk(directory2): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                print("Adding image: " + filename)
                with open(filepath, 'rb') as photo_file:
                    raw_photo = photo_file.read()
                    self.trainer.create_images_from_data(self.project.id, raw_photo, [self.bad_tag.id])

        print("--- Adding done!")

        print("--- AzureCustomVision: dataset loaded")
        end = time.time()
        
        results['dataset_loaded'] = True
        times['dataset_load_time'] = round(end - start)

    def train_model(self,results, times):
        print("--- AzureCustomVision: train model")
        start = time.time()


        print("--- Training...")
        # self.iteration = self.trainer.train_project(self.project.id)
        self.iteration = self.trainer.train_project(self.project.id,training_type='Advanced',reserved_budget_in_hours=2)
        while self.iteration.status == "Training":
            self.iteration = self.trainer.get_iteration(self.project.id, self.iteration.id)
            print("Training status: " + self.iteration.status)
            time.sleep(5)

        # The iteration is now trained. Make it the default project endpoint
        publish_iteration_name="model"+self.iteration.id # TODO: change this
        # The iteration is now trained. Make it the default project endpoint
        self.trainer.update_iteration(self.project.id, self.iteration.id,publish_iteration_name, is_default=True)
        print("--- Training done!")

        print("publich iteration ...")
        self.publish_iteration_name = "published_model_"+str(self.iteration.id)
        self.trainer.publish_iteration(self.project.id, self.iteration.id, self.publish_iteration_name, self.resource_id)
        print ("Done! publishing")


        end = time.time()
        print("--- AzureCustomVision: model has been trained")
        results['model_trained'] = True
        print("train model time",round(end - start))
        times['model_train_time'] = round(end - start)

    def download_model(self,results, times):
        print("--- AzureCustomVision: download model")
        start = time.time()

        # exports = self.trainer.get_exports(self.project.id, self.iteration.id)
        # if not exports:
        #     self.trainer.export_iteration(self.project.id, self.iteration.id, 'TensorFlow')
        #     print("Exporting...")
        #     exports = self.trainer.get_exports(self.project.id, self.iteration.id)
        #     url = exports[0].download_uri
        #     while url is None:
        #         url = exports[0].download_uri
        #         exports = self.trainer.get_exports(self.project.id, self.iteration.id)
        #     print("Exporting done!")
        # url = exports[0].download_uri
        # try:
        #     print("Downloading...")
        #     urllib.request.urlretrieve(url, "model.zip")
        #     os.mkdir('./services/azure_custom_vision/models/'+self.iteration.id)
        #     with zipfile.ZipFile('model.zip') as zip_file:
        #         zip_file.extractall('./services/azure_custom_vision/models/'+self.iteration.id)
        # except URLError as e:
        #     if hasattr(e, 'reason'):
        #         print('We failed to reach a server.')
        #         print('Reason: ', e.reason)
        #     elif hasattr(e, 'code'):
        #         print('The server couldn\'t fulfill the request.')
        #         print('Error code: ', e.code)
        # else:
        #     print("Downloading done!")

        end = time.time()
        results['model_downloaded'] = True
        times['model_download_time'] = round(end-start)


    def test_model(self,results, times, output):
        print("--- AzureCustomVision: test model")
        start = time.time()

        ##uncoment this if u test only this function: 
        # it_id = '6aa7f6ac-7818-4c38-9043-d01f403b4d72'
        # predict = Predict()
        # predict.init('./services/azure_custom_vision/models/'+it_id)

        ##this is fot usage of local model
        # predict = Predict()
        # predict.init('./services/azure_custom_vision/models/'+self.iteration.id)

        # ## uncoment this if u want to test only this function in cloud prediction: 
        # print("publich iteration ...")
        # self.publish_iteration_name = "published_model_3b757e27-a41f-41b1-b593-ed520d6ed756"
        # self.trainer.publish_iteration(self.project.id, '3b757e27-a41f-41b1-b593-ed520d6ed756', self.publish_iteration_name, self.resource_id)
        # print ("Done! publishing")

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

        print("--- AzureCustomVision: testing data with good labels ...")
        for root, directories, files in os.walk(directory1): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 

                # for usage of local model
                # print("precidtion for: ", filepath)
                # res, predicted_class = predict.run(filepath)



                with open(filepath, "rb") as image_contents:
                    res = self.predictor.classify_image(self.project.id, self.publish_iteration_name, image_contents.read())
                if res.predictions[0].tag_name == "good":
                    if float(res.predictions[0].probability) > float(res.predictions[1].probability):
                        predicted_class = "good"
                    else: 
                        predicted_class = "bad"
                elif res.predictions[0].tag_name == "bad":
                    if float(res.predictions[0].probability) > float(res.predictions[1].probability):
                        predicted_class = "bad"
                    else: 
                        predicted_class = "good"
                print("good",filename, predicted_class)



                good_number +=1
                if predicted_class == "good":
                    correct_good += 1
                elif predicted_class == "bad":
                    incorrect_bad += 1

                



        print("--- AzureCustomVision: testing data with bad labels ...")
        for root, directories, files in os.walk(directory2): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 

                # for usage of local model
                # print("precidtion for: ", filepath)
                # res, predicted_class = predict.run(filepath)


                with open(filepath, "rb") as image_contents:
                    res = self.predictor.classify_image(self.project.id, self.publish_iteration_name, image_contents.read())
                if res.predictions[0].tag_name == "good":
                    if float(res.predictions[0].probability) > float(res.predictions[1].probability):
                        predicted_class = "good"
                    else: 
                        predicted_class = "bad"
                elif res.predictions[0].tag_name == "bad":
                    if float(res.predictions[0].probability) > float(res.predictions[1].probability):
                        predicted_class = "bad"
                    else: 
                        predicted_class = "good"
                print("bad",filename, predicted_class)
                

        
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

        print("--- AzureCustomVision: model tested")

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
    
    def do_cleaning(self,results, times):
        print("--- AzureCustomVision: do cleaning")
        start = time.time()

        #TODO delete oldest iteration in 5 items buffer
        print("--- AzureCustomVision: unpublishing")
        self.trainer.unpublish_iteration(self.project.id, self.iteration.id)
        print("--- AzureCustomVision: done unpublished")

        end = time.time()
        # print("Photos deleted!")

        results['done_cleaning'] = True
        times['do_cleaning_time'] = round(end - start)

        #TODO:

        # usun interacje jesli jest ich za duzo. 
    
 
        print("--- AzureCustomVision: cleaned up")




