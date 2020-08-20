import base64
import datetime
import json
import struct
import threading
import time
import os
import csv
import boto3
from datetime import datetime
from distutils.dir_util import copy_tree

import cv2

##TODO do przetestowania w całości , dostało przetestowanie dodawanie datasetu oraz trenowanie osobno. 
##nie zostało przetestowane sprawdzanie prezycji. 

class AmazonAWS:

    def __init__(self, parser, logs, output_path, model_name, datasets_dir,folder_with_dataset_name):
        print("--- AmazonAWS: init")
        print(parser.get('amazonService', 'description'))

        # global user variables from config file:
        self.model_name = model_name
        self.datasets_dir = datasets_dir
        self.folder_with_dataset_name = folder_with_dataset_name
        self.output_path = output_path

        # parse from config file specific config for this service: 
        credentials_file_path = parser.get('amazonService', 'credentials_file_path')
        region_name = parser.get('amazonService', 'region_name')

        with open(credentials_file_path,'r') as input:
            next(input)
            reader = csv.reader(input)
            for line in reader: 
                print(line[0])
                access_key_id = line[0]
                print(line[1])
                secret_access_key = line[1]

        self.client = boto3.client('rekognition', region_name=region_name, aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
        self.clientS3 = boto3.client('s3', region_name=region_name, aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
        self.s3 = boto3.resource('s3')

        self.logs = logs

    def configure(self, results,times):
        print("--- AmazonAWS: configure")
        start = time.time()
        
        date = datetime.now().strftime("%d-0%m-%Y_%H-%M-%S")
 
        response = self.client.describe_projects( MaxResults=5 )
        if len(response['ProjectDescriptions']) == 0:
            print("Error, no projects found, you can create one in https://us-east-2.console.aws.amazon.com/rekognition/custom-labels?#/projects")
            return
        print('Project Arn',response['ProjectDescriptions'][0]['ProjectArn'])
        self.project_arn = response['ProjectDescriptions'][0]['ProjectArn']
        self.version_name = self.model_name+"_"+date

        response = self.clientS3.list_buckets()
        if len(response['Buckets']) == 0:
            print("Error, no buckets in project, you can create one in https://console.aws.amazon.com/s3/home")
            return
        print('Bucket name',response['Buckets'][0]['Name'])
        self.bucket_name = response['Buckets'][0]['Name']

        
        self.manifest_folder = 'datasets/'+self.folder_with_dataset_name+'/manifests/output.manifest'
        self.output_folder = 'datasets/'+self.folder_with_dataset_name+'/models'

        self.output_config = json.loads('{"S3Bucket":"'+self.bucket_name+'", "S3KeyPrefix":"'+self.output_folder+'"}')
        self.training_dataset= json.loads('{"Assets": [{ "GroundTruthManifest": { "S3Object": { "Bucket": "'+self.bucket_name+'", "Name": "'+self.manifest_folder+'" } } } ] }')
        self.testing_dataset= json.loads('{"AutoCreate":true}')


        print("--- AmazonAWS: configured")
        end = time.time()
        results['configured'] = True
        times['configure_time'] = round(end - start)

    def get_manifest_record(self,label,filename):
        date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        record = {
            "source-ref":"s3://"+self.bucket_name+"/datasets/"+self.folder_with_dataset_name+"/"+label+"/"+filename,
            self.folder_with_dataset_name+"_"+label:1,
            self.folder_with_dataset_name+"_"+label+"-metadata":
                {
                    "confidence":1,
                    "job-name":"labeling-job/"+self.folder_with_dataset_name+"_"+label,
                    "class-name":label,
                    "human-annotated":"yes",
                    "creation-date":date,
                    "type":"groundtruth/image-classification"
                }
            }
        return json.dumps(record)+"\n"

    def load_dataset(self,results, times):
        print("--- AmazonAWS: load dataset")
        start = time.time()

        copy_tree(self.datasets_dir + "/" +self.folder_with_dataset_name, "./services/amazon_aws/datasets/"+self.folder_with_dataset_name)


        directory1 = './services/amazon_aws/datasets/'+self.folder_with_dataset_name+'/good'
        directory2 = './services/amazon_aws/datasets/'+self.folder_with_dataset_name+'/bad'
        try:
            os.mkdir('./services/amazon_aws/datasets/'+self.folder_with_dataset_name+'/manifests')
        except:
            print("directory has been created")
        manifest = './services/amazon_aws/datasets/'+self.folder_with_dataset_name+'/manifests/output.manifest'
        try:
            os.remove(manifest)
        except:
            print("manifest cant be deleted")

        
        # crawling through directory and subdirectories 
        for root, directories, files in os.walk(directory1): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename)  
                print("Adding", filename)
                self.s3.Object(self.bucket_name, 'datasets/'+self.folder_with_dataset_name+'/good/'+filename).upload_file(filepath)
                with open(manifest, 'a+') as the_file:
                    the_file.write(self.get_manifest_record('good',filename))
                

        # crawling through directory and subdirectories 
        for root, directories, files in os.walk(directory2): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                print("Adding", filename)
                self.s3.Object(self.bucket_name, 'datasets/'+self.folder_with_dataset_name+'/bad/'+filename).upload_file(filepath)
                with open(manifest, 'a+') as the_file:
                    the_file.write(self.get_manifest_record('bad',filename))


        #add manifest file
        self.s3.Object(self.bucket_name, 'datasets/'+self.folder_with_dataset_name+'/manifests/output.manifest').upload_file(manifest)

        print("--- AmazonAWS: dataset loaded")
        end = time.time()
        results['dataset_loaded'] = True
        times['dataset_load_time'] = round(end - start)


    def train_model(self,results, times):
        print("--- AmazonAWS: train model")
        start = time.time()
    
        print('--- AmazonAWS: Starting training of: ' + self.version_name)
        
    
        response=self.client.create_project_version(ProjectArn=self.project_arn, 
            VersionName=self.version_name,
            OutputConfig=self.output_config,
            TrainingData=self.training_dataset,
            TestingData=self.testing_dataset)

        # Wait for the project version training to complete
        project_version_training_completed_waiter = self.client.get_waiter('project_version_training_completed')
        project_version_training_completed_waiter.wait(ProjectArn=self.project_arn,
        VersionNames=[self.version_name])
    
        #Get the completion status
        describe_response=self.client.describe_project_versions(ProjectArn=self.project_arn,
            VersionNames=[self.version_name])
        for model in describe_response['ProjectVersionDescriptions']:
            print("Status: " + model['Status'])
            print("Message: " + model['StatusMessage']) 
            self.model_arn = model['ProjectVersionArn']

        print('--- AmazonAWS: Trining Done...')


        print("--- AmazonAWS: model has been trained")
        end = time.time()
        print("train model time",round(end - start))
        results['model_trained'] = True
        times['model_train_time'] = round(end - start)

    def download_model(self,results,times):
        print("--- AmazonAWS: download model")

        start = time.time()

        end = time.time()

        results['model_downloaded'] = True
        times['model_download_time'] = round(end-start)

    def test_model(self,results, times, output):
        print("--- AmazonAWS: test model")
        start = time.time()


        min_inference_units = 1

        ## test (uncomment this if u are running only this function)
        # self.version_name  = 'model1_27-005-2020_18-50-57'

        # #Get the running status
        # describe_response=self.client.describe_project_versions(ProjectArn=self.project_arn,
        #     VersionNames=[self.version_name])
        # for model in describe_response['ProjectVersionDescriptions']:
        #     print("Status: " + model['Status'])
        #     print("Message: " + model['StatusMessage']) 
        #     self.model_arn = model['ProjectVersionArn']
        # self.model_arn = 'arn:aws:rekognition:us-east-2:925392216157:project/Inzynierka3/version/model1_08-007-2020_22-14-43/1594240435605'
        # # /test

        # Start the model
        try:
            print('--- AmazonAWS: Starting model: ' + self.model_arn)
            response=self.client.start_project_version(ProjectVersionArn=self.model_arn, MinInferenceUnits=min_inference_units)
            # Wait for the model to be in the running state
            project_version_running_waiter = self.client.get_waiter('project_version_running')
            project_version_running_waiter.wait(ProjectArn=self.project_arn, VersionNames=[self.version_name])

            #Get the running status
            describe_response=self.client.describe_project_versions(ProjectArn=self.project_arn,
                VersionNames=[self.version_name])
            for model in describe_response['ProjectVersionDescriptions']:
                print("Status: " + model['Status'])
                print("Message: " + model['StatusMessage']) 
            print('--- AmazonAWS: Starting Model done...')
        except Exception as e: 
            print("EXCEPTED AMAZON IN Turning on model : ------------", e)
            time.sleep(500)



        print('--- AmazonAWS: Testing Model ...')
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
        
        try: 
            print("--- AmazonAWS: testing data with good labels ...")
            for root, directories, files in os.walk(directory1): 
                for filename in files: 
                    # join the two strings in order to form the full filepath. 
                    filepath = os.path.join(root, filename) 
                    # print("precidtion for: ", filepath)
                    imCv = cv2.imread(filepath)
                    rows,cols, depth = imCv.shape
                    if rows < 80 or cols<80:
                        print("too small photo")
                        continue

                    with open(filepath, "rb") as img_file:
                        content = img_file.read()  
                        response = self.client.detect_custom_labels(Image={'Bytes': content}, MaxResults= 2, MinConfidence= 0, ProjectVersionArn= self.model_arn)

                    
                    result_label = ""
                    probability = 0
                    for customLabel in response['CustomLabels']:
                        if customLabel['Confidence'] > probability:
                            probability = customLabel['Confidence']
                            result_label = customLabel['Name']

                    #calc: 
                    print("good",filename, result_label)
                    good_number +=1
                    if result_label == "good":
                        correct_good += 1
                    elif result_label == "bad":
                        incorrect_bad += 1

                    

            print("--- AmazonAWS: testing data with good labels ...")
            for root, directories, files in os.walk(directory2): 
                for filename in files: 
                    # join the two strings in order to form the full filepath. 
                    filepath = os.path.join(root, filename) 
                    # print("precidtion for: ", filepath)
                    imCv = cv2.imread(filepath)
                    rows,cols, depth = imCv.shape
                    if rows < 80 or cols<80:
                        print("too small photo")
                        continue

                    with open(filepath, "rb") as img_file:
                        content = img_file.read()  
                        response = self.client.detect_custom_labels(Image={'Bytes': content}, MaxResults= 2, MinConfidence= 0, ProjectVersionArn= self.model_arn)

                    
                    result_label = ""
                    probability = 0
                    for customLabel in response['CustomLabels']:
                        if customLabel['Confidence'] > probability:
                            probability = customLabel['Confidence']
                            result_label = customLabel['Name']

                    #calc: 
                    print("bad",filename, result_label)
                    bad_number +=1
                    if result_label == "bad":
                        correct_bad += 1
                    elif result_label == "good":
                        incorrect_good += 1
        except Exception as e: 
            print('excepted AMAZON', e)
            print("stopping model")
            response=self.client.stop_project_version(ProjectVersionArn=self.model_arn)
            print("stopped model", response)
            return


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

        print("stopping model")
        response=self.client.stop_project_version(ProjectVersionArn=self.model_arn)
        print("stopped model", response)

        print("--- AmazonAWS: model tested")

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
        print("--- AmazonAWS: clear staff")
        start = time.time()

        end = time.time()

        results['done_cleaning'] = True
        times['do_cleaning_time'] = round(end - start)
    
    def do_some_staff(self):
        print("--- AmazonAWS: do_some_staff")
