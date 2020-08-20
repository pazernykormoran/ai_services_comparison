import os
import sys
import time
import json
import csv
from datetime import datetime


from google.oauth2 import service_account
import googleapiclient.discovery

from google.cloud import automl, storage


class GoogleCloudVision:

    def __init__(self, parser, logs, output_path, model_name, datasets_dir,folder_with_dataset_name):
        print("--- GoogleCloudVision: init")
        
         
        print(parser.get('googleService', 'description'))

        # global user variables from config file:
        self.model_name = model_name
        self.datasets_dir = datasets_dir
        self.folder_with_dataset_name = folder_with_dataset_name
        self.output_path = output_path
        
        # parse from config file specific config for this service: 
        project_id = parser.get('googleService', 'project_id')
        api_endpoint = parser.get('googleService', 'api_endpoint')
        bucket_name = parser.get('googleService', 'bucket_name')
        project_location = parser.get('googleService', 'project_location')
        model_type = parser.get('googleService', 'model_type')
        google_application_credentials = parser.get('googleService', 'google_application_credentials')

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=google_application_credentials

        # TODO kopionwanie fotek do folderu i tam labelowanie ich czy coś tam
        # TODO zmienic project location wszedzie na zmienną

        self.dataset_id = ""
        self.project_id = project_id
        client_options = {'api_endpoint': api_endpoint}
        self.bucket_name = bucket_name
        self.client = automl.AutoMlClient(client_options=client_options)
        self.storage_client = storage.Client()
        self.credentials = service_account.Credentials.from_service_account_file(
                filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
                scopes=['https://www.googleapis.com/auth/cloud-platform'])
        
        self.project_location_name = project_location
        self.project_location = self.client.location_path(self.project_id, project_location)
        self.model_type = model_type
        
        self.logs = logs
        

    def configure(self, results, times):
        print("--- GoogleCloudVision: configure")
        start = time.time()



        os.remove('./services/google_cloud_vision/data.csv')
        directory1 = os.path.join(self.datasets_dir, self.folder_with_dataset_name, "good")
        directory2 = os.path.join(self.datasets_dir, self.folder_with_dataset_name, "bad")
        
        # crawling through directory and subdirectories 
        for root, directories, files in os.walk(directory1): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename)  
                with open('./services/google_cloud_vision/data.csv','a+') as fd:
                    fd.write('gs://'+self.bucket_name+'/'+self.folder_with_dataset_name+'/good/'+filename + ',good\n')
                

        # crawling through directory and subdirectories 
        for root, directories, files in os.walk(directory2): 
            for filename in files: 
                # join the two strings in order to form the full filepath. 
                filepath = os.path.join(root, filename) 
                with open('./services/google_cloud_vision/data.csv','a+') as fd:
                    fd.write('gs://'+self.bucket_name+'/'+self.folder_with_dataset_name+'/bad/'+filename + ',bad\n')



        end = time.time()
        print("--- GoogleCloudVision: configured")
        results['configured'] = True
        times['configure_time'] = round(end - start)


    def load_dataset(self,results, times):
        print("--- GoogleCloudVision: load_dataset")
        start = time.time()



        print("Exporting data to google cloud")
        os.system("gsutil -m cp -R " + os.path.join(self.datasets_dir,self.folder_with_dataset_name,"good") + " gs://"+self.bucket_name+"/"+self.folder_with_dataset_name+"/")
        os.system("gsutil -m cp -R " + os.path.join(self.datasets_dir,self.folder_with_dataset_name,"bad") + " gs://"+self.bucket_name+"/"+self.folder_with_dataset_name+"/")
        os.system("gsutil -m cp ./services/google_cloud_vision/data.csv gs://"+self.bucket_name+"/"+self.folder_with_dataset_name+"/data.csv")

        print("Creating dataset")
        dataset_name = "dataset1"

        # A resource that represents Google Cloud Platform location.
        metadata = automl.types.ImageClassificationDatasetMetadata(
            classification_type=automl.enums.ClassificationType.MULTICLASS
        )
        dataset = automl.types.Dataset(
            display_name=dataset_name,
            image_classification_dataset_metadata=metadata,
        )

        # Create a dataset with the dataset metadata in the region.
        response = self.client.create_dataset(self.project_location, dataset)

        created_dataset = response.result()
        
        print("Dataset creeated")
        # Display the dataset information
        print("Dataset name: {}".format(created_dataset.name))
        print("Dataset id: {}".format(created_dataset.name.split("/")[-1]))

        print("Export images from google cloud to dataset using csv file")
        self.dataset_id = format(created_dataset.name.split("/")[-1])
        path = "gs://"+self.bucket_name+"/"+self.folder_with_dataset_name+ "/data.csv"

        # Get the full path of the dataset.
        dataset_full_id = self.client.dataset_path(
            self.project_id, self.project_location_name, self.dataset_id
        )
        # Get the multiple Google Cloud Storage URIs
        input_uris = path.split(",")
        gcs_source = automl.types.GcsSource(input_uris=input_uris)
        input_config = automl.types.InputConfig(gcs_source=gcs_source)
        # Import data from the input URI
        response = self.client.import_data(dataset_full_id, input_config)

        print("Processing import...")
        print("Data imported. {}".format(response.result()))



        end = time.time()
        print("--- GoogleCloudVision: dataset loaded")
        results['dataset_loaded'] = True
        times['dataset_load_time'] = round(end - start)

    def train_model(self,results, times):
        print("--- GoogleCloudVision: train model")
        start = time.time()


        # Leave model unset to use the default base model provided by Google
        # train_budget_milli_node_hours: The actual train_cost will be equal or
        # less than this value.
        # https://cloud.google.com/automl/docs/reference/rpc/google.cloud.automl.v1#imageclassificationmodelmetadata
        # 1000 do 100 000 oraz od 8000 do 800 000 dla typu modelu "cloud" mili node hours, czyli 1h lub 8h do 100 h lub 800h dzialania nodow ktore moga dzialac rownoczesnie przy bardziej zlozonych taskach.
        metadata = automl.types.ImageClassificationModelMetadata(
            train_budget_milli_node_hours=24000,
            model_type =self.model_type 
        )
        model = automl.types.Model(
            display_name=self.model_name,
            dataset_id=self.dataset_id,
            image_classification_model_metadata=metadata,
        )

        # Create a model with the model metadata in the region.
        response = self.client.create_model(self.project_location, model)

        print("Training operation name: {}".format(response.operation.name))
        print("Training started...")

        self.check_training_status()


        end = time.time()
        results['model_trained'] = True
        print("train model time",round(end - start))
        times['model_train_time'] = round(end - start)

    def check_training_status(self):

        while True: 
            time.sleep(20)

            response = self.client.transport._operations_client.list_operations(
                self.project_location, ""
            )

            counter = 0
            print("--- GoogleCloudVision: training ...", datetime.now().strftime("%d-%m-%Y_%H:%M:%S"))
            for operation in response:
                if operation.done == True: 
                    continue
                counter +=1 
                
            if counter == 0:
                print("--- GoogleCloudVision: Training has been finished")
                break

    def download_model(self,results, times):
        print("--- GoogleCloudVision: download model")
        start = time.time()

        # parent = self.client.location_path(self.project_id, self.project_location_name)
        # for element in self.client.list_models(parent):
        #     self.model_id = element.name[element.name.find("ICN"):]

        # #TODO pobieraj dobry model
        # #TODO model format ogarnij

        # data = {
        #             'outputConfig': {
        #                 'modelFormat': 'tflite',
        #                 'gcsDestination': {
        #                 'outputUriPrefix': 'gs://' + self.bucket_name + '/'+ self.folder_with_dataset_name +'/models/'
        #                 },
        #             }
        #         }

        # f= open("./services/google_cloud_vision/request.json","w+")
        # f.write(json.dumps(data))
        # f.close() 

        # os.system('curl -X POST \
        #     -H \"Authorization: Bearer \"$(gcloud auth application-default print-access-token) \
        #     -H \"Content-Type: application/json; charset=utf-8\" \
        #     -d @services/google_cloud_vision/request.json \
        #     https://automl.googleapis.com/v1/projects/inzynierka1/locations/'+self.project_location_name+'/models/ICN'+self.model_id+':export')

        # blobs = self.storage_client.list_blobs(self.bucket_name, prefix="" + self.folder_with_dataset_name + "/models/model-export/icn/")

        # for blob in blobs:
        #     print("downloading: " + blob.name)
        #     task = "gsutil cp gs://" + self.bucket_name + "/" + blob.name + " " + "./services/google_cloud_vision/models"+blob.name[blob.name.find("icn")+3:]
        #     os.system(task)



        end = time.time()
        results['model_downloaded'] = True
        times['model_download_time'] = round(end-start)

    def do_cleaning(self,results,times):
        print("--- GoogleCloudVision: do cleaning")
        start = time.time()

        # print("Deleting model")
        # name = self.client.model_path(self.project_id, 'us-central1', self.model_id)
        # response = self.client.delete_model(name)
        # print(response)

        # name = self.client.dataset_path(self.project_id, 'us-central1', self.dataset_id)
        # response = self.client.delete_dataset(name)
        # print(response)

        end = time.time()
        results['done_cleaning'] = True
        times['do_cleaning_time'] = round(end - start)

    
    def do_some_staff(self):
        print("--- GoogleCloudVision: do_some_staff")


    def test_model(self,results, times, output):
        print("--- GoogleCloudVision: test model")
        
        # #uncomment line under if u want test only this function
        # self.dataset_id = "ICN6460999705160581120"
        

        ### prepare model
        start = time.time()
        model_id = ""
        prediction_client = automl.PredictionServiceClient()
        parent = self.client.location_path(self.project_id, self.project_location_name)
        for element in self.client.list_models(parent):
            if element.dataset_id == self.dataset_id:
                model_id = element.name

        print('model id ', model_id)

        # uncomment to turn on deploying model 
        response = self.client.deploy_model(model_id)
        print("Model deployment finished. {}".format(response.result()))


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

        try:

            ### walk through directories
            directory1 = os.path.join(self.datasets_dir, self.folder_with_dataset_name, "test", "good")
            directory2 = os.path.join(self.datasets_dir, self.folder_with_dataset_name, "test", "bad")

            print("--- GoogleCloudVision: testing data with good labels ...")
            for root, directories, files in os.walk(directory1): 
                for filename in files: 
                    # join the two strings in order to form the full filepath. 
                    filepath = os.path.join(root, filename) 
                    # print("precidtion for: ", filepath)
                    with open(filepath, "rb") as content_file:
                        content = content_file.read()
                    image = automl.types.Image(image_bytes=content)
                    payload = automl.types.ExamplePayload(image=image)
                    response = prediction_client.predict(model_id, payload)
                    # print("Prediction results:")
                    for result in response.payload:
                        print("1. Predicted class name: {}".format(result.display_name))
                        print("2. Predicted class score: {}".format(result.classification.score))

                    if len(response.payload) > 1: 
                        print("WARNINIG !")
                        print("WARNINIG.. !")
                        print("WARNINIG.... ! 2 labels")
                        continue

                    #calc: 
                    print("good",filename, response.payload[0].display_name)
                    good_number +=1
                    if len(response.payload) == 1:
                        if response.payload[0].display_name == "good":
                            correct_good += 1
                        elif response.payload[0].display_name == "bad":
                            incorrect_bad += 1

                    



            print("--- GoogleCloudVision: testing data with bad labels ...")
            for root, directories, files in os.walk(directory2): 
                for filename in files: 
                    # join the two strings in order to form the full filepath. 
                    filepath = os.path.join(root, filename) 
                    # print("precidtion for: ", filepath)
                    with open(filepath, "rb") as content_file:
                        content = content_file.read()
                    image = automl.types.Image(image_bytes=content)
                    payload = automl.types.ExamplePayload(image=image)
                    response = prediction_client.predict(model_id, payload)
                    # print("Prediction results:")
                    for result in response.payload:
                        print("1. Predicted class name: {}".format(result.display_name))
                        print("2. Predicted class score: {}".format(result.classification.score))

                    if len(response.payload) > 1: 
                        print("WARNINIG !")
                        print("WARNINIG.. !")
                        print("WARNINIG.... ! 2 labels")
                        continue

                    #calc: 
                    print("bad",filename, response.payload[0].display_name)
                    bad_number +=1
                    if len(response.payload) == 1:
                        if response.payload[0].display_name == "bad":
                            correct_bad += 1
                        elif response.payload[0].display_name == "good":
                            incorrect_good += 1
        except Exception as e: 
            print('google excepted ', e)
            response = self.client.undeploy_model(model_id)
            print("Model UNdeployed. {}".format(response.result()))
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

        response = self.client.undeploy_model(model_id)
        print("Model UNdeployed. {}".format(response.result()))

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









