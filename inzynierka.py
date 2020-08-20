import os
import threading
import time
import json
from prettytable import PrettyTable
import configparser
from datetime import datetime
import requests

# import from local imports
from imports.logs import Logs
from imports.service import Service

# import services
from services.amazon_aws.amazon_aws import AmazonAWS
from services.google_cloud_vision.google_cloud_vision import GoogleCloudVision
from services.ibm_watson_visual_recognition.ibm_watson_visual_recognition import IBMWatsonVisualRecognition
from services.clarifai.clarifai import Clarifai
from services.vize_ai.vize_ai import VizeAi
from services.azure_custom_vision.azure_custom_vision import AzureCustomVision
from services.nano_nets.nano_nets import NanoNets
from services.image_ai.image_ai import ImageAI

print("")
print(" #### HELLO IN AI VISION SERVICES COMPARISON ####")
print("")

### BIG TODO: 
# - rozbudowac skrypt aby dzialal na dowolna ilosc labeli w datasecie oraz o netagive photos tak jak w skrypcie to zrobilem.
#    ogolnie duzo mozna wziac ze skryptu jesli chodzi o liczenie parametrow modelu

### TODO Must be done: 
# - zmienic zeby wszedzie byly 2 labele a nie jeden label i negative photos. (Ta sama konwencja i trzeba sie tego trzymac. )
    # do przerobienia: clarifai, ibm

# TODO zastanowic sie nad loggerem co logowac a czego nie. W razie czego relpace wlacza sie tam gdzie jest ctrl+f
# TODO przerobic wszystkie łączenie ścieżek na os.path.join. 
# TODO dopisac TEST WALIDUJACY 
       # czy dobrze ustawiles wsyzstko i dobre sciezki sa w datasecie. ( z foldername trzbea usunac ukosniki na pewno)
       # ile jest modeli w ibm watson.
# TODO google deploy model in train model
# TODO postaraj sie pobierac datasety bez kopiowania ich do siebie albo chociaz potem usuwaj ten folder.
# TODO dodac usuwanie stworzonych modeli tak zeby po dzialaniu skryptu nic nie zostalo
# TODO mozna dodac porownywanie metrics stworzonych przez serwisy z metrics moimi
# TODO dodać wyswietlanie wszystkich przewidywan tak jak w skrypcie 
# TODO zapisz w script autput wszystkie modele i po zakonczeniu pracy programow wyczysz ich wlasne foldery z datasetami jesli byly niezbedne


### list config: 
print("Your Configuration: ")
print('')
parser = configparser.ConfigParser()
parser.read('/home/george/.inzynierka-credentials/config.cfg')
for sect in parser.sections():
    print('Section:', sect)
    for k,v in parser.items(sect):
        print(' {} = {}'.format(k,v))
    print()


### parses user variables: 
model_name = parser.get('user_parameters', 'model_name')
datasets_dir = parser.get('user_parameters', 'datasets_dir')
folder_with_dataset_name = parser.get('user_parameters', 'folder_with_dataset_name')

### create output folder in dataset folder: 
output_folder = "script_output_"+datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
output_path = os.path.join(datasets_dir,folder_with_dataset_name,output_folder)
os.mkdir(output_path)

### check internet connection
r = requests.get('https://www.google.com/')
if r.status_code is not 200: 
    print('Check internet connection')
    exit()

### configure logger: 
logs = Logs(datasets_dir,folder_with_dataset_name, datetime.now().strftime("%d-%m-%Y_%H:%M:%S"),output_folder)

### test logger: 
logs.print('logger test')


logs.print(" ")
logs.print('##########################  CREATE SERVICES [not parallel]  #######################################')
logs.print(" ")

services = []
if parser.get('enabled_services', 'amazonService') == 'True':
    try: 
        amazonService = Service("Amazon", AmazonAWS(parser, logs, output_path, model_name,datasets_dir,folder_with_dataset_name))
        amazonService.results['initialized'] = True
    except Exception as e:
        amazonService.results['initialized'] = False
        print(e)
    services.append(amazonService)
if parser.get('enabled_services', 'googleService') == 'True':
    try: 
        googleService = Service("Google", GoogleCloudVision(parser, logs, output_path, model_name,datasets_dir,folder_with_dataset_name))
        googleService.results['initialized'] = True
    except Exception as e:
        googleService.results['initialized'] = False
        print(e)
    services.append(googleService)
if parser.get('enabled_services', 'IBMService') == 'True':
    try: 
        IBMService = Service("IBM", IBMWatsonVisualRecognition(parser, logs, output_path, model_name,datasets_dir,folder_with_dataset_name))
        IBMService.results['initialized'] = True
    except Exception as e:
        IBMService.results['initialized'] = False
        print(e)
    services.append(IBMService)
if parser.get('enabled_services', 'clarifaiService') == 'True':
    try: 
        clarifaiService = Service("ClarifAI", Clarifai(parser, logs, output_path, model_name,datasets_dir,folder_with_dataset_name))
        clarifaiService.results['initialized'] = True
    except Exception as e:
        clarifaiService.results['initialized'] = False
        print(e)
    services.append(clarifaiService)
if parser.get('enabled_services', 'vizeAiService') == 'True':
    try: 
        vizeAiService = Service("VizeAi", VizeAi(parser, logs, output_path, model_name,datasets_dir,folder_with_dataset_name))
        vizeAiService.results['initialized'] = True
    except Exception as e:
        vizeAiService.results['initialized'] = False
        print(e)
    services.append(vizeAiService)
if parser.get('enabled_services', 'azureCustomVisionService') == 'True':
    try: 
        azureCustomVisionService = Service("Azure Custom Vision", AzureCustomVision(parser, logs, output_path, model_name,datasets_dir,folder_with_dataset_name))
        azureCustomVisionService.results['initialized'] = True
    except Exception as e:
        azureCustomVisionService.results['initialized'] = False
        print(e)
    services.append(azureCustomVisionService)
if parser.get('enabled_services', 'nanoNetsService') == 'True':
    try: 
        nanoNetsService = Service("NanoNets", NanoNets(parser, logs, output_path, model_name,datasets_dir,folder_with_dataset_name))
        nanoNetsService.results['initialized'] = True
    except Exception as e:
        nanoNetsService.results['initialized'] = False
        print(e)
    services.append(nanoNetsService)
if parser.get('enabled_services', 'imageAIService') == 'True':
    try: 
        imageAIService = Service("ImageAI", ImageAI(parser, logs, output_path, model_name,datasets_dir,folder_with_dataset_name))
        imageAIService.results['initialized'] = True
    except Exception as e:
        imageAIService.results['initialized'] = False
        print(e)
    services.append(imageAIService)

for service in services:
    if service.results['initialized']:
        logs.print('Succesfully initialized for service: ', service.name)
    else:
        service.error['status'] = True
        service.error['text'].append('ERROR in initialize, service rejected from comparison')
        logs.print(str(service.error['text']))

logs.print(" ")
logs.print('##########################  CONFIGURE [parallel]  #######################################')
logs.print(" ")

for service in services:
    if service.error['status'] is not True:
        service.configureThread.start()

for service in services:
    if service.error['status'] is not True:
        service.configureThread.join()
        # results = service.configureThread[1]
        if service.results['configured']:
            logs.print('Succesfully configured service: ', service.name)
        else:
            service.error['status'] = True
            service.error['text'].append('ERROR in configure, service rejected from comparison')
            logs.print(str(service.error['text']))

logs.print(" ")
logs.print('##########################  LOAD DATASET [not parallel] #######################################')
logs.print(" ")

for service in services:
    if service.error['status'] is not True:
        try: 
            service.service.load_dataset(service.results, service.times)
        except Exception as e:
            logs.print("Encountered error while loading dataset for service:", service.name)
            service.results['dataset_loaded'] = False
            print(e)


for service in services:
    if service.error['status'] is not True:
        if service.results['dataset_loaded']:
            logs.print('Succesfully loaded dataset for service: ', service.name)
        else:
            service.error['status'] = True
            service.error['text'].append('ERROR in loaded dataset, service rejected from comparison')
            logs.print(str(service.error['text']))
            

logs.print(" ")
logs.print('##########################  TRAINING [parallel] #######################################')
logs.print(" ")

for service in services:
    if service.error['status'] is not True:
        service.trainModelThread.start()

for service in services:
    if service.error['status'] is not True:
        service.trainModelThread.join()
        if service.results['model_trained']:
            logs.print('Succesfully trained model for service: ', service.name)
        else:
            service.error['status'] = True
            service.error['text'].append('ERROR in training, service rejected from comparison')
            logs.print(str(service.error['text']))

logs.print(" ")
logs.print('##########################  DOWNLOAD [not parallel] #######################################')
logs.print(" ")

for service in services:
    if service.error['status'] is not True:
        try: 
            service.service.download_model(service.results, service.times)
        except Exception as e:
            logs.print("Encountered error while downloading model for service:", service.name)
            service.results['model_downloaded'] = False
            print(e)

for service in services:
    if service.error['status'] is not True:
        if service.results['model_downloaded']:
            logs.print('Succesfully downloaded model for service: ', service.name)
        else:
            service.error['status'] = True
            service.error['text'].append('ERROR in downloading, service rejected from comparison')
            logs.print(str(service.error['text']))

logs.print(" ")
logs.print('##########################  TESTING [not parallel] #######################################')
logs.print(" ")

# #uncomment this and thread fucntion in service class if you want to test model paralelly
# for service in services:
#     if service.error['status'] is not True:
#         service.testModelThread.start()

# for service in services:
#     if service.error['status'] is not True:
#         service.testModelThread.join()
#         if service.results['model_tested']:
#             logs.print('Succesfully tested model for service: ', service.name)
#         else:
#             service.error['status'] = True
#             service.error['text'].append('ERROR in testing, service rejected from comparison')
#             logs.print(str(service.error['text']))



for service in services:
    if service.error['status'] is not True:
        try: 
            service.service.test_model(service.results, service.times, service.output)
        except Exception as e: 
            logs.print("Encountered error while testing model for service:", service.name)
            service.results['model_tested'] = False
            print(e)


for service in services:
    if service.error['status'] is not True:
        if service.results['model_tested']:
            logs.print('Succesfully tested model for service: ', service.name)
        else:
            service.error['status'] = True
            service.error['text'].append('ERROR in testing, service rejected from comparison')
            logs.print(str(service.error['text']))

logs.print(" ")
logs.print('##########################  CLEANING [parallel] #######################################')
logs.print(" ")

for service in services:
    if service.error['status'] is not True:
        service.doCleaningThread.start()

for service in services:
    if service.error['status'] is not True:
        service.doCleaningThread.join()
        if service.results['done_cleaning']:
            logs.print('Succesfully cleaned up for service: ', service.name)
        else:
            service.error['status'] = True
            service.error['text'].append('ERROR in cleaning, service rejected from comparison')
            logs.print(str(service.error['text']))

logs.print(" ")
logs.print('##########################           #######################################')
logs.print(" ")
logs.print('##########################  SUMMARY  #######################################')
logs.print(" ")
logs.print('##########################           #######################################')

try: 

    services_finished_num = 0
    for service in services:
        if service.error['status'] is not True:
            services_finished_num +=1

    logs.print('')
    logs.print("----------------",services_finished_num, "FROM", len(services), "SERVICES FINISHED WORK ")
    logs.print('')
    logs.print('---------------------------------------------------')
    logs.print("---------------- SHOW ERRORS ----------------------")

    errorsTable = PrettyTable(['Name','Status'])

    for service in services: 
        errorsTable.add_row([ service.name, 
                        'ERROR' if service.error['status']==True else 'Success'])   

    logs.print(errorsTable)
    logs.print(" ")
    for service in services: 
        if(service.error['status'] == True):
            logs.print('['+service.name+'] ERROR: ', service.error['text'])
            logs.print(" ")

    logs.print('')
    logs.print('---------------------------------------------------')
    logs.print("---------------- SHOW TIMES [s]--------------------")

    timesTable = PrettyTable(['Name','Configure', 'Load Dataset', 'Train', 'Download', 'Test', 'Cleaning'])

    for service in services: 
        timesTable.add_row([ service.name,
                    service.times['configure_time'], 
                    service.times['dataset_load_time'], 
                    service.times['model_train_time'], 
                    service.times['model_download_time'], 
                    service.times['model_test_time'], 
                    service.times['do_cleaning_time'] ])

    logs.print(timesTable)

    logs.print('')
    logs.print('---------------------------------------------------')
    logs.print("---------------- SHOW MODEL DETAILS ---------------")

    modelsTable = PrettyTable(['Name','good_precision', 'good_recall', 'good_f1', 'bad_precision', 'bad_recall', 'bad_f1', ])

    for service in services: 
        modelsTable.add_row([ service.name,
                    str(service.output['good_precision']* 100) + "%" , 
                    str(service.output['good_recall']* 100) + "%",
                    str(service.output['good_f1']* 100) + "%",
                    str(service.output['bad_precision']* 100) + "%", 
                    str(service.output['bad_recall']* 100) + "%",
                    str(service.output['bad_f1']* 100) + "%" ])

    logs.print(modelsTable)

except Exception as e: 
    logs.print("Error while printing summary")
    print(e)


### save summary to json file: 
try: 

    summary_array = []
    for service in services: 
        service_dict = {
            'name' : service.name,
            'error' : service.error,
            'results' : service.results,
            'times' : service.times,
            'output' : service.output
        }
        summary_array.append(service_dict)
    summary_json = json.dumps(summary_array,indent=2)

    with open(os.path.join(output_path,'summary.json'), 'w') as f:
        print(summary_json, file=f)

except Exception as e: 
    logs.print("Error while saving summary to json file")
    print(e)
