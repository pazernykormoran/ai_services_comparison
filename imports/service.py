import threading

class Service:
    def __init__(self, name, service):
        self.name = name
        self.service = service
        self.error = {"status": False, "text": []}
        self.results = {
            'initialized': False,
            'configured': False,
            'dataset_loaded': False,
            'model_trained':False,
            'model_downloaded':False,
            'model_tested':False,
            'done_cleaning':False
        }
        self.times = {
            'configure_time' : 0,
            'dataset_load_time' : 0,
            'model_train_time' : 0,
            'model_download_time' : 0,
            'model_test_time' : 0,
            'do_cleaning_time' : 0
        }
        self.output = {
            'good_precision': 0,
            'good_recall': 0,
            'good_f1': 0,
            'bad_precision': 0,
            'bad_recall': 0,
            'bad_f1': 0,
        }
        # simultanous functions
        self.configureThread = threading.Thread(target=self.service.configure, args=(self.results, self.times, ))
        self.trainModelThread = threading.Thread(target=self.service.train_model, args=(self.results, self.times, ))
        # self.testModelThread = threading.Thread(target=self.service.test_model, args=(self.results, self.times, ))
        self.doCleaningThread = threading.Thread(target=self.service.do_cleaning, args=(self.results, self.times, ))