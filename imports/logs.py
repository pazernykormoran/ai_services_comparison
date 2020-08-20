class Logs:
    def __init__(self, datasets_dir, folder_with_dataset_name, date,output_folder):
        self.datasets_dir = datasets_dir
        self.folder_with_dataset_name = folder_with_dataset_name
        self.date = date
        self.output_folder = output_folder

    def print(self,*args, **kwargs):
        print(*args, **kwargs)
        with open(self.datasets_dir+'/'+self.folder_with_dataset_name+'/'+ self.output_folder+'/logs_'+self.date+'.log','a') as file:
            print(*args, **kwargs, file=file)
