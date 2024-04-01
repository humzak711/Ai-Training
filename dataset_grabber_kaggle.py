# To gather datasets from kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
# For sorting the downloaded files
import os

# Function to find datasets in kaggle and sort them
def Dataset_finder(target_directory:str, number_downloads:int=10) -> str:
    ''' 
    Uses Kaggles API to gather datasets and sort them into a designated folder

    Ensure that you have your kaggle.json file containing your kaggle api credentials
    stored in your Users/(your username)/.kaggle folder on your os to use this function
    '''
    
    # Initialize kaggle api
    api = KaggleApi()
    api.authenticate()
    
    # Create target directory if it doesn't exist
    if os.path.exists(target_directory) == False:
        os.makedirs(target_directory)
        
    # List available datasets
    datasets = api.dataset_list()

    # Download given number of CSV files 
    for dataset in datasets[:number_downloads]:  
        # api.dataset_download_files overwrites existing files if the target path already exists
        api.dataset_download_files(dataset.ref, path=target_directory, unzip=True, quiet=True)
        
    return 'files successfully downloaded'

# Test function
datasets_amount = int(input('How many datasets do you want to grab: '))
print(Dataset_finder('Training_Data', datasets_amount))
