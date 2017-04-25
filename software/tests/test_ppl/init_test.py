import os
import requests
import zipfile

from multicut_src import DataSet

##########################
# FIXME this does not work
##########################
# download data from https://drive.google.com/open?id=0B4_sYa95eLJ1M1VicG1WUm5Id1k

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

#####################
#####################

def download_data():
    test_file   = "./test_data"
    if not os.path.exists(test_file):

        # download from gdrive
        id = '0B4_sYa95eLJ1M1VicG1WUm5Id1k'
        download_file_from_google_drive(id, test_file)

        # unzip
        zip_ref = zipfile.ZipFile("test_data.zip", 'r')
        zip_ref.extractall('./test_data')
        zip_ref.close()
        os.remove("test_data.zip")


# TODO upload neuroproof data somewhere and automatically download it
# TODO read cache folder in as parameter with argparse
data_folder_isotropic = './test_data/isotropic'
meta_folder_isotropic = './cache_isotropic'

data_folder_anisotropic = './test_data/anisotropic'
meta_folder_anisotropic = './cache_anisotropic'

def init_ds(data_folder, meta_folder):
    ds = DataSet(meta_folder, 'test')
    ds.add_raw(   os.path.join(data_folder,'raw.h5'),  'data')
    ds.add_input( os.path.join(data_folder,'pmap.h5'), 'data')
    ds.add_seg(   os.path.join(data_folder,'seg.h5'),  'data')
    ds.add_gt(    os.path.join(data_folder,'gt.h5'),   'data')

if __name__ == '__main__':
    download_data()
    init_ds(data_folder_isotropic, meta_folder_isotropic)
    init_ds(data_folder_anisotropic, meta_folder_anisotropic)
