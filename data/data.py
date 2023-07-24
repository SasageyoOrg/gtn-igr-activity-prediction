"""
    File to load dataset based on user control from main file
"""

from data.IGs import IGsDataset

def LoadData(DATASET_NAME):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """
    if DATASET_NAME == 'IG':
      return IGsDataset()
    