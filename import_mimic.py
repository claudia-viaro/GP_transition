import gzip
import csv
import json
import pandas as pd
import numpy as np


import zipfile

zf = zipfile.ZipFile('C:/Users/cvcla/my_py_projects/GP_intervention/mimic-iii-clinical-database-1.4.zip') 
#df = pd.read_csv(zf.open('intfile.csv'))
text_files = zf.infolist()
list_ = []

print ("Uncompressing and reading data... ")
for text_file in text_files:
    print(text_file.filename)

addmissions_df = pd.read_csv(zf.open('mimic-iii-clinical-database-1.4/ADMISSIONS.csv.gz'), compression='gzip',
                   error_bad_lines=False) # (58976, 19)


for col in addmissions_df.columns:
    print(col)
print(addmissions_df.shape   ) 
print(addmissions_df.info)

df2 = addmissions_df.pivot_table(index = ['SUBJECT_ID'], aggfunc ='size')
#print(data['ADMISSION_TYPE'].unique().tolist())
count_groups = addmissions_df.groupby(['ADMISSION_TYPE']).size().reset_index(name='counts')
print(count_groups)
trim_data = pd.DataFrame(data={'ID': addmissions_df['SUBJECT_ID']                        
                        })
trim_data['admission'] = np.where(addmissions_df['ADMISSION_TYPE']=='URGENT', 1, 0)
print(addmissions_df['DEATHTIME'].isnull().sum(),len(addmissions_df['DEATHTIME']))

'''
outputevents_df = pd.read_csv(zf.open('mimic-iii-clinical-database-1.4/OUTPUTEVENTS.csv.gz'), compression='gzip',
                   error_bad_lines=False) # (58976, 19)
for col in outputevents_df.columns:
    print(col)
print(outputevents_df.shape   ) 
print(outputevents_df.info)
print(len(outputevents_df['HADM_ID'])-len(outputevents_df['HADM_ID'].drop_duplicates()))
print(len(outputevents_df['ICUSTAY_ID'])-len(outputevents_df['ICUSTAY_ID'].drop_duplicates()))


microbiology_events_df = pd.read_csv(zf.open('mimic-iii-clinical-database-1.4/MICROBIOLOGYEVENTS.csv.gz'), compression='gzip',
                   error_bad_lines=False) # (58976, 19)
for col in microbiology_events_df.columns:
    print(col)
#print(microbiology_events_df.info)
'''
labitems_df = pd.read_csv(zf.open('mimic-iii-clinical-database-1.4/D_LABITEMS.csv.gz'), compression='gzip',
                   error_bad_lines=False) # (58976, 19)
for col in labitems_df.columns:
    print(col)
print(labitems_df.info)
#print(labitems_df['DOSE_VAL_RX'][0:10])