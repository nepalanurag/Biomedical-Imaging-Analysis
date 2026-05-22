import argparse
import distutils.spawn
import os
import pandas as pd
import numpy as np
import mdai
import logging
import pydicom
import tqdm
import itk
import cv2
import copy
import glob
import subprocess
import glob
import sys
import dicom2nifti
import dcmqi

# logging setup

logging.basicConfig()

# By default the root logger is set to WARNING and all loggers you define
# inherit that value. Here we set the root logger to NOTSET. This logging
# level is automatically inherited by all existing and new sub-loggers
# that do not set a less verbose level.
logging.root.setLevel(logging.NOTSET)

# The following line sets the root logger level as well.
# It's equivalent to both previous statements combined:
logging.basicConfig(level=logging.NOTSET)

logger = logging.getLogger('dcmqi.mdai2dcm')
logger.setLevel(logging.ERROR)

# Helper functions
dcmqi_template =  {
"ContentCreatorName": "",
"ClinicalTrialSeriesID": "Session1",
"ClinicalTrialTimePointID": "1",
"SeriesDescription": "Segmentation",
"SeriesNumber": "300",
"InstanceNumber": "1",
"BodyPartExamined": "CHEST",
"segmentAttributes": [
],
"ContentLabel": "SEGMENTATION",
"ContentDescription": "Image segmentation",
"ClinicalTrialCoordinatingCenterName": "dcmqi"
}

def makeHash(text, length=6):
    from base64 import b64encode
    from hashlib import sha1
    return b64encode(sha1(str.encode(text)).digest()).decode('ascii')[:length]

segment_template = {
"labelID": "",
"SegmentDescription": "",
"SegmentAlgorithmType": "MANUAL",
"SegmentedPropertyCategoryCodeSequence": {
"CodeValue": "49755003",
"CodingSchemeDesignator": "SCT",
"CodeMeaning": "Morphologically Abnormal Structure"
},
"SegmentedPropertyTypeCodeSequence": {
"CodeValue": "",
"CodingSchemeDesignator": "99RICORD",
"CodeMeaning": ""            
},
"recommendedDisplayRGBValue": [177,122,101
]
}

def convertToSEG2(input_dicom_dir, seg_dir):

    print("Saving DICOM SEG to "+seg_dir)
    json_files = glob.glob(seg_dir+"/*.json")
    
    for label_json in json_files:
        
        creator = os.path.split(label_json)[1].split('-')[0]
        label_dcm = os.path.join(seg_dir,creator+".dcm")
        
        labels = glob.glob(seg_dir+'/'+creator+'*.nii')
        labels.sort()
            
        cmd = ['itkimage2segimage','--inputDICOMDirectory',input_dicom_dir,'--inputImageList',','.join(labels),"--inputMetadata",label_json, "--outputDICOM", label_dcm, "--skip"]
        print('Running SEG conversion with:\n'+' '.join(cmd))
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        print(result.stderr)
        print(result.stdout)


def main():
    parser = argparse.ArgumentParser(
    usage="%(prog)s --inputDICOM <dir> --inputJSON <name> --outputDirectory <dir>\n\n""Warning: This is an experimenta script in development!\n""The intent of this helper script is to enable conversion""of the MD.ai annotations into appropriate standard DICOM objects.\n")
    
    parser.add_argument(
    '--inputDICOM',
    dest="inputDICOM",
    default="/Users/anurag/Downloads/CSC821 Files/manifest-1608266677008/MIDRC-RICORD-1A/MIDRC-RICORD-1A-660042-000107/11-20-2008-NA-NA-25678/4.000000-NA-25679",
    metavar='Input directory with the DICOM images being annotated',
    help="Directory with the input DICOM images. It is expected that"" the content of this directory is organized into the following""hierarchy: <StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm")

    
    parser.add_argument(
    '--inputJSON',
    dest="inputJSON",
    default="/Users/anurag/Downloads/CSC821 Files/MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json",
    metavar='Input MD.ai annotations in the native JSON representation',
    help="MD.ai nnotations stored in MD.ai format.")
    parser.add_argument(
    '--outputDirectory',
    dest="outputDirectory",
    default="/Users/anurag/Downloads/CSC821 Files/on",
    metavar='Output directory to store the resulting DICOM files',
    help="Directory to store resulting converted DICOM objects.")
    args = parser.parse_args()
    if not os.path.exists(args.outputDirectory):
        os.mkdir(args.outputDirectory)

    results = mdai.common_utils.json_to_dataframe(args.inputJSON)  
    annotations_df = results['annotations']
    logger.info(f"{annotations_df.shape[0]} annotations loaded")

    all_series_uids = annotations_df['SeriesInstanceUID'].unique()
    # TODO: Why there are NaNs?
    all_series_uids = all_series_uids[~pd.isnull(all_series_uids)]  
    progress_bar = tqdm.tqdm(total=len(all_series_uids))    
    for this_series_uid in all_series_uids: 
    #['1.2.826.0.1.3680043.10.474.419639.300423266679936330916265249312']:  # ['1.2.826.0.1.3680043.10.474.440808.1993']: #all_series_uids: # ['1.2.826.0.1.3680043.10.474.2969551981555819856670502082591727602']: 
        print(f"Processing {this_series_uid}")

        # filter our irrelevant rows
        this_series_annotations = annotations_df[annotations_df["SeriesInstanceUID"] == this_series_uid]
        this_series_annotations = this_series_annotations[this_series_annotations['annotationMode'] == 'freeform']

        if this_series_annotations.shape[0] == 0:
            continue

        print('Creators of freeform annotations for series '+this_series_uid+" are "+str(this_series_annotations['createdById'].unique()))

        one_row = this_series_annotations.iloc[0]

        series_uid = one_row["SeriesInstanceUID"]
        study_path = os.path.join(args.inputDICOM)
        segmentations_path = os.path.join(args.outputDirectory,series_uid+"_SEG")
        reconstruction_path = os.path.join(args.outputDirectory,series_uid+"_Reconstruction")

        if not os.path.exists(segmentations_path):
            os.mkdir(segmentations_path)
        if not os.path.exists(reconstruction_path):
            os.mkdir(reconstruction_path)
        
        series_path = os.path.join(study_path)
        label_images_per_creator = {}
        label_json_per_creator = {}
        creators = this_series_annotations['createdById'].unique()

        for creator in creators:
            label_images_per_creator[creator] = []
            label_json_per_creator[creator] = copy.deepcopy(dcmqi_template)
            label_json_per_creator[creator]['ContentCreatorName'] = creator
            label_json_per_creator[creator]['segmentAttributes'] = []

        for creator in creators:
            print('Creator '+creator+' has '+str(len(label_images_per_creator[creator]))+' segmentations and '+str(len(label_json_per_creator[creator]['segmentAttributes']))+' seg attrs')
            print('Processing labels for '+creator)
            for idx, label_itk_image in enumerate(label_images_per_creator[creator]):
                output_file_name = creator+"-"+('%03d' % idx)
                itk.imwrite(label_itk_image, os.path.join(segmentations_path,output_file_name+".nii"))

            import json
            output_json_name = creator+"-metadata.json"
            with open(os.path.join(segmentations_path,output_json_name), "w") as json_file:
                json_file.write(json.dumps(label_json_per_creator[creator], indent=2))

        convertToSEG2(series_path, segmentations_path)

        if len(glob.glob(segmentations_path+"/*dcm")) == 0:
            logger.error(f"SEG conversion failed for {segmentations_path}!")

        progress_bar.update(1)

    print("Done")

    return

if __name__ == "__main__":
    main()
