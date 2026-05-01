# Plan for processing data

## all data are from https://co2datashare.org/dataset/smeaheia-dataset.
downloaded and extracted in data/download

## there are majors useful data that i focus on are
fault_sticks , seismics and reports

    (for seismic inputs)
    seismic_2d _lines
    seismic_3d_surveys
    fault_sticks

    (for seismic contexts)
    reports
    horizons
    surfaces
    well_logs (optional)

## FAULT Detection

### fault datas like fault_sticks are not labels. They are picked fault traces in 3d seismic.
So , first extrack parsing and extracting position , Then map to seismic 2d data extracted from 3d seismic also labeling it, finally packed it to datasets for fault detection.

### main pair for fault detections
    Input seismic:
    data/download/seismic_3d_surveys/Seismic_3D_Surveys/data/GN1101_Scaled(Realized)

    Fault labels:
    data/download/fault_sticks/Fault_Sticks/data/fault_Sticks_GN1101_2012

#### TNE01 is 3d seismics data , but not be use for training since it has no direct fault labels , but possible for validation

## CONTEXT Understanding
### for context in PDF's report to understand images data. 
My Idea is to extracted reference images from reports with its underline explanation into multimodal datasets

