# Flex Glove Gesture Recognition (ML)

This repository contains the machine learning and data processing components Team 7's Senior Design Project

# System Overview
The system uses 5 flex sensors attached to a glove to detect hand positions and classify them into ASL gestures based on a KNN machine learning model

# Input Data

Input Data will be read in this format and monitored at a rate of 20Hz:

thumb,index,middle,ring,pinky,label
32,45,78,80,60,A

# User Interface
The UI is divided into two columns: the left column is for parameters, and the right column is for real-time visualizations, and where the actual "guessing" comes from

# Preferences
Connection - Connect/Disconnect to glove through specified COM port

Calibrate sensors - A bandaid solution to people having different size hands, stores maxs and mins for each finger.

Hand Size - 0.5x to 2.0x, another bandaid solution to people having different size hands

KNN Neighbors - Purpose is to set the 'k' parameter for KNN algorithm, where lower values cause a more sensitive algorithm, while higher values are more stable, but may underfit. In other words, its the number of values from the training data that are being used in the recognition.

Confidence Threshold Slider - Show minimum confidence required to display a prediction.

Smoothing - average the last N sensor readings to reduce noise + smooth data

# Normal Operation
Collecting Data:
1. Enter label you are signing
2. Hit record (Currently recording at 10Hz)
3. Save csv
4. Load ML with data

# Sensor Format Styles
{"sensors":[0,10,20,30,40],"timestamp":143622}
{"sensors":[0,10,20,30,40],"timestamp":143672}
{0,10,20,30,40} <- using this one

# Bluetooth
We've been going back and forth between using BLE and SPP bluetooth. For now, we are using SPP because it is a little less complicated code-wise, and its functionally the same to just using a regular serial connection. For the future, we intend to implement BLE.

# Things to add / TO-DO
Maybe a pulsing dot to represent the frequency of data being sent
^ adding onto this maybe a display of the current rate that data is being received/read at (sample rate)
Current Max/min of sensor data
    - could help with debugging
Disconnect button / dynamically changing connect glove button
Make the UI a little more smooth and a little less rigid

#
