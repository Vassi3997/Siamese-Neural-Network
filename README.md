# Siamese-Neural-Network-
Facial recognition using Siamese Network based Attendance System

Execution Sequence :
1. Execute Generating_training_data.py(save the first name as <name0> then the next name is to be saved as <name1> and so on)
2. Move the newly created name(number) folder in a people folder
3. Execute Siamese_Test.py
4. Create Database of name new2 (command : use new2)
   Create Collection of name pa (command : db.createCollection('pa'))
   Insert 'attendance' as 0 for all the 'name' (command : db.pa.insert({"name":"<name>","attendance":0}))
5. Execute FaceRecognizer.py
 
