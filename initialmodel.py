import csv as csv
import numpy as np

csv_file = csv.reader(open('./csv/train.csv', 'rb'))
header = csv_file.next() 

data = []
for row in csv_file:
	data.append(row)
# convert list into array, each format = string
data = np.array(data)

# gender column
gender_column = data[0::,4]

# since each type is in strings, need to convert to float to perform calculations
# class column
class_column = data[0::,2].astype(np.float)

# survival number
number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

# subset of data by gender
#yields data array of boolean value 'True' if female
number_women = data[0::,4] == "female" 
number_men = data[0::,4] != "female"

women_onboard = data[number_women,1].astype(np.float)
men_onboard = data[number_men,1].astype(np.float)

proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)

# reading test file
test_file = open('./csv/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# creating new (final) file for submission
prediction_file = open("./csv/genderbasedmodel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)

# for this subission, only two columns are allowed and written here
# file objects are used as pointers to read and write lines
prediction_file_object.writerow(["PassengerId", "Survived"])

for row in test_file_object:
	if row[3] == 'female':
		prediction_file_object.writerow([row[0],'1']) # predict 1
	else:
		prediction_file_object.writerow([row[0],'0']) # predict 0

# close files
test_file.close() 
prediction_file.close()






