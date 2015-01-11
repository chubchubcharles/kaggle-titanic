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

# we are going to create a model based on gender,class,ticket price
# which will be a 2 x 3 x 4 matrix table

# add a ceiling
fare_ceiling = 40
# then modify the data in the Fare/Cabin column to = 39, if it is greater or equal to the ceiling
# because we want to classify into (0-9), (10-19), (20-29), and (30-39)
data[ data[0::,9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0
# notice the "data[0::,9].astype(np.float) >= fare_ceiling" is a boolean statement
# which takes a subset of data that is True for that boolean statement
fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size

# even though we know that there are 3 classes from the problem synopsis, we should verify from our data
# number_of_classes = 3
number_of_classes = len(np.unique(data[0::,2]))

# initialize the survival table with all zeros with our provided dimensions (2,3,4)
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

# model generation
for i in xrange(number_of_classes): #xrange() ~ range(), but returns xrange obj instead of list
	for j in xrange(number_of_price_brackets):
		# element which is female, ith class was greater than this bin and less than next bin in 2nd column
		women_only_stats = data[(data[0::,4] == "female") \
			& (data[0::,2].astype(np.float) == i+1) \
			& (data[0:,9].astype(np.float) >= j*fare_bracket_size) \
			& (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size) , 1]

		men_only_stats = data[(data[0::,4] != "female") \
			& (data[0::,2].astype(np.float) == i+1) \
			& (data[0:,9].astype(np.float) >= j*fare_bracket_size) \
			& (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size) , 1]

		# explanation: in i=0, j=0 of first loop, loop calculates survival 1st class (i+1) females
		# who paid less than 10 and likewise for males

		# calculate proportion of survival by averaging survival rate
		survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
		survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))

		# issue: since in reality no women paid less than $10 for a first class, there will be no data
		# but we don't want Python to return NaN (and a warning). We'd rather have 0 for the survival rate.
		survival_table[ survival_table != survival_table ] = 0 #comment this line to see warning

		# print survival_table. example: 0.97727273 means about 97% of 1st class females who paid $30-39 survived
		# we will arbitrarily set any sample to fit these conditions, which have >50% survival rate to also survive
		survival_table[ survival_table < 0.5 ] = 0
		survival_table[ survival_table >= 0.5 ] = 1

# open test file
test_file = open('./csv/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# open new and final file
predictions_file = open("./csv/genderclassmodel.csv", "wb")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])

for row in test_file_object:
	for j in xrange(number_of_price_brackets):
		try:
			row[8] = float(row[8]) # to check for the presence of Fare data
		except:
			bin_fare = 3 - float(row[1]) # if no Fare data, bin data according to Class
			break
		if row[8] > fare_ceiling: # if data.exists and fare is greater than $40, their bin_fare is 3 or first class 
			bin_fare = number_of_price_brackets - 1
			break
		if row[8] >= j * fare_bracket_size and row[8] < (j+1) * fare_bracket_size: 
		# assign bin_fare based on how much they paid
		# notice first bin is j = 0, instead of j = 1, because it will be used as indices in survival table
			bin_fare = j
			break
			
	if row[3] == 'female':
		# If passenger is female, write her survival boolean 
		p.writerow([row[0], "%d" % int(survival_table[0, float(row[1])-1, bin_fare])])
		# male
	else:
		p.writerow([row[0], "%d" % int(survival_table[1, float(row[1])-1, bin_fare])])

#Close files
test_file.close()
predictions_file.close()

















