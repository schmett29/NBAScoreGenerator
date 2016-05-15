from pyspark.sql import SQLContext, functions
from pyspark import SparkConf, SparkContext
import math

# Authors: David Schmetterling, Neal McGovern, Ade Afonja
# How to run: Change the file path to your games.json MongoDB dump result
# in the code that reads the JSON object, then run spark submit on the python program

import sys  # Need to have acces to sys.stdout
fd = open('/Users/David/Dropbox/2016S/CSCI3390/FinalProject/GameError.txt','w') # open the result file in write mode
old_stdout = sys.stdout   # store the default system handler to be able to restore it
sys.stdout = fd # Now your file is used by print as destination 

# Initialize Spark Context
conf = SparkConf().setAppName("Basketball").set("spark.executor.memory","7g")
# sc is an existing SparkContext.
sc = SparkContext(conf = conf)

# Normalize features function
def normalize(aList):
        s = sum(aList)
        return map(lambda x: float(x)/s, aList)

# Root mean squared between game nodes
def get_Distance(aList,bList):
	x = 0;
	for i in range(0, len(aList)):
		x += (aList[i] - bList[i])**2
	return math.sqrt(x)

sqlContext = SQLContext(sc)

# A JSON dataset is pointed to by path.
# The path can be either a single text file or a directory storing text files.
people = sqlContext.read.json("/Users/David/Dropbox/2016S/CSCI3390/FinalProject/dump/nba/games.json")

# Register this DataFrame as a table.
people.registerTempTable("people")

# SQL statements can be run by using the sql methods provided by `sqlContext`.
teenagers = sqlContext.sql("SELECT box.team FROM people")
teenagers.registerTempTable("teenagers")

# use this to scale the large data file
# teenagers,dump = teenagers.randomSplit([.2,.8])

#Split RDD into training and test set
trainingDF,testDF = teenagers.randomSplit([.90,.10])

trainingDF.registerTempTable("trainingDF")
testDF.registerTempTable("testDF")

#Use SQL to get the winning team's box score for training
wScores = sqlContext.sql("SELECT team[0] FROM trainingDF")
#Use SQL to get the losing team's box score for training
lScores = sqlContext.sql("SELECT team[1] FROM trainingDF")
training = []

# Puts all the winning teams' features into a list
# Excluded points (feature at index 16) because that tells you who won
# Excluded percentages because they were not useful
for row in wScores.rdd.collect():
	temp = []
	for i in range(0, 20):
		if (type(row[0][i]) == int and i != 16):
			temp.append(row[0][i])
	training.append(temp)
j = 0
# Concatonates the losing teams' features into the same training list from above
for row in lScores.rdd.collect():
	temp = []
	for i in range(0, 20):
		if (type(row[0][i]) == int and i != 16):
			temp.append(row[0][i])
		
	training[j] += temp
	j+=1

#Use SQL to get the winning team's box score for test
wScores = sqlContext.sql("SELECT team[0] FROM testDF")
#Use SQL to get the losing team's box score for test
lScores = sqlContext.sql("SELECT team[1] FROM testDF")
test = []

#Repeat the above processes for the test dataset
for row in wScores.rdd.collect():
	temp = []
	for i in range(0, 20):
		if (type(row[0][i]) == int and i != 16):
			temp.append(row[0][i])
	test.append(temp)
j = 0
for row in lScores.rdd.collect():
	temp = []
	for i in range(0, 20):
		if (type(row[0][i]) == int and i != 16):
			temp.append(row[0][i])
	test[j] += temp
	j+=1

eList1 = []
eList2 = []
trainingError1 = []
trainingError2 = []

# Get "relatively good k" by taking square root of the total number of game node
k = int(math.sqrt(len(training)+len(test)))
print("K: " + str(k))

# Creates error list for winning and losing feature vectors for each game
for game in test:
	# Switch the winning and the losing teams' features
	flipGame = game[15:] + game[:15]
	eList1 = []
	eList2 = []
	# Normalize features for better accuracy
	flipGame = normalize(flipGame)
	game = normalize(game)
	# Finds the KNN's distance from each vector
	for tGame in training:
		error1 = get_Distance(game, tGame)
		if len(eList1) < k:
			eList1.append(error1)
		elif error1 < max(eList1):
			eList1.append(error1)
			eList1.remove(max(eList1))
		error2 = get_Distance(flipGame, tGame)
		if len(eList2) < k:
			eList2.append(error2)
		elif error2 < max(eList2):
			eList2.append(error2)
			eList2.remove(max(eList2))
	trainingError1.append(eList1)
	trainingError2.append(eList2)

team1 = 0
team2 = 0

# Calculates who will win and adds it to the team's total wins to be printed
for x in range(0, len(test)):
	error = sum(trainingError1[x]) - sum(trainingError2[x])
	
	if error<0:
		team1+=1
		print("Error: " + str(error))
		print("Team 1 would win this game")
		
		
	else:
		team2+=1
		print("Error: " + str(error))
		print("Team 2 would win this game")
	
sys.stdout=old_stdout # here we restore the default behavior
print("Team 1 Wins: " + str(team1))
print("Team 2 Wins: " + str(team2))

# Prints the accuracy of our results
accuracy = float(team1) / (float(team1) + float(team2))
print("Accuracy: "+ str(accuracy))

fd.close() # to not forget to close your file
sc.stop()