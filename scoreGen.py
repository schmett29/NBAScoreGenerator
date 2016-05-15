# sc is an existing SparkContext.
from pyspark.sql import SQLContext, functions
from pyspark import SparkConf, SparkContext
import math

# Authors: David Schmetterling, Neal McGovern, Ade Afonja
# How to run: Change the file path to your games.json MongoDB dump result
# in the code that reads the JSON object, then run spark submit on the python program

import sys  # Need to have acces to sys.stdout
fd = open('/Users/David/Dropbox/2016S/CSCI3390/FinalProject/ScorePredictions.txt','w') # open the result file in write mode
old_stdout = sys.stdout   # store the default system handler to be able to restore it
sys.stdout = fd # Now your file is used by print as destination 

# Initialize Spark Context
conf = SparkConf().setAppName("ScoreGen").set("spark.executor.memory","7g")
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
# teenagers, dump = teenagers.randomSplit([.2, .8])

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
# Excluded percentages because they were not useful
for row in wScores.rdd.collect():
	temp = []
	for i in range(0, 20):
		if (type(row[0][i]) == int):
			temp.append(row[0][i])
	training.append(temp)
j = 0
# Concatonates the losing teams' features into the same training list from above
for row in lScores.rdd.collect():
	temp = []
	for i in range(0, 20):
		if (type(row[0][i]) == int):
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
		if (type(row[0][i]) == int):
			temp.append(row[0][i])
	test.append(temp)
j = 0
for row in lScores.rdd.collect():
	temp = []
	for i in range(0, 20):
		if (type(row[0][i]) == int):
			temp.append(row[0][i])
	test[j] += temp
	j+=1

eList1 = []
eList2 = []
trainingError1 = []
trainingError2 = []

# Get "relatively good k" by taking square root of the total number of game node
k = int(math.sqrt(len(test) + len(training)))
print("K: " + str(k))
max1 = 0
max2 = 0

# Creates error list for winning and losing feature vectors for each game
for game1 in test:
	# Get rid of the points feature
	game = game1[:12] + game1[13:28] + game1[29:]
	# Switch the winning and the losing teams' features and normalize
	flipGame = normalize(game[15:] + game[:15])
	# Normalize features for better accuracy
	game = normalize(game)
	eList1 = []
	eList2 = []
	# Finds the KNN's distance from each vector
	for tGame1 in training:
		tGame = tGame1[:12] + tGame1[13:28] + tGame1[29:]
		tGame = normalize(tGame)

		error1 = get_Distance(game, tGame)
		
		if len(eList1) < k:
			eList1.append((error1, (tGame1[12], tGame1[28])))
		elif error1 < max1:
			eList1.append((error1, (tGame1[12], tGame1[28])))
			eList1.pop([eList1[j][0] for j in range(len(eList1))].index(max1))

		error2 = get_Distance(flipGame, tGame)
		
		if len(eList2) < k:
			eList2.append((error2, (tGame1[12], tGame1[28])))
		elif error2 < max2:
			eList2.append((error2, (tGame1[12], tGame1[28])))
			eList2.pop([eList2[j][0] for j in range(len(eList2))].index(max2))
		max1 = max([eList1[j][0] for j in range(len(eList1))])
		max2 = max([eList2[j][0] for j in range(len(eList2))])
	trainingError1.append((eList1, (game1[12], game1[28])))
	trainingError2.append(eList2)

team1 = 0
team2 = 0

# Calculates each team's predicted score and prints out both teams
for x in range(0, len(test)):
	wTemp1 = 0.0
	wTemp2 = 0.0
	lTemp1 = 0.0
	lTemp2 = 0.0
	error = sum([trainingError1[x][0][j][0] for j in range(k)]) - sum([trainingError2[x][j][0] for j in range(k)])
	for i in range(k):
		wTemp1 += trainingError1[x][0][i][1][0]
		wTemp2 += trainingError1[x][0][i][1][1]
		lTemp1 += trainingError2[x][i][1][0]
		lTemp2 += trainingError2[x][i][1][1]
	if error<0:
		team1 += 1
		print("Error: " + str(error))
		print("team 1 would win this game with a predicted score of: "+ str(wTemp1/k) + " - " + str(wTemp2/k) +  "\n")
	else:
		team2 += 1
		print("Error: " + str(error))
		print "team 2 would win this game with a predicted score of: "+ str(lTemp1/k) + " - " + str(lTemp2/k) +  "\n"
	print "Actual result: " + str(trainingError1[x][1][0]) + " - " + str(trainingError1[x][1][1]) +  "\n"

sys.stdout=old_stdout # here we restore the default behavior
print("Team 1 Wins: " + str(team1))
print("Team 2 Wins: " + str(team2))
print "Overall Accuracy = " + str(float(team1)/(team1+team2))
fd.close() # to not forget to close your file
sc.stop()
