# Name: Austin Way
# Program: Big G Express Stock Price Prediction 
# Date: 7/6/2020
# Description: Big G Express is an ESOP company In which I am currently employed with. There was a contest to have users guess the stock price for 2019. 
#     The winner of the contest got a prise of $250. So, I made this little script to help me take an educated guess 
#     on predicitng the stock price with public data. I guessed the price of $54.96 based on linear progression.
#     However, that guess was low. The actual price was $59.81. 

import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import r2_score
from scipy import stats

# Stock Data
# Data found at: https://www.biggexpress.com/our-company/employee-ownership
x = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
y = [5.88, 11.46, 12.34, 12.65, 16.60, 27.75, 37.19, 43.34, 44.76, 49.40]


# Standard Deviation
def standardDeviation(array, description):
    std = numpy.std(array)
    return print("Standard Deviation of " + description + ": " + str(round(std, 2)))

# standardDeviation(x, "Year")
# standardDeviation(y, "Price")


# Polynomial Regression
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(2009, 2018, 50)

# plt.scatter(x, y)
# plt.plot(myline, mymodel(myline))
# plt.show()

poly_reg_guess = mymodel(2019)
print("Polynomial Regression: " + str(round(poly_reg_guess, 2)))


# Linear Regression
slope, intercept, r, p, std_err = stats.linregress(x, y)

def linearRegression(x):
    return slope * x + intercept

lin_reg_guess = linearRegression(2019)   
print("Linear Regression: " + str(round(lin_reg_guess, 2)))

mymodel = list(map(linearRegression, x))

# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()

guesses = [poly_reg_guess, lin_reg_guess]
avg_guess = numpy.mean(guesses)
print("Average: " + str(round(avg_guess, 2)))


# Train/Test & R2 Score
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(train_y, mymodel(train_x))

print("R2: " + str(round(r2, 2)))


# Output actual 2019 Stock price that was revealed
print("Actual 2019 Stock Price: 59.81")


# Example Output
# ==============
# Polynomial Regression: 49.78
# Linear Regression: 54.96
# Average: 52.37
# R2: 0.98
# Actual 2019 Stock Price: 59.81


