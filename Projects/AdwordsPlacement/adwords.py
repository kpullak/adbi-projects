#!/usr/bin/python

import numpy as np
import pandas as pd
import sys
import csv
import random

# Setting a seed to ensure replicability of results
random.seed(0)

# Greedy algorithm implementation function
def greedy(budgetDict, bidsDict, queries):

    revenue = 0.00
    for query in queries:
      bidder = find_bidder_greedy(bidsDict[query], budgetDict)
      if bidder != -1:
         revenue += bidsDict[query][bidder]
         budgetDict[bidder] -= bidsDict[query][bidder]
    return revenue

# Balance algorithm implementation function
def balance(budgetDict, bidsDict, queries):

    revenue = 0.00
    for query in queries:
      bidder = find_bidder_balance(bidsDict[query], budgetDict)
      if bidder != -1:
         revenue += bidsDict[query][bidder]
         budgetDict[bidder] -= bidsDict[query][bidder]
    return revenue

# MSVV algorithm implementation function
def msvv(rembudgetDict, budgetDict, bidsDict, queries):

    revenue = 0.00
    for query in queries:
      bidder = find_bidder_msvv(bidsDict[query], rembudgetDict, budgetDict)
      if bidder != -1:
         revenue += bidsDict[query][bidder]
         rembudgetDict[bidder] -= bidsDict[query][bidder]
    return revenue

# Helper function to check if all the bidders of a query have exhausted their budget or not.
# Returns: -1 if they have all exhausted their budgets, 0 otherwise
def checkBudget(bids, budgetDict):

   keys = bids.keys()
   for key in keys:
      if budgetDict[key] > bids[key]:
         return 0
   return -1


# Helper function to compute the psi value for a given fractional input
def psi(xu):
   return 1 - np.exp(xu-1)


# Scales the bid based on the remaining budget as per the MSVV algorithm
def scaledBid(bid, rembud, bud):
    xu = (bud-rembud)/bud
    return bid*psi(xu)


# Function to find the best bidder for a particular query using the 'greedy' algorithm
# Input: bids: bids of all available advertisers for a query, budgetDict: budgets of all advertisers
# Output: id of the bidder that should be matched if possible, -1 if no bidder possible
# Algorithm: For each bidder check if the bid is greater than max, if yes, update max else:
# if equal to max (meaning more than one bidder can be selected), compare the bidder id and choose the min
def find_bidder_greedy(bids, budgetDict):

   keys = list(bids.keys())
   maxBidder = keys[0]
   maxBid = -1
   isEnoughBudget = checkBudget(bids, budgetDict)
   if isEnoughBudget == -1:
      return -1
   for key in keys:
      if budgetDict[key] >= bids[key]:
         if maxBid < bids[key]:
            maxBidder = key
            maxBid = bids[key]
         elif maxBid == bids[key]: # incase of a tie we return the bidder with the lower bidderId (sorting lexicographically)
            if maxBidder > key:
               maxBidder = key
               maxBid = bids[key]
   return maxBidder


# Function to find the best bidder for a particular query using the 'balance' algorithm
# Input: bids: bids of all available advertisers for the query, budgetDict: budgets of all advertisers
# Output: id of the bidder that should be matched if possible, -1 if no bidder possible
# Algorithm: For each bidder check if the reamining budget is greater than max, if yes, update max else:
# if equal to max (meaning more than one bidder can be selected), compare the bidder id and choose the min
def find_bidder_balance(bids, budgetDict):

    keys = list(bids.keys())
    maxBidder = keys[0]
    isEnoughBudget = checkBudget(bids, budgetDict)
    if isEnoughBudget == -1:
      return -1
    for key in keys:
      if budgetDict[key] >= bids[key]:
         if budgetDict[maxBidder] < budgetDict[key]:
            maxBidder = key
         elif budgetDict[maxBidder] == budgetDict[key]: # incase of a tie we return the bidder with the lower bidderId (sorting lexicographically)
            if maxBidder > key:
               maxBidder = key
         
    return maxBidder  


# Function to find the best bidder for a particular query using the 'msvv' algorithm
# Input: bids: bids of all available advertisers for the query, budgetDict: budgets of all advertisers
# Output: id of the bidder that should be matched if possible, -1 if no bidder possible
# Algorithm: For each bidder check if the scaled bid is greater than max, if yes, update max else:
# if equal to max (meaning more than one bidder can be selected), compare the bidder id and choose the min
def find_bidder_msvv(bids, rembudgetDict, budgetDict):

    keys = list(bids.keys())
    maxBidder = keys[0]
    isEnoughBudget = checkBudget(bids, rembudgetDict)
    if isEnoughBudget == -1:
      return -1
    for key in keys:
      if budgetDict[key] >= bids[key]:
         scaledBidMaxBidder = scaledBid(bids[maxBidder], rembudgetDict[maxBidder], budgetDict[maxBidder])
         scaledBidKey = scaledBid(bids[key], rembudgetDict[key], budgetDict[key])
         if scaledBidMaxBidder < scaledBidKey:
            maxBidder = key
         elif scaledBidMaxBidder == scaledBidKey: # incase of a tie we return the bidder with the lower bidderId (sorting lexicographically)
            if maxBidder > key:
               maxBidder = key
    return maxBidder  


def revenue_computation(budgetDict, bidsDict, queries, type, total_revenue, iterations):

   for i in range(0, iterations):
      if iterations > 1: # we use the shuffle function to average the total_revenue to compute the competitive_ratio
         random.shuffle(queries)
      budgetDict_ = dict(budgetDict)
      if type == 1:
         revenue = greedy(budgetDict_, bidsDict, queries)
      elif type == 2:
         revenue = msvv(budgetDict_, dict(budgetDict), bidsDict, queries)
      elif type == 3:
         revenue = balance(budgetDict_, bidsDict, queries)
      else:
         revenue = 0.00
      total_revenue += revenue   

   return total_revenue/iterations


# Runs a particular algorithm 100 times and reports the average revenue
# INPUT: Budget of all Advertisers, Bids of all queries for all bidders, all queries & type, where: type 1: Greedy, type 2: MSVV & type 3: Balance
# returns average revenue
def compute_revenue(budgetDict, bidsDict, queries, type):

   total_revenue = 0.00

   revenue = revenue_computation(budgetDict, bidsDict, queries, type, total_revenue, iterations=1)
   average_revenue = revenue_computation(budgetDict, bidsDict, queries, type, total_revenue, iterations=100) 
   return revenue, average_revenue



# The readData function takes input from the bidder_dataset.csv file, stores it into dictionaries
# Also, takes input queries from queries.txt.
def readData(type):

    budgetDict = dict()
    bidsDict = dict()

    csv_file = pd.read_csv('bidder_dataset.csv')

    for i in range(0, len(csv_file)):
      advID = csv_file.iloc[i]['Advertiser']
      keyword = csv_file.iloc[i]['Keyword']
      bidValue = csv_file.iloc[i]['Bid Value']
      budget = csv_file.iloc[i]['Budget']

      if not (advID in budgetDict):
          budgetDict[advID] = budget

      if not (keyword in bidsDict):
          bidsDict[keyword] = {}

      if not (advID in bidsDict[keyword]): 
          bidsDict[keyword][advID] = bidValue

    with open('queries.txt') as content:
      queries = content.readlines()

    queries = [word.strip() for word in queries]
    revenue, revenue_iterations = compute_revenue(budgetDict, bidsDict, queries, type)

    print ('Revenue - ', str(round(revenue, 2)))
    competitive_ratio = revenue_iterations/sum(budgetDict.values())
    print ('Competitive Ratio - ', str(round(competitive_ratio, 2)))


# checks runtime arguments and accordingly runs the appropriate algorithm
def main(argv):

    algorithm = sys.argv[1]
    
    type = 0
    if algorithm == 'greedy':
      type = 1
    elif algorithm == 'msvv':
      type = 2
    elif algorithm == 'balance':
      type = 3
    else:
      print ('Please input a valid algorithm!')
      return

    readData(type)

if __name__ == "__main__":
   main(sys.argv[1:])