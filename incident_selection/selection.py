INFO = 1
MINOR = 2
MAJOR = 3
CRITICAL = 4
BLOCKER = 5

MAX_WEIGHT = 100

import json
import sys
import re
from dwave.system import LeapHybridSampler
from math import log2, floor
import dimod
import numpy

def processSeverity(severity):
    if severity == 'INFO':
        return INFO
    elif severity == 'MINOR':
        return MINOR
    elif severity == 'MAJOR':
        return MAJOR
    elif severity == 'CRITICAL':
        return CRITICAL
    elif severity == 'BLOCKER':
        return BLOCKER

def processEffort(effort):
    splitted = re.split('(\d+)', effort) #['', 'digit', 'unit']
    digit = splitted[1] #El número
    unit = splitted[2] #La unidad

    #Minutos
    if unit == 'min':
        return int(digit)
    else:
        return 0

def processSeverity(severity):
    if severity == 'BLOCKER':
        return 5
    elif severity == 'CRITICAL':
        return 4
    elif severity == 'MAJOR':
        return 3
    elif severity == 'MINOR':
        return 2
    elif severity == 'INFO':
        return 1
    else:
        return 0

def readIncidents(path):
    with open(path) as file:
        try:
            total = json.load(file)
        except:
            print('El archivo es incorrecto.')
            sys.exit()
    
    incidents = total['issues']
    return incidents

def processIncidents(incidents):
    length = len(incidents)
    costs = numpy.empty(length, dtype=object)
    weights = numpy.empty(length, dtype=object)    

    for i in range(len(incidents)):
        costs[i] = processEffort(incidents[i]['effort'])
        weights[i] = processSeverity(incidents[i]['severity'])

    return costs, weights

def buildBQM(costs, weights, weight_capacity):
    bqm = dimod.AdjVectorBQM(dimod.Vartype.BINARY)

    penalization = max(costs)
    x_size = len(costs)

    M = floor(log2(weight_capacity))
    num_slack_variables = M + 1

    y = [2**n for n in range(M)]
    y.append(weight_capacity + 1 - 2**M)

    #Término linear x
    for i in range(x_size):
        bqm.set_linear('x' + str(i), penalization * (weights[i]**2) - costs[i])

    #Término cuadrático xi-xj
    for i in range(x_size):
        for j in range(i + 1, x_size):
            key = ('x' + str(i), 'x' + str(j))
            bqm.quadratic[key] = 2 * penalization * weights[i] * weights[j]

    #Término linear y
    for i in range(num_slack_variables):
        bqm.set_linear('x' + str(i), penalization * (y[i]**2))

    #Término linear yi-yj
    for i in range(num_slack_variables):
        for j in range(i+1, num_slack_variables):
            key = ('y' + str(i), 'y' + str(j))
            bqm.quadratic[key] = 2 * penalization * y[i] * y[j]

    for i in range(x_size):
        for j in range(num_slack_variables):
            key = ('x' + str(i), 'y' + str(j))
            bqm.quadratic[key] = -2 * penalization * weights[i] * y[j]

    return bqm

def solve_knapsack(bqm):
    sampler = LeapHybridSampler()

    sampleset = sampler.sample(bqm)
    sample = sampleset.first.sample
    energy = sampleset.first.energy

    selected_item_indices = []
    for varname, value in sample.items():
        # For each "x" variable, check whether its value is set, which
        # indicates that the corresponding item is included in the
        # knapsack
            # The index into the weight array is retrieved from the
            # variable name
        if(sample[varname] == 1):
            selected_item_indices.append(varname)

    return sampleset, sample, energy, sorted(selected_item_indices)

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else "response.json"
    incidents = readIncidents(path)
    costs, weights = processIncidents(incidents)

    bqm = buildBQM(costs, weights, MAX_WEIGHT)
    sampleset, sample, energy, selected_item_indices = solve_knapsack(bqm)

    print("Found solution at energy {}".format(energy))
    print("Selected item numbers (0-indexed):", selected_item_indices)