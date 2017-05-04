# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 07:24:48 2016

@author: guillermo
"""

import random
from deap import base, creator, tools, algorithms
import numpy as np
import time


def genetic_search(eval_function, vector_size, population_size):
    """
    Performs genetic algoritm search using the given inputs
    """
    #TODO: allow to give a population for start (it will need to be mutated)
    start_time = time.time()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=vector_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    print('The search took %i minutes.' % ((start_time-time.time())/60))
    return pop, logbook, hof