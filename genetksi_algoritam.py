import random
import math
import time as tm



distance = {
        'A' : {'B':5,'C':5,'D':3,'E':2,'F':2,'G':1,'I':3,'H':1,'J':7,'K':2},
        'B' : {'A':5,'C':5,'D':4,'E':3,'F':6,'G':1,'I':4,'H':5,'J':21,'K':9},
        'C' : {'A':5,'B':5,'D':3,'E':7,'F':5,'G':42,'I':1,'H':6,'J':8,'K':21},
        'D' : {'A':3,'B':4,'C':3,'E':7,'F':7,'G':17,'I':10,'H':7,'J':6,'K':12},
        'E' : {'A':2,'B':3,'C':7,'D':7,'F':3,'G':2,'I':11,'H':32,'J':4,'K':7},
        'F' : {'A':2,'B':6,'C':5,'D':7,'E':3,'G':4,'I':5,'H':1,'J':2,'K':6},
        'G' : {'A':1,'B':5,'C':10,'D':11,'E':7,'F':14,'I':2,'H':2,'J':1,'K':5},
        'I' : {'A':3,'B':5,'C':10,'D':11,'E':32,'F':1,'G':2,'H':4,'J':4,'K':4},
        'H' : {'A':3,'B':5,'C':10,'D':11,'E':32,'F':1,'G':2,'I':3,'J':2,'K':3},
        'J' : {'A':4,'B':2,'C':5,'D':11,'E':12,'F':12,'G':12,'I':8,'H':3,'K':2},
        'K' : {'A':4,'B':2,'C':5,'D':11,'E':12,'F':12,'G':12,'I':8,'H':3,'J':2}
    }

time = {
        'A' : {'B':3,'C':5,'D':4,'E':5,'F':2,'G':2,'I':2,'H':9,'J':3,'K':21},
        'B' : {'A':3,'C':2,'D':2,'E':5,'F':3,'G':1,'I':12,'H':9,'J':2,'K':16},
        'C' : {'A':5,'B':2,'D':5,'E':6,'F':4,'G':1,'I':5,'H':5,'J':12,'K':23},
        'D' : {'A':4,'B':2,'C':5,'E':2,'F':5,'G':23,'I':4,'H':21,'J':5,'K':7},
        'E' : {'A':5,'B':5,'C':6,'D':2,'F':3,'G':1,'I':3,'H':5,'J':2,'K':2},
        'F' : {'A':2,'B':3,'C':4,'D':5,'E':3,'G':32,'I':2,'H':4,'J':1,'K':8},
        'G' : {'A':1,'B':5,'C':10,'D':11,'E':7,'F':14,'I':3,'H':3,'J':7,'K':7},
        'I' : {'A':3,'B':5,'C':10,'D':11,'E':32,'F':1,'G':2,'H':2,'J':5,'K':5},
        'H' : {'A':13,'B':5,'C':1,'D':3,'E':12,'F':1,'G':2,'I':1,'J':4,'K':3},
        'J' : {'A':4,'B':2,'C':5,'D':11,'E':12,'F':12,'G':12,'I':8,'H':3,'K':2},
        'K' : {'A':7,'B':3,'C':5,'D':1,'E':21,'F':2,'G':12,'I':8,'H':3,'J':2}
    }


cityList = ['A','B','C','D','E','F','G','I','H','J','K']
population_size = 20
tournament_size = 5
elitism_size = 2
mutation_rate = 0.1
generation_size = 3000

BEST_CHROMOSOME = None

class chromosome:
    def __init__(self,perm):
        self.permutation = perm
        self.fitness = 0
        self.time = self.calculate_time()
        self.dist = self.calculate_dist()

    def calculate_dist(self):
        sum = 0
        node = self.permutation[0]
        for i in range(1,len(self.permutation)):
            sum += distance[node][self.permutation[i]]
            node = self.permutation[i]
        sum += distance[self.permutation[len(self.permutation)-1]][self.permutation[0]]
        return sum
    
    def calculate_time(self):
        sum = 0
        node = self.permutation[0]
        for i in range(1,len(self.permutation)):
            sum += time[node][self.permutation[i]]
            node = self.permutation[i]
        sum += time[self.permutation[len(self.permutation)-1]][self.permutation[0]]
        return sum
    
    def __str__(self):
        return "Chromosome:" + ' ' + str(self.permutation) + ' ' + str(self.fitness) + ' ' + str(self.dist) + ' ' + str(self.time)

# TOURNAMENT SELECTION
def selection(population):
    tmp_population = population.copy()
    random_chromosomes = []
    
    for i in range(tournament_size):
        chr = random.choice(tmp_population)
        random_chromosomes.append(chr)
        tmp_population.remove(chr)
    random_chromosomes = sorted(random_chromosomes,key=lambda chr : chr.fitness,reverse=True)
    
    return random_chromosomes[0]

def crossover(parent1,parent2):
    par1 = parent1.permutation.copy()
    par2 = parent2.permutation.copy()
    tmp = par1.pop(0)
    par2.pop(0)
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(par1))
    geneB = int(random.random() * len(par2))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(par1[i])
        
    childP2 = [item for item in par2 if item not in childP1]

    child = childP1 + childP2
    child.insert(0,tmp)
    return child

def mutation(chr):
    code = chr.permutation
    if random.random() <= mutation_rate:
        i = random.randrange(1,len(code))
        j = random.randrange(1,len(code))
        code[i],code[j] = code[j],code[i]
    return chromosome(code)        

def calculate_fitness(population,alfa = 0.5,beta = 0.5):
    sum_dist = sum(x.dist for x in population)
    sum_time = sum(x.time for x in population)
    for chr in population:
        chr.fitness = 1 - (alfa * chr.dist / sum_dist + beta * chr.time / sum_time)
    return population

    
def swap(route):
    index = math.inf
    for i in range(len(route)):
        if route[i] == 'A':
            index = i
            break
    if index != math.inf:
        route[0],route[index] = route[index],route[0]
    # print(route)
    return route

def createRoute():
    route = random.sample(cityList, len(cityList))
    route = swap(route)
    return list(route)

def initial_population(size):
    population = []
    for i in range(size):
        chr = chromosome(createRoute())
        population.append(chr)
    return population

def print_population(population):
    print("#############POPULATION##############")
    for i in population:
        print(i)
    print("#############POPULATION##############")

start_time = tm.time()

population = initial_population(population_size)
calculate_fitness(population)
population = sorted(population,key = lambda chr : chr.fitness,reverse=True)
print_population(population)
BEST_CHROMOSOME = population[0]
print("BEST CHROMOSOME:",BEST_CHROMOSOME)


# GENETIC ALGORITHM
for i in range(generation_size):
    new_population = []
# ELITISM
    for i in range(elitism_size):
        new_population.append(chromosome(population[i].permutation))
# new population
    for i in range(elitism_size,population_size,2):
        par1 = selection(population)
        par2 = selection(population)
        child1_permutation = crossover(par1,par2)
        child2_permutation = crossover(par1,par2)
        chr1 = chromosome(child1_permutation)
        chr2 = chromosome(child2_permutation)
        chr1 = mutation(chr1)
        chr2 = mutation(chr2)
        new_population.append(chr1)
        new_population.append(chr2)

    calculate_fitness(new_population)
    new_population = sorted(new_population,key = lambda chr : chr.fitness,reverse=True)    
    population = new_population
    print_population(population)

    BEST_CHROMOSOME = population[0]
    print("THE BEST CHROMOSOME:",BEST_CHROMOSOME)

print("Vreme izvrsavanja programa: ",tm.time() - start_time)