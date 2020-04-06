import math

routes = []

def find_paths(node, cities,time, path, distance,distance1):

    path.append(node)

    if len(path) > 1:
        distance += cities[path[-2]][node]
        distance1 += time[path[-2]][node]

    if (len(cities) == len(path)) and (path[0] in cities[path[-1]]):
        global routes
        path.append(path[0])
        distance += cities[path[-2]][path[0]]
        distance1 += time[path[-2]][path[0]]
        routes.append([distance,distance1, path])
        return

    #recursion
    for city in cities:
        if (city not in path) and (node in cities[city]):
            find_paths(city, dict(cities), dict(time), list(path), distance,distance1)

def tezinska_fja(alfa,beta,skal_duzina,skal_vreme):
    return alfa * skal_duzina + beta * skal_vreme

if __name__ == '__main__':
    cities = {
        'A' : {'B':5,'C':5,'D':3,'E':2},
        'B' : {'A':5,'C':5,'D':4,'E':3},
        'C' : {'A':5,'B':5,'D':3,'E':7},
        'D' : {'A':3,'B':4,'C':3,'E':7},
        'E' : {'A':2,'B':3,'C':7,'D':7}
    }

    time = {
        'A' : {'B':3,'C':5,'D':4,'E':5},
        'B' : {'A':3,'C':2,'D':2,'E':5},
        'C' : {'A':5,'B':2,'D':5,'E':6},
        'D' : {'A':4,'B':2,'C':5,'E':2},
        'E' : {'A':5,'B':5,'C':6,'D':2}
    }

    find_paths('A', cities,time, [], 0, 0)
    print("\n")
    print("Velicina svih ruta je: ",len(routes))
    print("distance/time/route")
    for i in routes:
        print(i)
    
    sum_dist = 0
    sum_time = 0
    for i in routes:
        sum_dist += i[0]
        sum_time += i[1]
    for i in range(len(routes)):
        procenat_dist = routes[i][0]/sum_dist
        procenat_time = routes[i][1]/sum_time
        routes[i].append(procenat_dist)
        routes[i].append(procenat_time)
        routes[i].append(tezinska_fja(0.5,0.5,procenat_dist,procenat_time))
    print()
    print("#####################################")
    print("Nove rute:")
    print("duzina/vreme/rute/norm_duzina/norm_vreme/tezinska_fja")
    for i in sorted(routes,key=lambda route:route[5]):
        print(i)
    
    
    
        
    