routes = []
routes1 = []
def find_paths(node, cities,time, path, distance,path1,distance1):

    path.append(node)
    path1.append(node)

    if len(path) > 1:
        distance += cities[path[-2]][node]
        distance1 += time[path1[-2]][node]

    if (len(cities) == len(path)) and (path[0] in cities[path[-1]]):
        global routes
        global routes1
        path.append(path[0])
        path1.append(path1[0])
        distance += cities[path[-2]][path[0]]
        distance1 += time[path1[-2]][path1[0]]
        # print(path, distance)
        # print(path1, distance1)
        routes.append([distance, path])
        routes1.append([distance1, path1])
        return

    #recursion
    for city in cities:
        if (city not in path) and (node in cities[city]):
            find_paths(city, dict(cities), dict(time), list(path), distance, list(path1),distance1)


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

    find_paths('A', cities,time, [], 0, [], 0)
    print("\n")
    routes.sort()
    routes1.sort()
    
    print()
    print("Distance routes:")
    print()

    for i in routes:
        print(i)
    
    print()
    print("Time routes:")
    print()

    for i in routes1:
        print(i)

    if len(routes) != 0 and len(routes1) != 0:
        print("Shortest distance route: %s" % routes[0])
        print("Shortest time route: %s" % routes1[0])
    else:
        print("error")

    best_distance_route = routes[0]
    best_time_route = routes1[0]
    print("Best route depending on min time/distance:")
    if(best_distance_route <= best_time_route):
        print(best_distance_route)
    else:
        print(best_time_route)