from matplotlib.cbook import _premultiplied_argb32_to_unmultiplied_rgba8888
import pandas as pd
import math
import pickle
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import glob
import multiprocessing
import sys
import getopt
from termcolor import cprint
from pyfiglet import figlet_format
import os


def get_cell_to_voisin(p1, points, radius_min, radius_max):
    """List neigboors of cell p1 """
    neighbors = []
    for p2 in points:
        if p1 != p2:
            x = float(p1.split("_")[0])
            y = float(p1.split("_")[1])
            x_candidate = float(p2.split("_")[0])
            y_candidate = float(p2.split("_")[1])
            distance = math.sqrt((x - x_candidate)**2 + (y - y_candidate)**2)
            if distance <= radius_max and distance >= radius_min:
                neighbors.append(p2)
    return (p1, neighbors)


def read_cells_from_file(file_name:str) -> list:
    """Get list of cells in file_name """
    points = []
    df = pd.read_csv(file_name)
    for index, row in df.iterrows():
        x = row['Centroid X µm']
        y = row['Centroid Y µm']
        points.append(f"{x}_{y}")
    return points


def get_cell_to_voisin_multiproc(points, d_min, d_max):
    """Multiprocessing neigborhood computation"""
    with multiprocessing.Pool() as pool:
        results = pool.starmap(get_cell_to_voisin, [(p1, points, d_min, d_max) for p1 in points])
    cell_to_voisin = {p1: neighbors for p1, neighbors in results}
    return cell_to_voisin


def map_cells_to_pop(data_file:str)->dict:
    """ """
    cell_to_pop = {}
    df = pd.read_csv(data_file)
    for index, row in df.iterrows():
        x = row['Centroid X µm']
        y = row['Centroid Y µm']
        population = row['OmiqFilter']
        cell_id = f"{x}_{y}"
        cell_to_pop[cell_id] = population

    return cell_to_pop


def get_pop_to_voisin(cell_to_voisin, cell_to_pop):
    """ """

    all_pop_in_file = []
    for pop in list(cell_to_pop.values()):
        if pop not in all_pop_in_file:
            all_pop_in_file.append(pop)

    pop_to_voisin = {}
    for p1 in all_pop_in_file:
        pop_to_voisin[p1] = {}
        for p2 in all_pop_in_file:
            pop_to_voisin[p1][p2] = 0
    
    for c in cell_to_voisin:
        p = cell_to_pop[c]
        for cv in cell_to_voisin[c]:
            pv = cell_to_pop[cv]
            pop_to_voisin[p][pv] += 1
    return pop_to_voisin

def get_cellpop_list_from_folder(folder_file):
    """ """
    cell_pop_list = []
    for tf in glob.glob(f"{folder_file}/*.csv"):
        df = pd.read_csv(tf)
        if 'OmiqFilter' in list(df.keys()):
            pop_list = list(df['OmiqFilter'])
            for pop in pop_list:
                if pop not in cell_pop_list:
                    cell_pop_list.append(pop)
    return cell_pop_list
    

def compute_proximity_matrix(radius_min, radius_max, data_file):
    """ """

    # load data
    cell_list = read_cells_from_file(data_file)
    
    # compute distances
    cell_to_voisin = get_cell_to_voisin_multiproc(cell_list, radius_min, radius_max)
    cell_to_pop = map_cells_to_pop(data_file)
    pop_to_voisin = get_pop_to_voisin(cell_to_voisin, cell_to_pop)

    # return proximity matrix
    return pop_to_voisin 


def compute_proximity_matrix_folder(folder, manifest, radius_min, radius_max):
    """ """

    # ectract class
    class_to_file = {}
    df = pd.read_csv(manifest)
    for index, row in df.iterrows():
        if row['Groupe'] not in class_to_file:
            class_to_file[row['Groupe']] = [f"{folder}/{row['file']}"]
        else:
            class_to_file[row['Groupe']].append(f"{folder}/{row['file']}")

    # extract pop list
    pop_list = []
    for tf in glob.glob(f"{folder}/*.csv"):
        if tf != manifest:
            local_list = pd.read_csv(tf)['OmiqFilter'].unique()
            for pop in local_list:
                if pop not in pop_list:
                    pop_list.append(pop)

    # for each class compute pop to voisin
    class_to_pop_to_voisin = {}
    for c in class_to_file:
        file_list = class_to_file[c]
        dict_list = []
        for f in file_list:
            pop_to_voisin = compute_proximity_matrix(radius_min, radius_max, f)
            dict_list.append(pop_to_voisin)

        # assemble results
        result = {}
        for d in dict_list:
            for pop in d:
                if pop not in result:
                    result[pop] = d[pop]
                else:
                    for p2 in result[pop]:
                        if p2 in d[pop]:
                            result[pop][p2] += d[pop][p2]

        # assign pop to voisin
        class_to_pop_to_voisin[c] = result

    # return class to pop_to_voisin
    return class_to_pop_to_voisin

    


def display_proximity_matrix(pop_to_voisin):
    """ """

    # craft vector list
    vector_list = []
    vector_list_percentage = []
    for k1 in pop_to_voisin:
        vector = []
        for k2 in pop_to_voisin:
            if k2 in pop_to_voisin[k1]:
                scalar = pop_to_voisin[k1][k2]
            else:
                scalar = 0
            vector.append(scalar)
        vector_list.append(vector)

        # deal with percentages
        total = sum(vector)
        percentages = [(value / total) * 100 for value in vector]
        vector_list_percentage.append(percentages)
        print(vector_list_percentage)

    # plot graph
    ax = sns.heatmap(
        vector_list_percentage,
        linewidth=0.5,
        annot=True,
        xticklabels = list(pop_to_voisin.keys()),
        yticklabels = list(pop_to_voisin.keys())
    )
    plt.title('Proximity Matrix (%)')
    plt.show()
   

def generate_proximity_matrix_heatmap(pop_to_voisin, fig_name):
    """ """

    import numpy as np

    # craft vector list
    vector_list = []
    vector_list_percentage = []
    for k1 in pop_to_voisin:
        vector = []
        for k2 in pop_to_voisin:
            if k2 in pop_to_voisin[k1]:
                scalar = pop_to_voisin[k1][k2]
            else:
                scalar = 0
            vector.append(scalar)
        vector_list.append(vector)

        # deal with percentages
        total = 0
        for s in vector:
            if str(s) != 'nan':
                total +=s
        percentages = []
        for v in vector:
            if str(v) != 'nan':
                p = v / total *100
            else:
                p = np.nan
            percentages.append(p)
        vector_list_percentage.append(percentages)

    # plot graph
    ax = sns.heatmap(
        vector_list_percentage,
        linewidth=0.5,
        annot=True,
        xticklabels = list(pop_to_voisin.keys()),
        yticklabels = list(pop_to_voisin.keys())
    )
    plt.title('Proximity Matrix (%)')
    plt.savefig(fig_name)
    plt.close()



def display_voisin_bar(pop_to_voisins, pop):
    """ """

    # load data
    data = pop_to_voisins[pop]
    data = dict(sorted(data.items(), key=lambda item: item[1]))

    # compute percentage
    total = sum(data.values())
    data_percentage = {}
    for k in data:
        data_percentage[k] = (data[k] / total)*100
      
    # display
    plt.barh(data_percentage.keys(), data_percentage.values())
    plt.title(f"Voisinage de {pop}")
    plt.show()



def display_voisin_pie(pop_to_voisins, pop):
    """ """

    # load data
    data = pop_to_voisins[pop]
    data = dict(sorted(data.items(), key=lambda item: item[1]))

    # Extracting the labels and sizes from the dictionary
    labels = data.keys()
    sizes = data.values()

    # Plotting the pie chart
    plt.figure(figsize=(6, 6))  # Optional: Adjust the figure size
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    # Display the pie chart
    plt.title(f"Voisinage de {pop}")
    plt.show()



def display_multiclass_bar(group_to_voisin, pop):
    """ """

    data = []
    group_list = []
    for group in group_to_voisin:
        data.append(group_to_voisin[group][pop])
        group_list.append(group)

    # Conversion des valeurs en pourcentages
    pop_list = []
    for series in data:
        total = sum(series.values())
        for key in series:
            series[key] = (series[key] / total) * 100
            if key not in pop_list:
                pop_list.append(key)
    
    # fill missing pops
    for series in data:
        for pop in pop_list:
            if pop not in series:
                series[pop] = 0

    # create figure
    categories = list(data[0].keys())
    n_categories = len(categories)
    bar_width = 0.2  
    index = np.arange(n_categories)
    for i, series in enumerate(data):
        values = list(series.values())
        plt.bar(index + i * bar_width, values, bar_width, label=group_list[i])
    plt.xlabel('Populations')
    plt.ylabel('Valeurs (%)')
    plt.title(f"Voisins de la population {pop} (valeurs en %)")
    plt.xticks(index + bar_width, categories, rotation=45)
    plt.legend()
    plt.show()


    
def display_class_matrix(group_to_voisin, group):
    """ """
    pop_to_voisin = group_to_voisin[group]
    display_proximity_matrix(pop_to_voisin)    


def display_help():
    """ """

    help = """


    This program computes a matrix of neighbors from one or multiple FCS files.

    Arguments:
    * input: Path to the FCS file or a directory containing multiple FCS files.
    * output: Path to the output directory where results will be saved.
    * radius_min: Minimum radius used to compute the neighborhood.
    * radius_max: Maximum radius used to compute the neighborhood.
    
    Example Usage:
        
        spice.py -i 'path/to/my/fcs_file' -o '/path/to/my/output/dir' -r 1 -R 10

    """

    print(help)



def run(input, output, radius_min, radius_max):
    """ """


    # init output folder
    if not os.path.isdir(output):
        os.mkdir(output)

    # compute matrix for one file
    if os.path.isfile(input):

        # compute matrix
        matrix = compute_proximity_matrix(radius_min, radius_max, input)

        # save matrix
        with open(f"{output}/matrix.pickle", 'wb') as handle:
            pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save matrix as csv
        df = pd.DataFrame(matrix)
        df.to_csv(f"{output}/matrix.csv", index=False)
        df_percentage = df.div(df.sum(axis=1), axis=0) * 100
        df_percentage.to_csv(f"{output}/matrix_percentage.csv", index=False)

        # generate figures
        display_proximity_matrix(matrix)


    # compute matrix for one folder
    if os.path.isdir(input):

        ## collect list of cell pop
        cell_pop_list = get_cellpop_list_from_folder(input)

        ## loop over roi files
        for roi_file in glob.glob(f"{input}/*.csv"):

            ## compute matrix
            matrix = compute_proximity_matrix(radius_min, radius_max, roi_file)
    
            ## adjsut matrix for missing pop
            for c in cell_pop_list:
                if c not in matrix:
                    matrix[c] = {}
                    for c2 in cell_pop_list:
                        matrix[c][c2] = np.nan
                else:
                    for c2 in cell_pop_list:
                        if c2 not in matrix[c]:
                            matrix[c][c2] = np.nan
    
            # reorder matrix
            matrix_clean = {}
            for c1 in cell_pop_list:
                matrix_clean[c1] = {}
                for c2 in cell_pop_list:
                    matrix_clean[c1][c2] = matrix[c1][c2]
            matrix = matrix_clean

            # save matrix
            matrix_filename = roi_file.split("/")[-1].replace('.csv', '_matrix.pickle')
            with open(f"{output}/{matrix_filename}", 'wb') as handle:
                pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # generate heatmap
            fig_name = roi_file.split("/")[-1].replace('.csv', '_heatmap.png')
            generate_proximity_matrix_heatmap(matrix, f"{output}/{fig_name}")
    
            # save matrix as csv
            matrix_file = roi_file.split("/")[-1].replace('.csv', '_matrix.csv')
            df = pd.DataFrame(matrix)
            df.to_csv(f"{output}/{matrix_file}")
            df_percentage = df.div(df.sum(axis=1), axis=0) * 100
            df_percentage.to_csv(f"{output}/{matrix_file.replace('.csv', '_percentage.csv')}")




if __name__ == "__main__":

    # default values
    input = ""
    output = ""
    radius_min = 5
    radius_max = 10

    # catch arguments
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(
            argv, "hi:o:r:R:", ["input=", "output=", "radius_min=", "radius_max"]
        )
    except getopt.GetoptError:
        display_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            display_help()
            sys.exit()
        elif opt in ("-i", "--input"):
            input = arg
        elif opt in ("-o", "--output"):
            output = arg
        elif opt in ("-r", "--radius_min"):
            radius_min = arg
        elif opt in ("-R", "--radius_max"):
            radius_max = arg

    # display cool banner
    text = " =====\n> SPICE <\n =====\n"
    cprint(figlet_format(text, font="standard"), "yellow")

    # check that all arguments are present
    if input == "":
        print("[!] No input specified")
        print("[!] Use -h or --help options to get more informations")
        sys.exit()
    if output == "":
        print("[!] No output folder specified")
        print("[!] Use -h or --help options to get more informations")
        sys.exit()

    
    # run
    run(input, output, radius_min, radius_max)



    



    

    
