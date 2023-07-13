import numpy as np
import pandas as pd
from utils import *
import csv
import matplotlib.pyplot as plt
import seaborn as sb

containers_color = {
    'ausf':'tab:red',
    'udm': 'tab:blue',
    'udr': 'tab:green'
}


def data_processing(data, label, paths):
    raw_data1 = pd.read_csv(paths[0])
    values1 = raw_data1.iloc[:, -1].tolist()

    if len(paths)>1:
        try:
            raw_data2 = pd.read_csv(paths[1])
            values2 = raw_data2.iloc[:, -1].tolist()
        except pd.errors.EmptyDataError:
            values2 = []
    else:
        values2 = []
    values = values1 + values2

    data[label] = {}
    data[label]['processing_times'] = values

    # Compute the quartiles
    first_quartile = np.percentile(values, 25)
    median = np.percentile(values, 50)
    ninetieth_quartile = np.percentile(values, 90)
    data[label]['quartiles'] = [first_quartile, median, ninetieth_quartile]

    return data


def get_quartiles(results, output, type, deployments=2, cumulative=True, oneGraph=True):
    suffix1 = '/ue-data/ue1-benign_registration_time.csv'
    suffix2 = '/ue-data/ue1_registration_time.csv'
    data = {}

    for path, label in results.items():
        paths = []
        for d in range(deployments):
            paths.append(path + f'/ue-data/ue{d+1}-benign_registration_time.csv')
        """if 'Ghost' in label:
            paths = [path+'/ue-data_default/ue1-benign_registration_time.csv']
            data = data_processing(data, 'No storm', paths)"""
        data = data_processing(data, label, paths)

        #print(f'--  UE ({label})  --\n', data[label]['quartiles'])


    dataKeys, dataValues = [],[]
    for key, value in data.items():
        dataKeys.append(key) 
        dataValues.append(value['processing_times'])
        #print(key)
        #print(value['processing_times'])

    if type == 'box':
        plt.figure()
        positions = [i+1 for i in range(len(dataKeys))]
        plt.boxplot(dataValues, positions=positions,showfliers=True)
        plt.title('Comparison of initial registration time')
        plt.ylabel('Registration processing time (Second)')
        plt.xticks(positions, dataKeys)
    elif type == 'density':
        if oneGraph:
            plt.figure()

            # for scenario in range(len(dataKeys)):
            #     n = np.arange(1,len(dataValues[scenario])+1) / np.float(len(dataValues[scenario]))
            #     Xs = np.sort(dataValues[scenario])
            #     plt.step(Xs,n, label=dataKeys[scenario])
            # plt.title(f'{dataKeys[scenario]}')

            for scenario in range(len(dataKeys)):
                sb.ecdfplot(dataValues[scenario], label=dataKeys[scenario])

            plt.xlabel("Registration Processing Time (Second)")
            plt.ylabel("Probability") 
            plt.legend()

        else:    
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            positions = [(0,0), (1,0), (1,1), (0,1)]

            for scenario in range(len(dataKeys)):
                posx, posy = positions[scenario]
                sb.kdeplot(pd.to_numeric(dataValues[scenario]),label=dataKeys[scenario], cumulative=cumulative, bw_adjust = 0.3, fill = True, ax=axs[posx,posy])
                #axs[posx,posy].set_ylim(0, 0.3)
                axs[posx,posy].set_title(f'{dataKeys[scenario]}')
                if posx==1:
                    axs[posx,posy].set_xlabel("Registration Processing Time (Second)")
                if posy==0:
                    axs[posx,posy].set_ylabel("Fraction Of Total")
                axs[posx,posy].set_xlim(0,10)

    elif type == 'densityNbar':
        if oneGraph:
            print("TODO")
        else:
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            positions = [(0,0), (1,0), (1,1), (0,1)]

            for scenario in range(len(dataKeys)):
                posx, posy = positions[scenario]
                sb.distplot(dataValues[scenario], label=dataKeys[scenario], hist=True, kde=True, ax=axs[posx,posy], bins=60)
                axs[posx,posy].set_ylim(0, 0.3)
                axs[posx,posy].set_title(f'{dataKeys[scenario]}')
                if posx==1:
                    axs[posx,posy].set_xlabel("Registration Processing Time (Second)")
                if posy==0:
                    axs[posx,posy].set_ylabel("Fraction Of Total")

    elif type == 'violin':
        # Find the maximum length among the lists
        max_length = max(len(lst) for lst in dataValues)
        # Pad the lists with NaN values to make them the same length
        dataValues_padded = [lst + [np.nan] * (max_length - len(lst)) for lst in dataValues]
        df = pd.DataFrame({dataKeys[i]:dataValues_padded[i] for i in range(len(dataKeys))})
        sb.violinplot(data=df, width=1.4)

    elif type == 'bars':
        histtype = 'step' if cumulative else 'bar'
        myBins = 500 if cumulative else 60
        if oneGraph:
            plt.figure(figsize=(8,4))

            tab_colors = plt.get_cmap('tab10').colors
            checkpoints = []
            for scenario in range(len(dataKeys)):
                n,bins,patches = plt.hist(dataValues[scenario], histtype=histtype, bins=myBins, range=(0,10), cumulative=cumulative, label=dataKeys[scenario], density=True, color=tab_colors[scenario])
                
                subList1 = [val for val in dataValues[scenario] if val<1]
                checkpt = float(len(subList1))/len(dataValues[scenario])
                checkpoints.append(checkpt)

                plt.hlines(checkpt,0,1, linestyle="dashed", color=tab_colors[scenario], linewidth=1, dashes=(0,(5,5)))
                plt.xlim(0,10)
                # yList = []
                # hist, bin_edges = np.histogram(dataValues[scenario], bins=myBins)
                #plt.plot(hist, bin_edges[:-1], label=dataKeys[scenario])
                # Filter out the values that are zero
                # non_zero_indices = np.nonzero(hist)
                # filtered_bins = bin_edges[non_zero_indices]
                # filtered_n = n[non_zero_indices]
            existing_ticks = plt.yticks()[0]
            ticks = np.unique(np.concatenate((existing_ticks, checkpoints)))
            #plt.yticks(ticks, ticks)
            # Adjusting the size of specific ticks
            # for tick in checkpoints:
            #     plt.tick_params(axis='y', which='both', labelsize=8, where=[tick in checkpoints])

            plt.vlines(1,0,max(checkpoints), color='gray', linestyle="dashed", linewidth=1, dashes=(0,(5,10)))
            plt.legend(loc=4)


        else:
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            positions = [(0,0), (1,0), (1,1), (0,1)]
            edgecolor = 'black' if cumulative else 'blue'

            for scenario in range(len(dataKeys)):
                posx, posy = positions[scenario]
                axs[posx,posy].hist(dataValues[scenario], 
                                    histtype=histtype, 
                                    bins=myBins, 
                                    range=(0,10), 
                                    cumulative=cumulative, 
                                    weights=np.ones(len(dataValues[scenario])) / len(dataValues[scenario]), 
                                    edgecolor=edgecolor,
                                    color='steelblue')
                if not cumulative:
                    axs[posx,posy].set_ylim(0, 0.3)
                axs[posx,posy].set_title(f'{dataKeys[scenario]}')
                if posx==1:
                    axs[posx,posy].set_xlabel("Registration Processing Time (Second)")
                if posy==0:
                    axs[posx,posy].set_ylabel("Fraction Of Total")
        plt.xlabel("Registration Processing Time (Second)")
        plt.ylabel("Probability")




    plt.savefig(f'{output}{"general_" if oneGraph else ""}{type}_registrations.png')
    plt.close()


def get_data(filename, offset=0):
    data_file = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Convert the data to float
            x = float(row[0]) - offset
            y = float(row[1])
            data_file.append((x, y))
    x_values = [row[0] for row in data_file]
    y_values = [row[1] for row in data_file]
    return x_values, y_values


def generate_graphs(paths, y_max, output, deployments=2, withAtk=False):
    graphs = {}
    for filename in paths:
        one_filename = filename
        break

    x_waves = []
    with open(one_filename + '/ue-data/ue1_wave_time.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Convert the data to float
            x = float(row[0])
            x_waves.append(x)
    origin, end = x_waves[1]-10, x_waves[1]+240

    for filename in paths:
        x_values, y_values = [],[]
        for d in range(deployments):
            times_file_storm = filename + f'/ue-data/ue{d+1}_registration_time.csv'
            x_tmp, y_tmp = get_data(times_file_storm, origin)
            x_values += x_tmp
            y_values += y_tmp
        graphs[f'{paths[filename]}-{filename.split("/")[-1]}']= [[x_values, y_values],[]]

        x_values, y_values = [],[]
        for d in range(deployments):
            times_file_benign = filename + f'/ue-data/ue{d+1}-benign_registration_time.csv'
            x_tmp, y_tmp = get_data(times_file_benign, origin)
            x_values += x_tmp
            y_values += y_tmp
        graphs[f'{paths[filename]}-{filename.split("/")[-1]}'][1]= [x_values, y_values]
        if 'Ghost' in paths[filename]:
            ghosts_x, ghosts_y = x_values, y_values

        #========================================
        packets_file = filename + '/data/'
        filepaths = []
        data_file = []
        # Iterate over files in the folder
        for file in os.listdir(packets_file):
            file_path = os.path.join(packets_file, file)
            if os.path.isfile(file_path):
                file_metric = file.split('-')[-1].split('.')[0]
                if file_metric == 'irate(container_network_receive_packets_total':
                    filepaths.append(file_path)

        # Create a new figure for each metric group
        plt.figure()

        graphs[f'{paths[filename]}-{filename.split("/")[-1]}'][1]= [x_values, y_values]

        for file in filepaths:
            # Plotting the figure for the group of files
            X, Y = [],[]
            with open(file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for data_point in reader:
                    x, y = map(float, data_point)
                    X.append(x - origin)
                    Y.append(y)
            data = X,Y

            values = data[1]
            container = file.split('/')[-1].split('-')[0]

            linestyle = '-'
            if container in ['udm', 'udr', 'ausf']:
                plt.plot(data[0], values, label=container.upper(), color=containers_color[container], linestyle=linestyle)


                # Set the labels and title
                title, ylabel = f'[{paths[filename]}] Rate of packets received per NF', "Packets/s"
                plt.xlabel('Time (Second)')
                plt.ylabel(ylabel)
                plt.title(title)
                plt.ylim(0,800)
                plt.xlim(0, end-origin)  # Adjust the range as needed
                plt.legend(loc=1)

        # Save the chart as an image
        plt.savefig(f'{output}/{paths[filename]}-{filename.split("/")[-1]}_kpi')

        # Close the figure to free up resources
        plt.close()
    

    nameList = []
    fig, axs = plt.subplots(1,4, figsize=(22,4))
    positions = [0, 1, 2, 3]
    index = 0
    sec_first = True
    for name, values in graphs.items():
        pos = positions[index]
        nameList.append(name)
        x_storm, y_storm = values[0]
        x_benign, y_benign = values[1]

        
        if not 'No Storm' in name:
            first = True
            for t_wave in x_waves:
                if first:
                    axs[pos].axvline(x=t_wave, color='red', label='Storms', linestyle='dotted')
                    first = False
                else:
                    axs[pos].axvline(x=t_wave, color='red', linestyle='dotted')

        my_title = f'{name.split("-")[0]}'
        axs[pos].plot(x_benign, y_benign, 'o', label=f'Benign registrations', color='tab:green', alpha=0.3)
        axs[pos].set_xlabel('Registration Start Time (Second)',)
        if sec_first:
            axs[pos].set_ylabel('Registration Processing Time (Second)')
            sec_first = False
        axs[pos].set_title(my_title)
        #plt.grid(False)
        #plt.legend(loc=1)
        axs[pos].set_xlim(0, end-origin)  # Adjust the range as needed
        axs[pos].set_ylim(0, y_max)  # Adjust the range as needed
        index += 1

    # Save the figure
    plt.savefig(f'{output}analysis_rt.png')

    print(f'Charts generated.')


if __name__ == "__main__":
    prefix = '../results/'
    output = '../results/analysis/'


    results = {
        f'{prefix}Simulation-1688920339': 'No Storm',
        f'{prefix}Simulation-1688871852': 'Storm With No Defense',
        f'{prefix}Simulation-1688864626': 'Storm With Baseline',
        f'{prefix}Simulation-1688916026': 'Storm With Blockchain 5GAKA'            
    }


    max_time = 10
    

    generate_graphs(results, max_time, output, deployments=2)


    get_quartiles(results, output, 'bars', cumulative=False, oneGraph=False)


    get_quartiles(results, output, 'density', cumulative=True, oneGraph=True)