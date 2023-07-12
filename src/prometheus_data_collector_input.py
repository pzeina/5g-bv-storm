import requests
import csv
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import subprocess
import multiprocessing
import os
import shutil
import time
import schedule


title_and_label_for_metrics = {
    'container_fs_usage_bytes':['Bytes_used','Bytes'],
    'irate(container_fs_usage_bytes':['Bytes_used', 'bytes/s'],
    'rate(container_cpu_usage_seconds_total':['CPU_Usage',''],
    'container_cpu_usage_seconds_total':['CPU_usage',''],
    'container_network_receive_packets_total':['Total_received_packets', 'packets'],
    'container_network_receive_packets_dropped_total':['Packets_dropped', 'packets'],
    'irate(container_network_transmit_packets_total':['Rate_of_transmitted_packets','packets/s'],
    'irate(container_network_receive_packets_total':['Rate_of_received_packets','packets/s'],
    'rate(container_network_receive_packets_total':['Rate_of_received_packets','packets/s'],
    'irate(container_network_receive_packets_dropped_total':['Rate_of_received_packets_dropped','packets/s']
}

metric_lists = [
    #['container_fs_usage_bytes','','container'], 
    ['irate(container_fs_usage_bytes','[1m])','container'],
    ['rate(container_cpu_usage_seconds_total','[1m])', 'container'],
    #['container_cpu_usage_seconds_total','','pod'],
    ['container_network_receive_packets_total','', 'pod'],
    ['container_network_receive_packets_dropped_total','','pod'],
    ['irate(container_network_receive_packets_total','[1m])', 'pod'],
    ['rate(container_network_receive_packets_total','[1m])', 'pod'],
    ['irate(container_network_receive_packets_dropped_total','[1m])','pod'],
    ['irate(container_network_transmit_packets_total','[1m])','pod']
]


interfaces = ['n2', 'n3', 'n4', 'n6']


def get_pod(container, prefix='free5gc'):
    part1 = f"kubectl get pods -n paul | awk '/{prefix}-"
    part2 = "/ {print $1;exit}'"
    command = part1 + container + part2
    output = exec_command(command, getRes=True)
    # print(output)
    return output


def exec_command(command, getRes=False):
    if getRes:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode().strip()  # Decode and remove leading/trailing whitespace

        return output
    subprocess.run(command, shell=True)


containers = {
    'amf1':[['container','amf'],['pod',get_pod('amf1')]],
    'amf2':[['container','amf'],['pod',get_pod('amf2')]],
    'ausf':[['container','ausf']], 
    'udm':[['container','udm']], 
    'smf':[['container','smf']], 
    'upf':[['container','upf']], 
    'udr':[['container','udr']], 
    'gnb1':[['container','gnb'],['pod',get_pod('gnb1', prefix='ueransim')]],
    'gnb2':[['container','gnb'],['pod',get_pod('gnb2', prefix='ueransim')]],
    'ue1':[['container','ue'],['pod',get_pod('ue1-benign', prefix='ueransim')]],
    'ue2':[['container','ue'],['pod',get_pod('ue2-benign', prefix='ueransim')]]
    }


containers_color = {
    'amf1': 'tab:red',
    'amf2': 'tab:orange',
    'ausf':'tab:pink',
    'udm': 'tab:purple',
    'upf': 'tab:blue',
    'udr': 'tab:gray',
    'smf': 'tab:cyan',
    'gnb1': 'tab:green',
    'gnb2': 'tab:olive',
    'ue1':  'tab:gray',
    'ue2': 'tab:brown'
}

containers_complements = {'amf1':[['interface','n3']], 'amf2':[['interface','n3']], 'upf':[['interface','n3']], 'smf':[['interface','n4']]}

# Format of an entry:   {container: [[label, value], ...], ...}
pods = {container: [['pod',get_pod(container)]]+containers_complements.get(container, []) for container in containers if container not in ['gnb1','gnb2', 'ue1', 'ue2']}


labels_dict = {
    'container': containers,
    'pod': pods
}

queries_with_metric = {}

PROMETHEUS_URL = "http://129.97.168.51:30090"

def update_pods():
    global containers, pods
    containers = {
        'amf1':[['container','amf'],['pod',get_pod('amf1')]],
        'amf2':[['container','amf'],['pod',get_pod('amf2')]],
        'ausf':[['container','ausf']], 
        'udm':[['container','udm']], 
        'smf':[['container','smf']], 
        'upf':[['container','upf']],
        'udr':[['container','udr']], 
        'gnb1':[['container','gnb'],['pod',get_pod('gnb1', prefix='ueransim')]],
        'gnb2':[['container','gnb'],['pod',get_pod('gnb2', prefix='ueransim')]],
        'ue1':[['container','ue'],['pod',get_pod('ue1-benign', prefix='ueransim')]],
        'ue2':[['container','ue'],['pod',get_pod('ue2-benign', prefix='ueransim')]]
        }

    pods = {container: [['pod',get_pod(container)]]+containers_complements.get(container, []) for container in containers if container not in ['gnb1','gnb2', 'ue1', 'ue2']}



def get_prometheus_data(query, start_time, end_time, step, debug=False):
    url = PROMETHEUS_URL + '/api/v1/query_range'

    end_time = datetime.now(timezone.utc).timestamp()

    if debug:
        print(query)

    params = {
        'query': query,
        'start': str(start_time),
        'end': str(end_time),
        'step': str(step)
    }


    # print(str(params))

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract values from the range vector
        values = []
        for result in data['data']['result']:
            metric_values = result['values']
            values.extend([[max(float(value[0])-start_time,0), float(value[1])] for value in metric_values])
        if len(values) == 0 and debug:
            print(data)
        return values

    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return None


def save_to_csv(data, metric, label, folder):
    filename = folder + label + '-' + metric + '.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        try:
            writer.writerows(data)
        except:
            print(data)


def read_from_csv(filename):
    X, Y = [],[]
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for data_point in reader:
            x, y = map(float, data_point)
            X.append(x)
            Y.append(y)
    return X,Y


def plot_data_from_csv(source_folder_path, dest_folder_path, debug=False):

    # Dictionary to store file groups
    file_groups = {}

    # Iterate over files in the folder
    for filename in os.listdir(source_folder_path):
        file_path = os.path.join(source_folder_path, filename)
        if os.path.isfile(file_path):
            file_metric = filename.split('-')[-1].split('.')[0]
            if file_metric not in file_groups:
                file_groups[file_metric] = []
            file_groups[file_metric].append(file_path)

    # Perform actions on file groups
    for file_metric, files in file_groups.items():
        # Perform specific action on each file group
        if debug:
            print('Generating chart for metric {}...'.format(file_metric))
        # Create a new figure for each metric group
        plt.figure()

        for filename in files:
            print(filename)
            # Plotting the figure for the group of files
            data = read_from_csv(filename)

            if len(data[0]) > 0:
                values = data[1]
                container = filename.split('/')[-1].split('-')[0]
                if 'ue' in container:
                    linestyle = '--'
                else:
                    linestyle = '-'
                plt.plot(data[0], values, label=container, color=containers_color[container], linestyle=linestyle)
            elif debug:
                print('[WARNING]  ', filename, "is empty")
        
        # Set the labels and title
        plt.xlabel('Time (s)')
        title, ylabel = title_and_label_for_metrics[file_metric]

        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()

        # Save the chart as an image
        plt.savefig(dest_folder_path + title + '.png')

        # Close the figure to free up resources
        plt.close()
    
    if debug:
        print('Charts created.')


def generate_logs(output_folder, deployments=1):
    entities = ["free5gc-upf", "free5gc-udm", "free5gc-ausf", "free5gc-udr", "free5gc-smf"]
    commands = []
    for pod in range(deployments):
        entities.append(f'free5gc-amf{pod+1}')
        entities.append(f'ueransim-gnb{pod+1}')

    for item in entities:
        output_file = output_folder + item + ".txt"
        command = "kubectl logs -f deployments/" + item + " -n paul > " + output_file
        commands.append(command)
    
    # Create a process for each command
    processes = []
    for command in commands:
        process = multiprocessing.Process(target=exec_command, args=(command,))
        processes.append(process)
        process.start()
    
    # Wait for all processes to complete
    for process in processes:
        process.join()


def query_constructor(metric, label_entries):
    query = metric[0]
    labels = ''
    for label_entry in label_entries:
        name, value = label_entry
        labels += f'{name}="{value}",'
    query += '{' + labels + 'namespace="paul"}'
    query += metric[1]

    return query, metric[0]


def data_collection(metric_lists, labels_dict, start_time, end_time, step, dest_folder_path, debug=False):
    update_pods()
    if debug:
        print("Sending queries to Prometheus...")
    # Form queries from metrics and containers
    for metric_items in metric_lists:
        labels_list = labels_dict[metric_items[-1]]
        for label, label_entries in labels_list.items():
            query, metric = query_constructor(metric_items, label_entries)
            queries_with_metric[query] = metric
            if debug:
                print('  ', query)
            data = get_prometheus_data(query, start_time, end_time, step, debug)

            #if 'free5gc' in label:
            #    label = label.split('-')[1]

            save_to_csv(data, metric, label, dest_folder_path)
    if debug:
        print("Data collected.")


def commit_results(destination_directory, withDefault=False):
    shutil.copy('random_times_1.csv', '../tmp/ue-data/random_times_1.csv')
    shutil.copy('random_times_2.csv', '../tmp/ue-data/random_times_2.csv')

    # Define the source and destination paths
    current_path = os.getcwd()
    source_folders = ["../tmp/charts", "../tmp/data", "../tmp/logs", "../tmp/ue-data"]
    if withDefault:
        source_folders += ["../tmp/charts-default", "../tmp/data_default", '../tmp/logs-default', "../tmp/ue-data_default"]
    destination_parent = os.path.join(current_path, "../results")

    # Create the destination directory
    destination_path = os.path.join(destination_parent, destination_directory)
    os.makedirs(destination_path, exist_ok=True)

    # Copy the folders to the destination
    for folder in source_folders:
        source_path = os.path.join(current_path, folder)
        shutil.copytree(source_path, os.path.join(destination_path, folder))


def collect_and_plot(metric_lists, labels_dict, start_time, end_time, step, source_folder_path, dest_folder_path, debug=False):
    data_collection(metric_lists, labels_dict, start_time, end_time, step, source_folder_path, debug)
    plot_data_from_csv(source_folder_path, dest_folder_path, debug)


def real_time_charts(metric_lists, labels_dict, start_time, end_time, step, source_folder_path, dest_folder_path, debug=False):
    # Schedule the function to run every minute
    schedule.every(20).seconds.do(collect_and_plot, metric_lists, labels_dict, start_time, end_time, step, source_folder_path, dest_folder_path, debug)

    # Keep the script running until end time
    while datetime.now(timezone.utc).timestamp() < end_time:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    start_now = datetime.now(timezone.utc)
    start_time = start_now.timestamp()
    end_time = (start_now + timedelta(hours=1)).timestamp()

    step = '1s'

    utc_end_time = datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')[:-3]
    print(utc_end_time)

    source_folder_path, dest_folder_path = '../tmp/data_default/', '../tmp/charts-default/' 
    # Plot charts every 5 seconds for real-time visualisation
    real_time_charts(metric_lists, labels_dict, start_time, end_time, step, source_folder_path, dest_folder_path, debug=True)
