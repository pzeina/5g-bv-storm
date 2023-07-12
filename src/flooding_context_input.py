import subprocess
import multiprocessing
from datetime import datetime, timedelta, timezone
import re
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time
import csv
from threading import Lock
import prometheus_data_collector_input
from random_generator import *

coreSelector = 'free5gc'
gnbSelector = 'ueransim-gnb'
ueSelector = 'ueransim-ue-benign'
ueManifest = f'../5g-manifests/{ueSelector}/' # '../5gc-manifests/ueransim-v3.2.6-ue-slice-x3/ue-attacker/'

# Markers needed to read the logs
REGISTRATION_REQUEST = 'Sending Initial Registration'
REGISTRATION_ACCEPT = 'Registration accept received'
PDU_SESSION_EST_REQUEST = 'Sending PDU Session Establishment Request'
PDU_SESSION_EST_ACCEPT = 'PDU Session Establishment Accept received'
DEREGISTRATION_REQUEST = 'Starting de-registration procedure'
DEREGISTRATION_ACCEPT = 'De-registration accept received'


command_delete = f'kubectl delete -k ./{ueManifest} -n paul'
command_create = f'kubectl apply -k ./{ueManifest} -n paul'
command_delete_atk = f'kubectl delete -k ./../5g-manifests/ueransim-ue-attacker/ -n paul'
command_get_ue_atk = "kubectl get pods -n paul | awk '/ueransim-ue1-attacker/ {print $1;exit}'"
command_get_ue = "kubectl get pods -n paul | awk '/ueransim-ue1-benign/ {print $1;exit}'"

def exec_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode().strip()  # Decode and remove leading/trailing whitespace

    return output


def init_pod(deployments):  
    # Try to get existing attacker ue pod
    existing_pod_atk = exec_command(command_get_ue_atk)       
    if existing_pod_atk:
        # There is a running ue pod
        print("Existing pod to be terminated:", existing_pod_atk)
        # Delete existing pod
        exec_command(command_delete_atk)
        # Wait for the pod to terminate
        print('Waiting for the existing pod to terminate...')
        while(exec_command(command_get_ue_atk)):
            time.sleep(2)

    # Try to get existing benign ue pod
    existing_pod = exec_command(command_get_ue)      
    if existing_pod:
        # There is a running ue pod
        print("Existing pod to be terminated:", existing_pod)
        # Delete existing pod
        exec_command(command_delete)
        # Wait for the pod to terminate
        print('Waiting for the existing pod to terminate...')
        while(exec_command(command_get_ue)):
            time.sleep(2)

    # Create pod
    print("Creating pod...")
    if deployments > 1:
        exec_command(command_create)
    else:
        exec_command(f'kubectl apply -k ./{ueManifest} -n paul')

    pod = exec_command(f"kubectl get pods -n paul | awk '/ueransim-ue1-benign" + "/ {print $1;exit}'")
    while exec_command(f"kubectl get pod {pod} -n paul -o json | jq -r '.status.phase'") != 'Running':
        time.sleep(1)
    time.sleep(5)


def kill_pod():
    # Get nunning ue pod
    running_pod = exec_command(command_get_ue)

    if running_pod:
        # There is a running ue pod

        # Delete existing pod
        exec_command(command_delete)

        # Wait for the pod to terminate
        print(f'Pod {running_pod} terminating...')

        while(exec_command(command_get_ue)):
            time.sleep(2)
        
        print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S.%f")}]  [NORMAL-CASE]   Pod {running_pod} terminated.')


def benign_flow(ue_pod_md, csv_times, offset):

    my_pod = exec_command("kubectl get pods -n paul | awk '/ueransim-ue" + str(ue_pod_md) + "-benign/ {print $1;exit}'")

    with open(csv_times, 'r') as file:
        reader = csv.reader(file)
        header_row = next(reader)  # Read the header row

        # Find the element in the header row that starts with 'Time:'
        time_element = next((element for element in header_row if element.startswith('Time:')), None)

        # Extract the time value using regular expressions
        time_start_generation = float(re.search(r'Time:(.*)', time_element).group(1).strip())

        imsi = 0
        for row in reader:
            imsi += 1
            time_value = float(row[0])
            log_file_path = f'../tmp/logs-default/ueransim-ue{ue_pod_md}-benign_imsi-208930000001{str(imsi).zfill(3)}.txt'

            # Get current time
            current_time = datetime.now(timezone.utc).timestamp()

            # Wait until the specified time is reached
            while current_time < time_value + offset:
                #print('514')
                current_time = datetime.now(timezone.utc).timestamp()
                #print(f'Current time:  {current_time}     |    Time for next benign UE:  {time_value}')
                

            # Execute the command
            process = multiprocessing.Process(target=exec_command, args=(f'kubectl exec {my_pod} -n paul -- ./nr-ue -c config/free5gc-ue.yaml -i imsi-208930000001{str(imsi).zfill(3)} > {log_file_path}', ))
            process.start()



def analyze_benign_flow(benign_params, log_folder, log_key_prefix, output_queue=None):

    absolute_time_start = benign_params['start']

    registration_duration, pdu_establishment_duration, deregistration_duration = [[],[]], [[],[]], [[],[]]
    registration_start_times = {}
    pdu_session_start_times = {}
    deregistration_start_times = {}
    deregistrations_count = 0
    deregistrations_completed = False

    logs = parse_log_with_prefix(log_folder, log_key_prefix)
    for ue_id, ue_log_file in logs.items():
        #print(ue_id)
        for timestamp, line in ue_log_file:
            match = re.search(r'\[nas\]', line)
            if match:
                if REGISTRATION_REQUEST in line:
                    registration_start_times[ue_id] = timestamp
                elif REGISTRATION_ACCEPT in line:
                    #print(REGISTRATION_ACCEPT)
                    if ue_id in registration_start_times:
                        registration_start_time = registration_start_times[ue_id]
                        time_for_procedure = (timestamp - registration_start_time).total_seconds()
                        relative_registration_start_time = (registration_start_time - absolute_time_start).total_seconds()
                        registration_duration[0].append(relative_registration_start_time)
                        registration_duration[1].append(time_for_procedure)
                        del registration_start_times[ue_id]

                elif PDU_SESSION_EST_REQUEST in line:
                    pdu_session_start_times[ue_id] = timestamp
                elif PDU_SESSION_EST_ACCEPT in line:
                    #print(PDU_SESSION_EST_ACCEPT)
                    if ue_id in pdu_session_start_times:
                        pdu_session_start_time = pdu_session_start_times[ue_id]
                        time_for_procedure = (timestamp - pdu_session_start_time).total_seconds()
                        relative_pdu_session_start_time = (pdu_session_start_time - absolute_time_start).total_seconds()
                        pdu_establishment_duration[0].append(relative_pdu_session_start_time)
                        pdu_establishment_duration[1].append(time_for_procedure)
                        del pdu_session_start_times[ue_id]

                elif DEREGISTRATION_REQUEST in line:
                    deregistration_start_times[ue_id] = timestamp
                elif DEREGISTRATION_ACCEPT in line:
                    #print(DEREGISTRATION_ACCEPT)
                    if ue_id in deregistration_start_times:
                        deregistration_start_time = deregistration_start_times[ue_id]
                        time_for_procedure = (timestamp - deregistration_start_time).total_seconds()
                        relative_deregistration_start_time = (deregistration_start_time - absolute_time_start).total_seconds()
                        deregistration_duration[0].append(relative_deregistration_start_time)
                        deregistration_duration[1].append(time_for_procedure)
                        del deregistration_start_times[ue_id]
                        deregistrations_count += 1
        

        registrations_completed = not bool(registration_start_times) # Is True if the dictionary is empty, i.e. there is no registration ongoing
        pdu_session_est_completed = not bool(pdu_session_start_times) # Is True if the dictionary is empty, i.e. there is no PDU session establishment ongoing
        deregistrations_completed = not bool(deregistration_start_times)

        # In case procedures went wrong
        if not registrations_completed:
            #print(f'[ERROR]  [NORMAL-CASE]  {ue_id} has not been registered')
            #print(f'[ERROR]  [NORMAL-CASE]  {ue_id} has not established a PDU session')
            i = 1
        if not deregistrations_completed:
            if len(deregistration_start_times) == 0:
                print(f'[WARNING] Deregistration procedure has encountered errors but has now completed.')
            else:
                print(f'[ERROR]  {len(deregistration_start_times)} UEs have not been deregistered')

    if output_queue != None:
        output_queue.put((registration_duration, pdu_establishment_duration, deregistration_duration))
    
    return registration_duration, pdu_establishment_duration, deregistration_duration





def run_benign_users(benign_params, log_folder, ue_data_folder, output_charts_folder, csv_times, time_offset=0, printRegistrations = False):
    end_time = (benign_params['start'] + timedelta(minutes=benign_params['duration'])).timestamp()


    # Save attack parameters
    with open('../tmp/logs-default/benign_params_default.txt', 'w', newline="") as file:
        writer = csv.writer(file)
        # Write each parameter as a row in the CSV file
        for key, value in benign_params.items():
            writer.writerow([key, value])

    results = []
    for pod in range(benign_params['deployments']):
        dictionary = {'registration_duration':[[],[]], 'pdu_establishment_duration':[[],[]], 'deregistration_duration':[[],[]]}
        results.append(dictionary)


    # Create new nodes and store logs in a separate file
    prc = []
    processes = []

    for ue_pod in range(benign_params['deployments']):
        log_file_path = f'{log_folder}ueransim-ue{ue_pod+1}-benign.txt'
        my_pod = f"$(kubectl get pods -n paul | awk '/ueransim-ue{ue_pod+1}-benign" + "/ {print $1;exit}')"
        with Lock():
            prc.append(subprocess.Popen(f'kubectl exec {my_pod} -n paul -- ./nr-ue -c config/free5gc-ue.yaml > {log_file_path}', shell=True, stdout=subprocess.PIPE))
    
        process = multiprocessing.Process(target=benign_flow, args=(ue_pod+1, csv_times[pod], benign_params['start'].timestamp() + benign_params['wave_freq']))

        processes.append(process)
    
    time.sleep(1)

    for process in processes:
        process.start()

    # Keep the script running until end time
    while datetime.now(timezone.utc).timestamp() < end_time:
        time.sleep(15)

        # Get results
        for ue_pod in range (benign_params['deployments']):
            registration_duration, pdu_establishment_duration, deregistration_duration = analyze_benign_flow(benign_params, log_folder, f'ueransim-ue{ue_pod+1}-benign')
            if printRegistrations:
                print(registration_duration)

            # Save the data to csv files
            save_data_points(registration_duration, f'{ue_data_folder}ue{ue_pod+1}-benign_registration_time.csv')
            save_data_points(pdu_establishment_duration, f'{ue_data_folder}ue{ue_pod+1}-benign_pdu_establishment_time.csv')

            # Plot the chart
            output_charts_path = f'{output_charts_folder}ue{ue_pod+1}-benign_processing_times.png'
            plot_time_charts(registration_duration, pdu_establishment_duration, deregistration_duration, output_charts_path)

    for process in processes:
        process.terminate()
        process.join()

    time.sleep(2)

def delete_files_in_folder(folder_path, key=''):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    if key:
        files = [filename for filename in files if key in filename]

    # Iterate over the files and delete each one
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            #print(f"Deleted file: {file_path}")


def read_subprocess(process):
    logs = []
    line_counter = 0
    line_limit = 200

    # Read the output of the subprocess line by line
    for line_raw in iter(process.stdout.readline, b''):
        line = line_raw.decode('utf-8')
        match = re.search(r'^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3})\]', line)
        if match:
            #print(line)
            timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")

            # Ensure the timestamp is in UTC (offset-aware)
            timestamp_utc = timestamp.replace(tzinfo=timezone.utc)

            logs.append((timestamp_utc, line))
            line_counter += 1
        if line_counter > line_limit:
            break
    return logs



def save_data_points(data_points, filename):
    mode = "w"  # "w" for writing, "a" for appending
    x,y = data_points

    # Open the CSV file in the specified mode
    with open(filename, mode, newline="") as file:
        writer = csv.writer(file)
        
        # Write each point as a row in the CSV file
        for i in range (0,len(x),1):
            writer.writerow([x[i],y[i]])
    print(f'   Data saved as csv ({filename})')


def plot_time_charts(registration_duration, pdu_establishment_duration, deregistration_duration, output_chart_path):
    plt.plot(registration_duration[0],registration_duration[1], 'x', label='Initial Registration', color='tab:green')
    plt.plot(pdu_establishment_duration[0],pdu_establishment_duration[1], '+', label='PDU Session Establishment', color='tab:orange')
    plt.plot(deregistration_duration[0],deregistration_duration[1], 'x', label='Deregistration', color='tab:red')

    plt.xlabel('Time elapsed (s)',)
    plt.ylabel('Processing time (s)')
    plt.title('UE requests processing time')
    plt.legend()
    plt.grid(True)

    if output_chart_path:
        plt.savefig(output_chart_path)
        plt.close()
    else:
        plt.show()
    print(f'   Chart generated ({output_chart_path})')


def parse_log_with_prefix(folder_path, log_key_prefix, start_at=None):
    logs = {}

    print(log_key_prefix)
    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        log_file_path = os.path.join(folder_path, filename)
        if os.path.isfile(log_file_path) and log_key_prefix in log_file_path:
            one_log_output = []
            with open(log_file_path, 'r') as file:
                for line in file:
                    match = re.search(r'^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3})\]', line)
                    if match:
                        #print(line)
                        timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
                        # Ensure the timestamp is in UTC (offset-aware)
                        timestamp_utc = timestamp.replace(tzinfo=timezone.utc)

                        one_log_output.append((timestamp_utc, line))
            logs[log_file_path] = one_log_output
    return logs


def find_message(logs, ue_id, MESSAGE):
    for timestamp, line in logs:
        if f'[{ue_id}|nas]' in line and MESSAGE in line:
            return timestamp
    return None

def run_default_experiment(benign_params_default, time_start_atk):
    # Time parameters
    benign_params_default['start'] = datetime.now(timezone.utc)
    start_now = benign_params_default['start']
    start_time = start_now.timestamp()
    end_time = (start_now + timedelta(minutes=benign_params_default['duration'])).timestamp()
    end_collection_time = (start_now + timedelta(minutes=benign_params_default['duration']+benign_params_default['collection_extra_time'])).timestamp()
    #print(end_time)
    step = benign_params_default['collection_step']

    delete_files_in_folder('../tmp/charts-default/')
    delete_files_in_folder('../tmp/data_default/')
    delete_files_in_folder('../tmp/logs-default/')
    delete_files_in_folder('../tmp/ue-data_default/')

    # Initialization
    init_pod(benign_params_default['deployments'])

    real_time_charts_process = multiprocessing.Process(target=exec_command, args=('python3 prometheus_data_collector_input.py',))
    real_time_charts_process.start()
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S.%f")}]  Real-time charts running in background.')


    csv_files = [f'random_times_{i+1}.csv' for i in range(benign_params_default['deployments'])]

    # Collect data from the testbed (Prometheus) in the background
    log_collection_process = multiprocessing.Process(target=prometheus_data_collector_input.generate_logs, args=("logs-default/", benign_params_default['deployments'],))
    log_collection_process.start()
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S.%f")}]  Log collection running in background.')
    


    # Perform simulation and save results
    time_offset = start_time - time_start_atk
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S.%f")}]  Running the benign users...')
    run_benign_users(benign_params_default, '../tmp/logs-default/', '../tmp/ue-data_default/', '../tmp/charts-default/', csv_files, time_offset=time_offset)
    
    
    end2 = (start_now + timedelta(minutes=benign_params_default['duration']+benign_params_default['collection_extra_time']))
    time.sleep(max(0,(end2 - datetime.now(timezone.utc)).total_seconds()))

    kill_pod()


    log_collection_process.terminate()
    real_time_charts_process.terminate()
    real_time_charts_process.join()


    # Commit the results to the remote Git repository
    #git_commit_results('Default-'+str(int(start_time)))   



if __name__ == "__main__":


    benign_params = {
        'start': datetime.now(timezone.utc),
        'duration': 5,                  # in minutes
        'collection_extra_time': 2,     # in minutes
        'collection_step': 1,           # in seconds
        'wave_freq': 60,                # in seconds
        'cooldown': 1,                  # in seconds
        'nb_ues': 50,
        'nb_benign':80,
        'deployments': 1,       
        'restart_pod': True,           
        'deregistration': 'normal',
        'integrity_protection': '',
        'ciphering_algorithm': '',
        'gnb1_cpu_requests':'',
        'gnb1_cpu_limits':'',
        'ue1_cpu_requests':'',
        'ue1_cpu_limits':'',
        'amf1_cpu_requests':'',
        'amf1_cpu_limits':'',
        'gnb2_cpu_requests':'',
        'gnb2_cpu_limits':'',
        'ue2_cpu_requests':'',
        'ue2_cpu_limits':'',
        'amf2_cpu_requests':'',
        'amf2_cpu_limits':'',
        'ausf_cpu_requests': '',
        'ausf_cpu_limits': '',
        'udr_cpu_requests': '',
        'udr_cpu_limits': '',
        'udm_cpu_requests': '',
        'udm_cpu_limits': '',
        #'udm_rate':1,                   # in mbit
        #'udm_burst':10,                 # in kb
        #'udm_latency':70,               # in ms
    }


    # Initialization
    init_pod(benign_params['deployments'])

    ## TO DELETE !!!!
    delete_files_in_folder('logs-default/')
    delete_files_in_folder('ue-data_default/')

    # Time parameters
    start_now = benign_params['start']
    start_time = start_now.timestamp()
    end_time = (start_now + timedelta(minutes=benign_params['duration'])).timestamp()
    end_collection_time = (start_now + timedelta(minutes=benign_params['duration']+benign_params['collection_extra_time'])).timestamp()
    #print(end_time)
    step = benign_params['collection_step']


    csv_files = []
    # Generate random times for the benign users
    for i in range(benign_params['deployments']):
        csv_files.append(f'random_times_{i+1}.csv')


    # Collect data from the testbed (Prometheus) in the background
    log_collection_process = multiprocessing.Process(target=prometheus_data_collector_input.generate_logs, args=("logs/", benign_params['deployments'],))
    log_collection_process.start()
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S.%f")}]  Log collection running in background.')
    

    # Plot charts every 5 seconds for real-time visualisation
    delete_files_in_folder('../tmp/data_default/')
    real_time_charts_process = multiprocessing.Process(target=prometheus_data_collector_input.real_time_charts, args=(prometheus_data_collector_input.metric_lists, prometheus_data_collector_input.labels_dict, start_time, end_collection_time, step))
    real_time_charts_process.start()
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S.%f")}]  Real-time charts running in background.')


    # Perform attack and save results
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S.%f")}]  Running the benign users...')
    run_benign_users(benign_params, '../tmp/logs-default/', '../tmp/ue-data_default/', '../tmp/charts-default/', csv_files)
    kill_pod()


    # Waiting for the background process to complete
    log_collection_process.terminate()
    real_time_charts_process.terminate()
    real_time_charts_process.join()


    # Commit the results to the remote Git repository
    prometheus_data_collector_input.git_commit_results('Benign-'+str(int(start_time)))   
