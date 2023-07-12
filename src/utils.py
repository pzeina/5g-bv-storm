from datetime import datetime, timezone
from re import search
import os
from csv import writer

# ANSI escape sequences for text colors
COLOR_ERROR = '\033[91m'
COLOR_SUCCESS = '\033[92m'
COLOR_WARNING = '\033[93m'
COLOR_INFO = '\033[94m'
COLOR_RESET = '\033[0m'

def parse_log(log_file_path, start_at=None, format='UE'):
    logs = []
    if format == "UE":
        pattern = r'^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3})\]'
        expr = "%Y-%m-%d %H:%M:%S.%f"
    elif format == "AMF":
        pattern = r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)"
        expr = "%Y-%m-%dT%H:%M:%SZ"
    time_window = start_at == None
    with open(log_file_path, 'r') as file:
        for line in file:
            match = search(pattern, line)

            if match:
                #print(line)
                timestamp = datetime.strptime(match.group(1), expr)
                # Ensure the timestamp is in UTC (offset-aware)
                timestamp_utc = timestamp.replace(tzinfo=timezone.utc)

                if not time_window and timestamp_utc >= start_at:
                    time_window = True

                if time_window:
                    logs.append((timestamp_utc, line))
                    #print(line)
    return logs


def parse_log_with_prefix(log_key_prefix, start_at=None):
    logs = {}
    folder_path = 'logs' 

    #print(log_key_prefix)
    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        log_file_path = os.path.join(folder_path, filename)
        if os.path.isfile(log_file_path) and log_key_prefix in log_file_path:
            one_log_output = []
            with open(log_file_path, 'r') as file:
                for line in file:
                    match = search(r'^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3})\]', line)
                    if match:
                        #print(line)
                        timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
                        # Ensure the timestamp is in UTC (offset-aware)
                        timestamp_utc = timestamp.replace(tzinfo=timezone.utc)

                        one_log_output.append((timestamp_utc, line))
            logs[log_file_path] = one_log_output
    return logs


def read_subprocess(process):
    logs = []
    line_counter = 0
    line_limit = 200

    # Read the output of the subprocess line by line
    for line_raw in iter(process.stdout.readline, b''):
        line = line_raw.decode('utf-8')
        match = search(r'^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3})\]', line)
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


def find_message(logs, ue_id, MESSAGE):
    for timestamp, line in logs:
        if f'[{ue_id}|nas]' in line and MESSAGE in line:
            return timestamp
    return None


def save_data_points(data_points, filename):
    mode = "w"  # "w" for writing, "a" for appending
    x,y = data_points

    # Open the CSV file in the specified mode
    with open(filename, mode, newline="") as file:
        my_writer = writer(file)
        
        # Write each point as a row in the CSV file
        for i in range (0,len(x),1):
            my_writer.writerow([x[i],y[i]])
    print(f'   Data saved as csv ({filename})')


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