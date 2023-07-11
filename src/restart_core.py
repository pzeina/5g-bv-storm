import subprocess
import time
from datetime import datetime, timezone


def kill_pod_from_commands(commands): 
    COMMAND_GET_BENIGN = commands['getBenign']
    COMMAND_DELETE_BENIGN = commands['deleteBenign']
    COMMAND_GET_ATTACKER = commands['getAttacker']
    COMMAND_DELETE_ATTACKER = commands['deleteAttacker']

    # Get running ue pod
    running_benign_pod = exec_command(COMMAND_GET_BENIGN)
    running_attacker_pod = exec_command(COMMAND_GET_ATTACKER)

    if running_benign_pod:
        # Delete existing pod
        exec_command(COMMAND_DELETE_BENIGN)

        # Wait for the pod to terminate
        print(f'Benign pod(s) {running_benign_pod} terminating...')

        while(exec_command(COMMAND_GET_BENIGN)):
            time.sleep(2)
        
        print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S.%f")}]  Pod {running_benign_pod} terminated.')

    if running_attacker_pod:
        # Delete existing pod
        exec_command(COMMAND_DELETE_ATTACKER)

        # Wait for the pod to terminate
        print(f'Attacker pod(s) {running_attacker_pod} terminating...')

        while(exec_command(COMMAND_GET_ATTACKER)):
            time.sleep(2)
        
        print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S.%f")}]  Pod {running_attacker_pod} terminated.')


def exec_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode().strip()  # Decode and remove leading/trailing whitespace
    error = result.stderr.decode().strip()

    if error:
        print(f"Error occurred: {error}")
    return output


def termination_phase(_commands, pod_representants, order):
    kill_pod_from_commands(_commands)

    # Terminate the pods one by one
    for pod_name in order:

        # Terminate the pod
        exec_command(f'kubectl delete -k ./../5g-manifests/{pod_name} -n paul')

        # Wait until the pod is no longer in the terminating status
        pods = []
        for pod in pod_representants[pod_name]:
            pods.append(exec_command(f"kubectl get pods -n paul | awk '/{pod}" + "/ {print $1;exit}'"))

        for pod in pods:
            if pod: 
                while(exec_command("kubectl get pods -n paul | awk '/" + pod + "/ {print $1;exit}'")):
                    time.sleep(1)

        print(f"Pod '{pod_name}' terminated successfully.")  


def creation_phase(pod_representants, order):

    # Terminate the pods one by one
    for pod_name in order:

        # Terminate the pod
        exec_command(f'kubectl apply -k ./../5g-manifests/{pod_name} -n paul')

        time.sleep(0.5)
        # Wait until the pod is no longer in the terminating status
        pods = []
        for pod in pod_representants[pod_name]:
            pods.append(exec_command(f"kubectl get pods -n paul | awk '/{pod}" + "/ {print $1;exit}'"))
            print(exec_command(f"kubectl get pods -n paul | awk '/{pod}" + "/ {print $1;exit}'"))

        for pod in pods:
            while exec_command(f"kubectl get pod {pod} -n paul -o json | jq -r '.status.phase'") != 'Running':
                if exec_command(f"kubectl get pod {pod} -n paul -o json | jq -r '.status.phase'") in ['CrashLoopBaskOff', 'error', 'Completed']:
                    print('Failed', pod)
                    return
                time.sleep(1)

        time.sleep(2)
        print(f"Pod '{pod_name}' up and running.")  

if __name__ == "__main__":
    from flooding_simulation import _commands
    # Careful while modifying these!
    pod_representants = {
    'free5gc':['free5gc-nrf', 'free5gc-upf'],
    'ueransim-gnb':['ueransim-gnb1'],
    'ueransim-ue':['ueransim-ue1','ueransim-ue2']
    }

    deletion_order = ['ueransim-gnb', 'free5gc']#['ueransim-ue', 'ueransim-gnb', 'free5gc']
    creation_order = ['free5gc', 'ueransim-gnb']

    termination_phase(_commands, pod_representants, deletion_order)
    time.sleep(2)
    creation_phase(pod_representants, creation_order)