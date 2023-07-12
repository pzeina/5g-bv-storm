# 5g-bv-storm

# About
This repository contains the resources needed to perform a registration signaling storm attack on an emulated 5G core network, along with a blockchain-based mitigation solution designed by [Bohan Zhang](https://github.com/zbh888/free5gc-compose.git).

The Kubernetes cluster architecture (`5g-manifests`) comes from the [Network Research Group UW](Nhttps://github.com/nrg-uw/5g-manifests.git) project, on which testbed these experiments have been conducted.
It consists of a 5G core network using the [Free5GC](https://github.com/free5gc/free5gc) project and RAN using the [UERANSIM](https://github.com/aligungr/UERANSIM) project.

The scripts for performing the attacks (`src`) are an original work.

# Installation
Installation steps from 1 to 4 are copied from the [Network Research Group UW](Nhttps://github.com/nrg-uw/5g-manifests.git) project.
1. You need to have a working kubernetes cluster. Instructions for setting up a multi-node Kubernetes cluster is available in the [official docs](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/). Note that we are using `kubeadm=1.23.6-00 kubectl=1.23.6-00 kubelet=1.23.6-00` as this is the last version that supports Docker as a container runtime out of the box. If you are using the latest Kubernetes with docker, see instructions [here](https://kubernetes.io/docs/setup/production-environment/container-runtimes/#docker).

2. You need to install [Flannel CNI](https://github.com/flannel-io/flannel) and [Multus CNI](https://github.com/k8snetworkplumbingwg/multus-cni). Flannel is used for cluster networking. Multus CNI enables attaching multiple network interfaces to pods in Kubernetes, which is required for 5G NFs with multiple interfaces (e.g., UPF has the N3 interface towards gNB and the N4 interface towards SMF).

3. You need to create a local persistent volume for MongoDB. Free5GC uses MongoDB for storage. This can be created using the `create-free5gc-pv.sh` script.
Note that you need to change the `path` and `values` in the `free5gc-pv.yaml` file according to your cluster.

4. The Multus CNI interfaces in the manifests, denoted by `k8s.v1.cni.cncf.io/networks` in the deployment files have static IPs assigned to them according to our lab setup (129.97.168.0/24 subnet). All these IPs need to be changed according your scenario. Use an IP range that you can access from all nodes of your Kubernetes cluster.

5. The ganache environment required for the blockchain-based mitigation deployment can be installed as follows:
```
kubectl apply -k ./5g-manifests/ganache -n <namespace>
```
**WARNING: Do not stop this container otherwise you will have to build a new image for the AMF.**

6. Once the ganache container started, note down its IP address. Clone the repository for the [blockchain-assisted solution](https://github.com/zbh888/free5gc-compose.git). Then go to `free5gc-compose/base/free5gc/NFs/amf/internal/sbi/consumer/ue_authentication.go` and replace the IP address in the `web3url` variable line 97 with the IP address of your ganache container, port 8545.

7. After this change, compile the code using the `compile.sh` bash script located at the root of the same [repository](https://github.com/zbh888/free5gc-compose.git). You will get a docker image for the blockchain-assisted AMF, free5gc-compose-free5gc-amf. Store it to your docker hub or as a git package. With this image built, you do not need this code anymore except if your ganache container changes IP address (when restarting it for example).


8. In the deployment files located at `5g-bv-storm\5g-manifests\free5gc\nf\amf{1,2,3}-storm\amf-deployment.yaml`, set the field spec>template>spec>containers['image'] to the docker image of the AMF you have just built.

9. Before running the experiments, the last step you have to do is to populate the MongoDB. For the experiment use cases to be relevant, please add the following imsi ranges to the database: 
(20893000000000 to 20893000000120), (20893000001000 to 20893000001220) and (20893000002000 to 20893000002220). The other fields must all be fixed to the values stored in `5g-bv-storm\5g-manifests/ueransim-ue-benign/ue1/ue-configmap.yaml`. Do **not** add imsi values between 20893000005000 and 20893000005220.

**IMPORTANT: Make sure to back up this database once populated.**

10.  If you have set up Grafana/Prometheus, you may change the urls in the files `5g-bv-storm\src\prometheus_data_collector` and `5g-bv-storm\src\prometheus_data_collector_input`.
If you have not, make sure to comment lines 993-996 in `5g-bv-storm\src\flooding_simulation.py`:
```python
    real_time_charts_process = multiprocessing.Process(target=exec_command, args=('python3 prometheus_data_collector.py',))
    real_time_charts_process.start()
```

11. You now can start a registration storm signalling attack by simply running the following:
```
python3 src\flooding_simulation.py
```

# Simulation parameters
At the end of the `src\flooding_simulation.py` script, you can select your own simulation parameters. We suggest you first try the 4 following scenarios, with the remaining fields unchanged.

## No storm
```python
nb_ues = 0,
amf_mode = 'default',
ghost = False
```

## Storm with no protection
```python
nb_ues = 30,
amf_mode = 'default',
ghost = False
```

## Storm with baseline protection at the UDM
```python
nb_ues = 30,
amf_mode = 'default',
ghost = True
```

## Storm with blockchain-assisted protection at the AMF
```python
nb_ues = 30,
amf_mode = 'signalling_storm_patch',
ghost = False
```
As the AMF is unable to deconceal the SUCI and since the blockchain we deployed is feeded fixed addresses unrelated to the received SUCIs, we need for this scenario to disable the SUCI encryption for attackers so the AMF is able to deny them (note again that this work does not address the *detection* of the malicious UEs but focuses on the *mitigation* of the storm). For this, go to the files `5g-bv-storm/5g-manifests/ueransim-ue-attacker/ue{1,2,3}/ue-configmap.yaml` and set the field `protectionScheme` to 0.
 
Do not forget to set it back to 1 for the other scenarios.



# Running into issues?
The manifest files are not working? Please see the [Network Research Group UW's FAQ](Fhttps://github.com/nrg-uw/5g-manifests/blob/main/FAQ.md).

You have been running experiments for a while and you noticed a performance decrease? Drop the MongoDB and populate it again. Your may use the backup you once made while creating it.


# Credits
The Kubernetes environment architecture comes from the [Network Research Group UW](Nhttps://github.com/nrg-uw/5g-manifests.git). It is itself heavily inspired from [towards5gs-helm](https://github.com/Orange-OpenSource/towards5gs-helm) and the Docker images used are based on [free5gc-compose](https://github.com/free5gc/free5gc-compose).

   