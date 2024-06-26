# 5g-bv-storm

If you wish to utilize this work in your research, we kindly request that you include the following citation:

```
@INPROCEEDINGS{10327880,
  author={Zhang, Bohan and Zeinaty, Paul and Limam, Noura and Boutaba, Raouf},
  booktitle={2023 19th International Conference on Network and Service Management (CNSM)}, 
  title={Mitigating Signaling Storms in 5G with Blockchain-assisted 5GAKA}, 
  year={2023},
  volume={},
  number={},
  pages={1-9},
  keywords={Resistance;Protocols;Costs;5G mobile communication;Storms;Emulation;Authentication;5G;security;5GAKA;signaling storm;blockchain},
  doi={10.23919/CNSM59352.2023.10327880}
}
```



# About
This repository contains the resources needed to perform a registration signaling storm attack on an emulated 5G core network, along with a blockchain-based mitigation solution designed by [Bohan Zhang](https://github.com/zbh888/free5gc-compose.git).

The Kubernetes cluster architecture (`5g-manifests`) comes from the [Network Research Group UW](Nhttps://github.com/nrg-uw/5g-manifests.git) project, on which testbed these experiments have been conducted.
It consists of a 5G core network using the [Free5GC](https://github.com/free5gc/free5gc) project and RAN using the [UERANSIM](https://github.com/aligungr/UERANSIM) project.

The scripts for performing the attacks (`src`) are an original work.





# Installation
Installation steps from 1 to 4 are copied from the [Network Research Group UW](https://github.com/nrg-uw/5g-manifests.git) project, and step 5 is copied from [5g-monarch](https://github.com/niloysh/5g-monarch.git).
1. You need to have a working kubernetes cluster. Instructions for setting up a multi-node Kubernetes cluster is available in the [official docs](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/). Note that we are using `kubeadm=1.23.6-00 kubectl=1.23.6-00 kubelet=1.23.6-00` as this is the last version that supports Docker as a container runtime out of the box. If you are using the latest Kubernetes with docker, see instructions [here](https://kubernetes.io/docs/setup/production-environment/container-runtimes/#docker).

2. You need to install [Flannel CNI](https://github.com/flannel-io/flannel) and [Multus CNI](https://github.com/k8snetworkplumbingwg/multus-cni). Flannel is used for cluster networking. Multus CNI enables attaching multiple network interfaces to pods in Kubernetes, which is required for 5G NFs with multiple interfaces (e.g., UPF has the N3 interface towards gNB and the N4 interface towards SMF).

3. You need to create a local persistent volume for MongoDB. Free5GC uses MongoDB for storage. This can be created using the `create-free5gc-pv.sh` script.
Note that you need to change the `path` and `values` in the `free5gc-pv.yaml` file according to your cluster.

4. The Multus CNI interfaces in the manifests, denoted by `k8s.v1.cni.cncf.io/networks` in the deployment files have static IPs assigned to them according to our lab setup (129.97.168.0/24 subnet). All these IPs need to be changed according your scenario. Use an IP range that you can access from all nodes of your Kubernetes cluster.

5. Clone [5g-monarch](https://github.com/niloysh/5g-monarch.git) and deploy SSMC and Visualization module as follows. We use Prometheus as the SSMC and Grafana as the visualization module. We use [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack) to deploy Prometheus and Grafana using the custom values file `monacrh-components/ssmc/kube-prometheus-values-nuc.yaml`.

```
helm install prometheus prometheus-community/kube-prometheus-stack -f kube-prometheus-values-nuc.yaml
```

**Note**: The use of the namespace `monitoring` is very important! This will affect autodiscovery of monitoring exporters.


6. The ganache environment required for the blockchain-based mitigation deployment can be installed as follows:
```
kubectl apply -k ./5g-manifests/ganache -n <namespace>
```
**WARNING: Do not stop this container otherwise you will have to build a new image for the AMF.**

7. Once the ganache container started, note down its IP address. Clone the repository for the [blockchain-assisted solution](https://github.com/zbh888/free5gc-compose.git). Then go to `free5gc-compose/base/free5gc/NFs/amf/internal/sbi/consumer/ue_authentication.go` and replace the IP address in the `web3url` variable line 97 with the IP address of your ganache container, port 8545.

8. After this change, compile the code using the `compile.sh` bash script located at the root of the same [repository](https://github.com/zbh888/free5gc-compose.git). You will get a docker image for the blockchain-assisted AMF, free5gc-compose-free5gc-amf. Store it to your docker hub or as a git package. With this image built, you do not need this code anymore except if your ganache container changes IP address (when restarting it for example).


9. In the deployment files located at `5g-bv-storm/5g-manifests/free5gc/nf/amf{1,2,3}-storm/amf-deployment.yaml`, set the field spec>template>spec>containers['image'] to the docker image of the AMF you have just built.

10. Before running the experiments, the last step you have to do is to populate the MongoDB using the Free5GC webconsole. The webconsole can be accessed at port 30600 on any of the Kubernetes nodes. For the experiment use cases to be relevant, please add the following imsi ranges to the database: 
(20893000000000 to 20893000000120), (20893000001000 to 20893000001220) and (20893000002000 to 20893000002220). The other fields must all be fixed to the values stored in `free5gc-ue.yaml` in `5g-bv-storm/5g-manifests/ueransim-ue-benign/ue1/ue-configmap.yaml`. Do **not** add imsi values between 20893000005000 and 20893000005220.

**IMPORTANT: Make sure to back up this database once populated.**

11.  If you have set up Grafana/Prometheus, you may change the variable ```PROMETHEUS_URL``` in the top of `5g-bv-storm/src/flooding_simulation`.
If you have not, make sure to set the boolean `withPrometheus` line 28 of this same script to `False`.

12. You now can start a registration storm by simply running the following:
```
python3 src/flooding_simulation.py
```




# Description of the script

## Outline
The script is organized as follows:
- Clear the `5g-bc-storm/tmp` directory
- Deploy a Free5GC core with one or multiple AMFs.
- Deploy one set of UERANSIM (gNB, benign UE, malicious UE) containers per AMF.
- Start a background process of log collection from the main 5G Network Functions and the UERANSIM containers.
- If Grafana/Prometheus is set up, start a background process of data collection from Prometheus.
- Start a background process registering benign UEs at random times.
- Perform the storm consisting of successive waves of registrations performed by the attackers at regular intervals.

Please refer to the section **Simulation parameters** for further details about the script.

## Output
- Data and logs are saved in `5g-bc-storm/tmp` during the experiment. The content of this directory is copied to the `5g-bc-storm/results` directory at the end of the each experiment.
- Graphs representing the processing times of the registrations will be saved after each wave and stored in `5g-bc-storm/tmp/charts` as the experiment runs. These charts may sometimes be uncomplete and not show the last wave (although it is saved correctly).
- If Grafana/Prometheus is set up, graphs representing the collected data from Prometheus are generated at regular intervals during the experiment and stored in `5g-bc-storm/tmp/charts`.




# Simulation parameters
At the end of the `src/flooding_simulation.py` script, you can select your own simulation parameters. We suggest you first try the 4 following scenarios, with the remaining fields unchanged.

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
 
Do not forget to set it back to 1 when not using this `amf_mode`.





# Results
Results from our experiments are given in `5g-bv-storm/results`. 

In `5g-bv-storm/results/analysis` you can find graphs generated from these results using the script `5g-bv-storm/src/post_analysis.py`.




# Running into issues?
The manifest files are not working? Please see the [Network Research Group UW's FAQ](Fhttps://github.com/nrg-uw/5g-manifests/blob/main/FAQ.md).

You have been running experiments for a while and you noticed a performance decrease? Drop the MongoDB and populate it again. Your may use the backup you once made while creating it.


# Credits
The Kubernetes environment architecture comes from the [Network Research Group UW](https://github.com/nrg-uw/5g-manifests.git). It is itself heavily inspired from [towards5gs-helm](https://github.com/Orange-OpenSource/towards5gs-helm) and the Docker images used are based on [free5gc-compose](https://github.com/free5gc/free5gc-compose). 

We also make use of some Docker images from [Niloy Saha](https://github.com/niloysh?tab=packages)'s repository, without whom these experiments could not have been done.

The mitigation solution is based on [Bohan Zhang](https://github.com/zbh888/free5gc-compose.git)'s work, from whome the blockchain-based AMF is derived.
   
