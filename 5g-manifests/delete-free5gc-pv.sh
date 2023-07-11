#!/bin/bash

kubectl delete pvc datadir-mongodb-0 -n paul
kubectl delete pv free5gc-local-pv-paul