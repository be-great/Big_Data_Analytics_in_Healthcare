#!/usr/bin/env bash
set -e

# ---- A) Make sure WSL sees your NVIDIA GPU (Windows host must have the WSL CUDA driver) ----
# Ref: NVIDIA "CUDA on WSL" + Microsoft guide (see citations)
# On Windows: install the CUDA-enabled NVIDIA driver for WSL, then restart WSL.
# In Ubuntu/WSL:
sudo apt update
sudo apt install -y build-essential git curl wget unzip apt-transport-https ca-certificates lsb-release

# ---- B) Java + Spark (3.5.x works great; Hadoop client libraries included) ----
sudo apt install -y openjdk-17-jdk
JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc

SPARK_VERSION=3.5.3
wget -qO ~/spark.tgz https://downloads.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz
sudo tar -xzf ~/spark.tgz -C /opt && rm ~/spark.tgz
sudo mv /opt/spark-${SPARK_VERSION}-bin-hadoop3 /opt/spark
echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc
echo 'export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH' >> ~/.bashrc
