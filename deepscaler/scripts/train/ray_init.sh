#!/bin/bash
set -x

N_NODES=4
N_GPUS=4
MASTER_ADDR=10.150.0.216

echo "Start ray cluster with ${N_NODES} nodes"

#export RAY_BACKEND_LOG_LEVEL=debug
#export RAY_LOG_TO_STDERR=1
export MKL_SERVICE_FORCE_INTEL=1

ray stop --force
rm -rf /tmp/ray


# (1) 根据节点角色启动 Ray
if [ ${RANK} -eq 0 ]; then
  echo "Start the head node"
  # 指定 Ray head 的 IP/端口，以及 dashboard 端口
  nohup ray start --head \
            --node-ip-address=${MASTER_ADDR} \
            --port=6379 \
            --num-gpus=${N_GPUS} \
            --dashboard-port=8266 \
            --dashboard-host=127.0.0.1 &

  echo "Waiting for worker nodes to connect..."
  EXPECTED_NODE_COUNT=${N_NODES}  # Adjust this to your expected number of nodes
  while true; do
    CONNECTED_NODE_COUNT=$(ray list nodes | grep -c "ALIVE")
    if [ "$CONNECTED_NODE_COUNT" -eq "$EXPECTED_NODE_COUNT" ]; then
      echo "All nodes connected to the cluster."
      break
    fi
    echo "Connected nodes: $CONNECTED_NODE_COUNT/$EXPECTED_NODE_COUNT. Retrying in 30 seconds..."
    sleep 30
  done

else
  echo "Start the worker node"
  # worker 连到 head 节点
  MAX_RETRIES=512
  RETRY_COUNT=0
  SLEEP_INTERVAL=30

  while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    ray start --address ${MASTER_ADDR}:6379 --num-gpus=$N_GPUS --block
    if [ $? -eq 0 ]; then
      echo "Worker node successfully connected to head node."
      break
    else
      RETRY_COUNT=$((RETRY_COUNT + 1))
      echo "Failed to connect to head node. Retrying in ${SLEEP_INTERVAL} seconds... (Attempt $RETRY_COUNT of $MAX_RETRIES)"
      sleep $SLEEP_INTERVAL
    fi
  done

  if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Failed to connect to head node after $MAX_RETRIES attempts. Exiting."
    exit 1
  fi
fi

# Wait for dashboard to be ready
sleep 10

# Debug checks
#echo "Checking dashboard connectivity..."
#curl -v "http://${MASTER_ADDR}:8265/api/jobs/" || echo "Dashboard not accessible"

# Show listening ports
#echo "Checking listening ports..."
#netstat -tlpn | grep 8265
#netstat -tlpn | grep 6379
#netstat -tlpn | grep 8266
ray status

echo "Start the training job"