echo 'To kill jobs in: '${VM_NAME}' after 10s...'
sleep 10s

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
sudo pkill python
source ~/flax_dev/run_kill.sh
sudo lsof -w /dev/accel0
"
