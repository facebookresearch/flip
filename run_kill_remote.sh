echo 'To kill jobs in: '${VM_NAME}' after 5s...'
sleep 5s

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
sudo pkill python > /dev/null
source ~/flax_dev/run_kill.sh > /dev/null
sudo lsof -w /dev/accel0
"
