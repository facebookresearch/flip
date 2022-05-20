echo 'To kill jobs in: '${VM_NAME}' after 2s...'
sleep 2s

echo 'Killing jobs...'
gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
sudo pkill python > /dev/null
source /kmh_data/staging/run_kill.sh > /dev/null
sudo lsof -w /dev/accel0
" &> /dev/null

echo 'Killed jobs.'
