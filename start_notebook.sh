echo "On your local system please open a new terminal window and run: "
echo "ssh -N -L 8080:localhost:8080 -i ./your_key.key your_user@xxx.xxx.xxx.xxx"
echo ""
echo "After that open one of the links printed out below in your local browser"
echo ""
secs=$((10))
while [ $secs -gt 0 ]; do
   echo -ne "Launching jupyter notebook in: $secs\033[0K\r"
   sleep 1
   : $((secs--))
done

numactl --cpunodebind=0 --membind=0 jupyter notebook --no-browser --port=8080
