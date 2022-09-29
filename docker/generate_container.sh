# [Example] bash docker/generate_container.sh --image_name $IMAGE_NAME --container_name $CONTAINER_NAME --external_port 8888 

for ((argpos=1; argpos<$#; argpos++)); do
    if [ "${!argpos}" == "--container_name" ]; then
        argpos_plus1=$((argpos+1))
        container_name=${!argpos_plus1}
    fi
    if [ "${!argpos}" == "--image_name" ]; then
        argpos_plus1=$((argpos+1))
        image_name=${!argpos_plus1}
    fi
    if [ "${!argpos}" == "--external_port" ]; then
        argpos_plus1=$((argpos+1))
        external_port=${!argpos_plus1}
    fi
done

echo "Container Name: " $container_name
echo "Image Name: " $image_name
echo "External Port #: " $external_port

# --gpus '"device=0,1"'
# TODO: customize local directory
docker run --gpus '"device=all"' -td --ipc=host --name $container_name\
	-v ~/tak/repo:/repo\
	-v /etc/passwd:/etc/passwd\
    -v /etc/localtime:/etc/localtime:ro\
	-e TZ=Asia/Seoul\
	-p $external_port:$external_port $image_name