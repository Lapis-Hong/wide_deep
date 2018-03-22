#!/usr/bin/env bash
# This scripts run the TensorFlow serving
# Usage:
#       bash run_serving.sh [-p $port] [-n $model_name] [-d $model_dir]
set -e

# default setting
PORT=9000
MODEL_NAME='wide_deep'
MODEL_DIR=''

show_usage()
{
        echo -e "\nUsage:\n  $0 [options]\n\nOptions:"
        printf "  %-20s %-40s \n" "-h, --help" "Show help."
        printf "  %-20s %-40s \n" "-p, --port" "Port, default 9000."
        printf "  %-20s %-40s \n" "-n, --model_name" "Model name, default wide deep."
        printf "  %-20s %-40s \n" "-d, --model_dir" "SavedModel directory."
}

if [ $# == 0 ]; then
    echo "Using default args: port=${PORT}, model_name=${MODEL_NAME}, model_dir=${MODEL_DIR}"
fi

#-o或--options选项后面接可接受的短选项，如ab:c::，表示可接受的短选项为-a -b -c，
# 其中-a选项无冒号不接参数，-b选项后必须接参数，-c选项的参数为可选的, 必须紧贴选项
#-l或--long选项后面接可接受的长选项，用逗号分开，冒号的意义同短选项。
#-n选项后接选项解析错误时提示的脚本名字
ARGS=`getopt -o hp:n:d: -l help,port:,model_name:,model_dir: -n 'run_serving.sh' -- "$@"`
#将规范化后的命令行参数分配至位置参数（$1,$2,...)
eval set -- "${ARGS}"

while true
do
    case "$1" in
        -h|--help)
            show_usage;
            exit 1;;
        -p|--port)
            echo "Using port: $2"
            PORT=$2;
            shift 2;;
        -n|--model_name)
            echo "Using model_name: $2"
            MODEL_NAME=$2;
            shift 2;;
        -d|--model_dir)
            echo "Using model_dir: $2"
            MODEL_DIR=$2;
            shift 2;;
        --)
            shift
            break;;
        *)
            echo "Internal error!"
            exit 1;;
    esac
done

cd ../python/tensorflow_serving
python export_savedmodel.py
echo "Already export SavedModel."

# locally compiled ModelServer
# bazel build //tensorflow_serving/model_servers:tensorflow_model_server
# bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_base_path=$export_dir_base

# install using apt-get
# sudo apt-get update && sudo apt-get install tensorflow-model-server
tensorflow_model_server --port=${PORT} --model_name=${MODEL_NAME} --model_base_path=${MODEL_DIR}
