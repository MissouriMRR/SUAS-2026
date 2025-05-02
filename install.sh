script_dir=$(dirname "$0")

function install_nvidia() {
  echo install nvidia
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  sudo apt-get update && sudo apt-get install nvidia-container-toolkit

  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  nvidia-ctk cdi list

  which getenforce && getenforce | grep "Enforcing" && sudo setsebool -P container_use_devices true

  # copy the pre-made compose override file to include GPUs in the env container
  cp "$script_dir/templates/nvidia-compose.override.yml" "$script_dir/compose.override.yml"
}

function install_amd() {
  echo "AMD GPU installation not implemented (i have nvidia card)"
}


sudo apt update && sudo apt install -y podman pipx

pipx install podman-compose
pipx ensurepath

case $1 in
  nvidia)
    install_nvidia
  ;;
  amd)
    install_amd
  ;;
esac
