# kaggle-template-uv

## 環境構築

- インスタンス作成

  ```bash
  export PROJECT_ID=hogehoge
  export PROJECT_NAME=atma-18
  gcloud compute instances create $PROJECT_NAME \
      --project=$PROJECT_ID \
      --zone=us-central1-a \
      --machine-type=n1-highmem-8 \
      --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=kaggle \
      --no-restart-on-failure \
      --maintenance-policy=TERMINATE \
      --provisioning-model=SPOT \
      --instance-termination-action=STOP \
      --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
      --accelerator=count=1,type=nvidia-tesla-v100 \
      --create-disk=auto-delete=yes,boot=yes,device-name=atma-18,image=projects/ml-images/global/images/c0-deeplearning-common-cu122-v20241118-debian-11,mode=rw,size=100,type=pd-balanced \
      --no-shielded-secure-boot \
      --shielded-vtpm \
      --shielded-integrity-monitoring \
      --labels=goog-ec-src=vm_add-gcloud \
      --reservation-affinity=any
  ```

- インスタンスログイン

  ```bash
  gcloud compute ssh --zone "us-central1-a" $PROJECT_NAME  --project $PROJECT_ID --tunnel-through-iap -- -A
  ```

- uv install

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env
  ```

- 依存関係 install

  ```bash
  uv sync
  ```

- docker compose install
  ```bash
  sudo curl -L https://github.com/docker/compose/releases/download/v2.32.1/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
  ```

## データダウンロード

- guruguru のサイトからデータをダウンロード
  - データセットの部分を右クリックして、wget で持って来れる
- zip を解凍して input 以下に置く。
  - 以下のようになっていれば OK
  ```
    input
    |-- images
    |-- traffic_lights
    |-- test_features.csv
    `-- train_features.csv
  ```
