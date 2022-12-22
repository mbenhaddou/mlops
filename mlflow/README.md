# On-premises Configuration Example

You can quickly setup your MLflow on-premises environment with this example.

 [Batteries Included](https://www.python.org/dev/peps/pep-0206/#batteries-included-philosophy), settings files are basically all done.

## Quickstart

1. Install docker, then install MLflow and hydra.

    ```sh
    pip install mlflow hydra-core
    ```

2. Make a folder to store artifacts.

    Edit `.env` if you want to change folder, it's `/var/mlflow/artifacts'.

    ```sh
    sudo mkdir /var/mlflow/artifacts'
    ```
   change folder permissions
3. 
    ```sh
    sudo chmod 777 /var/mlflow'
    ```

3. Get your mlflow server up and running. This takes time.

    ```sh
    docker-compose up --build -d
    ```

4. Confirm your server is running properly.

    Open server URI. It's `http://your-server-ip-or-host-name:5000/`.

   You should see the mlflow ui


### Where are your artifacts on browser?

If you click on one of run, deteil will open. And you can also find artifacts at the bottom of the page.

![result image](images/on_pre_2.png)

## Trouble shooting

Stop containers first.

```sh
docker-compose down
```

See what's happening by running without `-d`.

```sh
docker-compose up
```

You might see some errors, check them and fix...

## Cleaning docker-created-files

Followings will clean up both containers/images.

```sh
docker ps -aq |xargs docker rm
docker images -aq |xargs docker rmi
```

Following will clean up cache.

```sh
docker system prune -a
```
