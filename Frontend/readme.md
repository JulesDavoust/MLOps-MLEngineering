# Lab 1 - Gitlab pipeline

![Lab 1 instructions](../images/lab1.png)

## Objectives

- [x] Setup your team tools
- [x] Install Gitlab (with docker)
- [ ] Find a web application
- [ ] Configure a pipeline
- [ ] Improve your network configuration
  - [x] Add synchronized directories between the host and the VM
  - [x] Configure port forwarding (ssh, http)
  - [x] Connect to the VM with ssh client
  - [ ] Trigger a pipeline with a curl from host and a second VM
- [ ] Improve yout components in order to move forward
  - [ ] Build a second docker and register as a runner
  - [ ] Build a VM to deploy the app
  - [ ] Customize your pipeline to build/test on the runner and deploy on the final VM

## Installation

### Configuration

Start VM

```bash
cd lab1
vagrant up
vagrant ssh
```

Start gitlab and gitlab-runner with docker compose

```bash
cd /vagrant_data
sudo docker compose up
```

Wait for the gitlab to boot on <http://localhost:8080/>

Login into gitlab with ```root``` and the password given by the following command

```bash
sudo docker exec -it gitlab grep 'Password:' /etc/gitlab/initial_root_password
```

In the gitlab UI, change the password of the root user and create a new repository for example "test-app".

=======================================================================

Return to your vagrant VM and create a docker network "gitlab_default" :
```shell
docker network create gitlab_default
```
Then in the /etc/hosts file add the line: "10.0.2.15 gitlab.example.com".

Then go to the python project folder and execute the following commands :
```shell
git remote set-url origin http://gitlab.example.com/root/devops_project.git
git push --all origin
````

=======================================================================

Afterwards, create the following three runners:
```shell
docker exec -it gitlab-runner gitlab-runner register \
  --non-interactive \
  --url http://gitlab.example.com \
  --registration-token {token} \
  --tag-list "pythonapp" \
  --executor docker \
  --docker-image python:3.9-slim-buster \
  --docker-network-mode gitlab_default
```
```shell
docker exec -it gitlab-runner gitlab-runner register \
  --url http://gitlab.example.com \   
  --registration-token {token} \   
  --tag-list "dockerapp" \   
  --executor docker \   
  --docker-image docker:26.1 \   
  --docker-network-mode gitlab_default \ 
  --docker-privileged true \ 
  --docker-volumes "/certs/client" \
```
```shell
docker exec -it gitlab-runner gitlab-runner register \
  --url http://gitlab.example.com \   
  --registration-token {token} \   
  --tag-list "shellapp" \   
  --executor shell \
  --docker-network-mode gitlab_default \
```
Replace {token} with your own runner token. To find it, go to 'Admin area', then click on 'Instance runner', click on the three dots, and copy your token.

=======================================================================

You will need to create an EC2 VM in AWS to deploy the web application.
- Create an account if you haven't done so already.
- Go to EC2 and create an instance.
- Choose Ubuntu for the OS.
- Then create the following security key.
 
![key](../images/key.png)
- Add the following rules to your security group.

![gds](../images/gsp.png)

=======================================================================

Then in your GitLab, you will need to go to the settings of your CI/CD, then into 'Variables', and add the following variables:
  - ```REGISTRY_PASS``` where you will put your Docker Hub password
  - ```REGISTRY_USER``` where you will put your Docker Hub username
  - ```SSH_PRIVATE_KEY``` where you will put your EC2 VM's SSH private key

=======================================================================

Finally, go to your GitLab project 'devops_project' and create a .gitlab-ci.yml file. Add the following content to it, and replace {public DNS} with the public DNS of your EC2:

```yml
variables:
    IMAGE_NAME: julesdavoust/devops_project
    IMAGE_TAG: pythonapp-1.1

stages:
    - test
    - build
    - deploy

run_test:
    stage: test
    tags:
        - pythonapp
    before_script:
        - apt-get update && apt-get install -y make
    script:
        - make test

build_image:
    stage: build
    services:
        - name: docker:26.1-dind
          alias: docker
    tags:
        - dockerapp
    variables:
        DOCKER_HOST: tcp://docker:2375
        DOCKER_TLS_CERTDIR: ""
    before_script:
        - docker info
        - echo "$REGISTRY_PASS" | docker login --username "$REGISTRY_USER" --password-stdin
    script:
        - docker build . -t $IMAGE_NAME:$IMAGE_TAG
        - docker push $IMAGE_NAME:$IMAGE_TAG

deploy:
    stage: deploy
    tags:
        - shellapp
    script:
        - cp "$SSH_PRIVATE_KEY" ~/.ssh/id_rsa 
        - chmod 600 ~/.ssh/id_rsa
        - ssh-keyscan -H ec2-13-36-37-239.eu-west-3.compute.amazonaws.com >> ~/.ssh/known_hosts
        - chmod 644 ~/.ssh/known_hosts
        - ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa ubuntu@{DNS public} "
          sudo apt-get update &&
          sudo apt-get install -y docker.io &&
          sudo echo $REGISTRY_USER && echo $REGISTRY_PASS &&
          echo $REGISTRY_PASS | sudo docker login -u $REGISTRY_USER --password-stdin &&
          sudo docker ps -aq | xargs -r sudo docker stop &&
          sudo docker ps -aq | xargs -r sudo docker rm &&
          sudo docker run -d -p 5000:5000 $IMAGE_NAME:$IMAGE_TAG"
```

=======================================================================

When your pipeline has finished executing, connect to: {public IPv4 address of the EC2 VM}:5000

***The configuration steps need to be done once since you don't destroy the VM***
