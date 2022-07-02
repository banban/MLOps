
# GraphQL vs REST: What's The Difference And When To Use Which?

GraphQL vs REST: they're both popular approaches to developing backend services. 
In this video I show you what the difference is between REST and GraphQL, how to build a basic API using either of these approaches and when to choose one over the other.
Here's the link to the video: https://youtu.be/7ccdWqGgHaM.

For Windows only
```PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Set up environment
```
cd code/api
pip install virtualenv 
virtualenv env 
py -m venv env 
.\env\Scripts\activate 

python.exe -m pip install --upgrade pip 
pip3 install -r requirements.txt 
#run you app.. 
deactivate 
```


## Docker container
```
docker image ls -a
docker run hello-world

cd code/api/rest
docker build --tag rest-docker .
docker run -d -p 5001:5001 rest-docker
docker run rest-docker
# docker stop rest-docker

cd code/api/graphql
docker build --tag graphql-docker .
docker run -d -p 5002:5002 graphql-docker

cd code/ui
docker build --tag ui-docker .
docker run -d -p 5000:5000 ui-docker


cd code/ui
docker build --tag ui-docker .
docker run -d -p 5002:5002 graphql-docker


#stop all containers
docker stop $(docker ps -a -q)
#remove all images
docker rmi -f $(docker images -a -q)

# docker stop graphql-docker
# docker kill graphql-docker
# docker rm graphql-docker
# docker rm rest-docker
# docker system prune --volumes --all
```



EXPOSE 5001

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt


# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "code\api\rest\app:app"]



## Git cheatsheet
```
git help
git pull
git push --progress "origin" master:master
git push --progress "origin2" master:master

git config --global user.name "John Doe"
git config --global user.email johndoe@example.com

git status -s
git difftool
```

## Docker cheatsheet
```
docker image ls -a
docker run hello-world
cd code/api/rest
docker build --tag rest_image .
docker run -d -p 5000:5000 rest_image
# docker stop rest_image
# docker rm rest_image
# docker system prune --volumes --all
```

## Swagger online editor
https://editor.swagger.io/#

## Poetry
https://github.com/python-poetry/poetry/issues/110
You can disable the creation of virtualenvs completely by setting a config option to False:
```
poetry config settings.virtualenvs.create false
```
## Travis CI
https://docs.travis-ci.com/user/languages/python