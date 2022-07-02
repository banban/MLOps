FROM ubuntu:20.04

LABEL version="4.0.2" \
      service_name="rapidxai" \
      service_author="Zhibin Liao (AIML)"

# timezone setting
ENV TZ=Australia/Adelaide
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone 
    # && dpkg-reconfigure -f noninteractive tzdata # apt-get install
    
RUN apt-get update \
    && apt-get remove python3.8 -y \
    && apt-get install python3.9-full curl tzdata vim-tiny git iputils-ping -y --no-install-recommends\
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.9 get-pip.py \
    # apt clean
    && rm -f get-pip.py \
    && apt-get remove curl -y \
    && apt-get clean -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    # pip install torch+cpu
    && pip3 install --upgrade pip \
    && pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Create a non-root user
ENV USERNAME rapidxai
ENV HOME /home/$USERNAME
RUN groupadd --gid 5000 $USERNAME \
  && useradd --home-dir $HOME --create-home --uid 5000 \
  --gid 5000 --shell /bin/sh --skel /dev/null $USERNAME
USER $USERNAME

ENV WORKSPACE $HOME/workspace
RUN mkdir -p $WORKSPACE
WORKDIR $WORKSPACE

COPY ../../aiml $WORKSPACE/aiml
COPY ../../swagger_server $WORKSPACE/swagger_server
COPY ../../requirements.txt $WORKSPACE

# pip install and clean
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install mlflow \
    && pip cache purge

EXPOSE 5000
ENTRYPOINT ["python3.9"]
CMD ["-m", "swagger_server"]

