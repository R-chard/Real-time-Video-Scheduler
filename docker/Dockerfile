FROM fizmath/gpu-opencv:latest

ARG local_uid
ARG local_user

RUN adduser --uid ${local_uid} --gecos "" --disabled-password ${local_user}

WORKDIR /home/${local_user}

USER ${local_user}

ENV PATH="/home/${local_user}/.local/bin:${PATH}"

COPY ./*.py ./
COPY requirements.txt ./
COPY media/* ./media/
COPY dnn_model/* ./dnn_model/
RUN pip install -r requirements.txt
CMD [ "python3", "./controller.py"]