FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt update && apt install -y curl wget iputils-ping

RUN pip3 install -U pip

ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN /root/.cargo/bin/uv pip install --system --no-cache -r requirements.txt

COPY . /app

EXPOSE 8081

CMD ["python3", "main.py"]